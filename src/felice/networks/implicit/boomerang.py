from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray

from ..utils import binary_op
from .base import Implicit, Normalizer


class ImplicitBoomerang(eqx.Module):
    d_model: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)
    max_iters: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    with_thr: bool = eqx.field(static=True)
    debug: bool = eqx.field(static=True)

    f_net: eqx.nn.Linear
    lambda_net: eqx.nn.Linear  # Λ: maps (z, x) → decay factor (diagonal)
    u_net: eqx.nn.Linear  # u: maps (z, x) → input state
    out_net: eqx.nn.Linear  # Output: maps (z, h) → y

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: int = 32,
        max_iters: int = 20,
        tol: float = 1e-5,
        dt: float = 1e-3,
        with_thr: bool = True,
        debug: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.max_iters = max_iters
        self.tol = tol
        self.dt = dt
        self.with_thr = with_thr
        self.debug = debug

        keys = jrandom.split(key, 6)

        self.f_net = eqx.nn.Linear(
            d_state + d_model,
            d_inner,
            key=keys[1],
        )

        self.lambda_net = eqx.nn.Linear(d_inner + d_model, d_state, key=keys[2])
        self.u_net = eqx.nn.Linear(d_inner + d_model, d_state, key=keys[3])
        self.out_net = eqx.nn.Linear(d_inner + d_state, d_model, key=keys[4])

    def compute_lambda(
        self, z: Float[Array, " d_inner"], x: Float[Array, " d_model"]
    ) -> Float[Array, " d_state"]:
        zx = jnp.concatenate([z, x], axis=-1)
        # Sigmoid to keep in (0, 1) for stability
        return jax.nn.sigmoid(jax.vmap(self.lambda_net)(zx))

    def compute_u(
        self, z: Float[Array, " d_inner"], x: Float[Array, " d_model"]
    ) -> Float[Array, " d_state"]:
        zx = jnp.concatenate([z, x], axis=-1)
        return jax.vmap(self.u_net)(zx)

    def f_theta(
        self,
        z: Float[Array, " d_inner"],
        h: Float[Array, " d_state"],
        x: Float[Array, " d_model"],
    ) -> Float[Array, " d_inner"]:
        """f_θ(z, h, x) → z - the implicit function."""
        hx = jnp.concatenate([h, x])

        alpha_sigma = jnp.split(self.f_net(hx), 2)

        rho = 30.0
        alpha = jax.nn.sigmoid(alpha_sigma[0])
        beta = 15.6
        gamma = 0.26
        sigma = jax.nn.sigmoid(alpha_sigma[1])
        u, v = jnp.split(z, 2)

        def true_fn(u, v):
            return jax.nn.tanh(rho * (v - u))

        def false_fn(u, v):
            return jnp.ones_like(u)

        thresh = jax.lax.cond(self.with_thr, true_fn, false_fn, u, v)
        du = (1 - alpha * jnp.exp(beta * v) * (1 - gamma * (0.3 - u))) + sigma * thresh
        dv = (-1 + alpha * jnp.exp(beta * u) * (1 + gamma * (0.3 - v))) + sigma * thresh
        dz = jnp.concat([du, dv])
        z = z + self.dt * dz

        # kk = 0.68  # fixed by tech
        # Ut = 0.025  # temperature dependent
        # I_r0 = 0.9
        # x1, x2 = jnp.split(z, 2)

        # alpha = 0.000129 + (0.0129 - 0.000129) * jax.nn.sigmoid(
        #     alpha_sigma[0]
        # )  # The circuit will get directly the current
        # Ia = jax.nn.tanh(alpha_sigma[1]) * 0.6

        # x3 = 1  # (np.tanh(20 * (x2 - x1)))  #smoother transition
        # dx1 = 2.3 * (
        #     1 - (alpha * jnp.exp(kk * x2 / Ut)) * (1 - I_r0 * (0.3 - x1)) + Ia * x3
        # )
        # dx2 = 2.3 * (
        #     -1 + (alpha * jnp.exp(kk * x1 / Ut)) * (1 + I_r0 * (0.3 - x2)) + Ia * x3
        # )
        # dz = jnp.concat([dx1, dx2])
        # z = z + self.dt * dz

        return z

    def get_z(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq d_model"]:
        seq_len, _ = x.shape

        def body_fn(state, _):
            z = state

            lambda_vals = self.compute_lambda(z, x)
            u_vals = self.compute_u(z, x)

            _, h = jax.lax.associative_scan(binary_op, (lambda_vals, u_vals), axis=0)
            h_prev = jnp.concatenate([jnp.zeros((1, self.d_state)), h[:-1]], axis=0)

            z_new = jax.vmap(self.f_theta)(z, h_prev, x)

            return z_new, z_new

        # Initialize
        z_init = jnp.zeros((seq_len, self.d_inner))

        # Run fixed-point iteration
        _, z = jax.lax.scan(
            body_fn,
            z_init,
            None,
            length=self.max_iters,
        )

        return z

    def sequential(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq d_model"]:

        def scan_fn(h, x_t):
            z_init = jnp.zeros((self.d_inner,))
            z_prev_init = jnp.full((self.d_inner,), jnp.inf)

            def cond_fn(state):
                i, z, z_prev = state
                converged = (
                    jnp.linalg.norm(z - z_prev) / jnp.linalg.norm(z_prev) < 0.001
                )
                return (i < self.max_iters) & ~converged

            def body_fn(state):
                i, z, _ = state
                z_new = self.f_theta(z, h, x_t)
                return (i + 1, z_new, z)

            # Run fixed-point iteration
            if self.debug:

                def scan_fn(state, _):
                    new_state = body_fn(state)
                    return new_state, new_state[1]

                _, z_star_debug = jax.lax.scan(
                    scan_fn, (0, z_init, z_prev_init), None, length=self.max_iters
                )
                z_star = z_star_debug[-1]
            else:
                _, z_star, _ = eqx.internal.while_loop(
                    cond_fn,
                    body_fn,
                    (0, z_init, z_prev_init),
                    max_steps=self.max_iters,
                    kind="bounded",
                )

            # Compute new hidden state
            lambda_val = self.compute_lambda(
                z_star[jnp.newaxis, :], x_t[jnp.newaxis, :]
            )[0]  # (d_state,)
            u_val = self.compute_u(z_star[jnp.newaxis, :], x_t[jnp.newaxis, :])[
                0
            ]  # (d_state,)

            h_new = lambda_val * h + u_val

            zh = jnp.concatenate([z_star, h_new])
            y = self.out_net(zh)

            return h_new, (y, z_star_debug)

        h_init = jnp.zeros(self.d_state)
        _, (y, z_star) = jax.lax.scan(scan_fn, h_init, x)

        if self.debug:
            return y, z_star
        else:
            return y

    def __call__(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq d_model"]:
        seq_len, _ = x.shape

        def cond_fn(state):
            i, z, z_prev = state
            converged = jnp.linalg.norm(z - z_prev) / jnp.linalg.norm(z_prev) < 0.001
            return (i < self.max_iters) & ~converged

        def body_fn(state):
            i, z, _ = state

            lambda_vals = self.compute_lambda(z, x)
            u_vals = self.compute_u(z, x)

            _, h = jax.lax.associative_scan(binary_op, (lambda_vals, u_vals), axis=0)
            h_prev = jnp.concatenate([jnp.zeros((1, self.d_state)), h[:-1]], axis=0)

            z_new = jax.vmap(self.f_theta)(z, h_prev, x)

            return (i + 1, z_new, z)

        # Initialize
        z_init = jnp.zeros((seq_len, self.d_inner))
        z_prev_init = jnp.full((seq_len, self.d_inner), jnp.inf)

        # Run fixed-point iteration
        if self.debug:

            def scan_fn(state, _):
                new_state = body_fn(state)
                return new_state, new_state[1]

            _, z_star_debug = jax.lax.scan(
                scan_fn, (0, z_init, z_prev_init), None, length=self.max_iters
            )
            z_star = z_star_debug[-1]
        else:
            _, z_star, _ = eqx.internal.while_loop(
                cond_fn,
                body_fn,
                (0, z_init, z_prev_init),
                max_steps=self.max_iters,
                kind="bounded",
            )

        lambda_vals = self.compute_lambda(z_star, x)
        u_vals = self.compute_u(z_star, x)
        _, h_star = jax.lax.associative_scan(binary_op, (lambda_vals, u_vals), axis=0)

        zh = jnp.concatenate([z_star, h_star], axis=-1)
        y_star = jax.vmap(self.out_net)(zh)

        if self.debug:
            return y_star, z_star_debug
        else:
            return y_star


class Boomerang(Implicit):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: int = 32,
        dt: float = 1.0,
        max_iters: int | None = 100,
        max_time: float = 100,
        solver: Optional[dfx.AbstractSolver] = None,
        adjoint: Optional[dfx.AbstractAdjoint] = None,
        rtol: float = 1e-4,
        atol: float = 1e-3,
        norm: Normalizer = optx.rms_norm,
        *,
        key: PRNGKeyArray,
    ):
        keys = jrandom.split(key)
        super(Boomerang, self).__init__(
            d_model=d_model,
            d_state=d_state,
            d_inner=d_inner,
            dt=dt,
            max_iters=max_iters,
            max_time=max_time,
            solver=solver,
            adjoint=adjoint,
            rtol=rtol,
            atol=atol,
            norm=norm,
            key=keys[0],
        )

        self.f_net = eqx.nn.Linear(
            d_state + d_model,
            d_inner,
            key=keys[1],
        )

    def f_theta(
        self,
        z: Float[Array, " d_inner"],
        h: Float[Array, " d_state"],
        x: Float[Array, " d_model"],
    ) -> Float[Array, " d_inner"]:
        """f_θ(z, h, x) → z - the implicit function."""
        hx = jnp.concatenate([h, x])

        alpha_sigma = jnp.split(self.f_net(hx), 2)

        kk = 0.68  # fixed by tech
        Ut = 0.025  # temperature dependent
        I_r0 = 0.9
        x1, x2 = jnp.split(z, 2)

        alpha = 0.000129 + (0.0129 - 0.000129) * jax.nn.sigmoid(
            alpha_sigma[0]
        )  # The circuit will get directly the current
        Ia = jax.nn.tanh(alpha_sigma[1]) * 0.6

        x3 = 1  # (np.tanh(20 * (x2 - x1)))  #smoother transition
        dx1 = 2.3 * (
            1 - (alpha * jnp.exp(kk * x2 / Ut)) * (1 - I_r0 * (0.3 - x1)) + Ia * x3
        )
        dx2 = 2.3 * (
            -1 + (alpha * jnp.exp(kk * x1 / Ut)) * (1 + I_r0 * (0.3 - x2)) + Ia * x3
        )
        dz = jnp.concat([dx1, dx2])

        return dz
