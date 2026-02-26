from typing import Callable, Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optimistix as optx
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from ..utils import binary_op


def _depth_step(_, z_prev, args):
    x, model = args

    # ── Eq (6): h_t^{(s)} = Λ(z_t^{(s-1)}, x_t) * h_{t-1}^{(s)} + u(z_t^{(s-1)}, x_t)
    lambda_vals = jax.vmap(model.compute_lambda)(z_prev, x)
    u_vals = jax.vmap(model.compute_u)(z_prev, x)
    _, h = jax.lax.associative_scan(binary_op, (lambda_vals, u_vals), axis=0)

    # ── Eq (7): z_t^{(s)} = f_θ(z_t^{(s-1)}, h_{t-1}^{(s)}, x_t)
    # z depends on h_{t-1}^{(s)}, i.e. the hidden state of the
    # PREVIOUS token h_{t-1}^{(s)}, NOT the just-computed h_t^{(s)}.
    h0 = jnp.zeros((1, model.d_state))
    h = jnp.concatenate([h0, h[:-1]], axis=0)

    dz = jax.vmap(model.f_theta)(z_prev, h, x)

    return dz


Normalizer = Callable[[PyTree[Array]], RealScalarLike]


class Implicit(eqx.Module):
    d_model: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    max_time: float = eqx.field(static=True)
    max_iters: int | None = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    norm: Normalizer = eqx.field(static=True)

    adjoint: dfx.AbstractAdjoint
    solver: dfx.AbstractSolver

    f_net: eqx.nn.Linear
    lambda_net: eqx.nn.Linear  # Λ: maps (z, x) → decay factor (diagonal)
    u_net: eqx.nn.Linear  # u: maps (z, x) → input state
    out_net: eqx.nn.Linear  # Output: maps (z, h) → y

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
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt = dt
        self.max_time = max_time
        self.max_iters = max_iters
        self.solver = solver if solver is not None else dfx.Tsit5()
        self.adjoint = adjoint if adjoint is not None else dfx.ImplicitAdjoint()
        self.rtol = rtol
        self.atol = atol
        self.norm = norm

        keys = jrandom.split(key, 4)

        self.f_net = eqx.nn.Linear(
            d_inner + d_state + d_model,
            d_inner,
            key=keys[0],
        )

        self.lambda_net = eqx.nn.Linear(d_inner + d_model, d_state, key=keys[1])
        self.u_net = eqx.nn.Linear(d_inner + d_model, d_state, key=keys[2])
        self.out_net = eqx.nn.Linear(d_inner, d_model, key=keys[3])

    def compute_lambda(
        self, z: Float[Array, " d_inner"], x: Float[Array, " d_model"]
    ) -> Float[Array, " d_state"]:
        zx = jnp.concatenate([z, x], axis=-1)
        # Sigmoid to keep in (0, 1) for stability
        return jax.nn.sigmoid(self.lambda_net(zx))

    def compute_u(
        self, z: Float[Array, " d_inner"], x: Float[Array, " d_model"]
    ) -> Float[Array, " d_state"]:
        zx = jnp.concatenate([z, x], axis=-1)
        return self.u_net(zx)

    def f_theta(
        self,
        z: Float[Array, " d_inner"],
        h: Float[Array, " d_state"],
        x: Float[Array, " d_model"],
    ) -> Float[Array, " d_inner"]:
        """f_θ(z, h, x) → z - the implicit function."""
        zhx = jnp.concatenate([z, h, x])
        dz = jax.nn.silu(self.f_net(zhx)) - z
        return dz

    def get_z(
        self,
        x: Float[Array, "seq d_model"],
        t0: float = 0,
        t1: float | None = None,
        num_points: int = 100,
    ) -> Float[Array, "seq d_model"]:
        seq_len, _ = x.shape
        t1 = t1 if t1 is not None else self.max_time

        z0 = jnp.zeros((seq_len, self.d_inner))
        terms = dfx.ODETerm(_depth_step)
        sol = dfx.diffeqsolve(
            terms,
            self.solver,
            t0=0.0,
            t1=t1,
            dt0=self.dt,
            y0=z0,
            args=(x, self),
            max_steps=self.max_iters,
            saveat=dfx.SaveAt(ts=jnp.linspace(t0, t1, num_points)),
        )

        return sol.ts, sol.ys

    def sequential(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq d_model"]:
        def scan_fn(h, x_t):

            def depth_step(_, z_prev, args):
                h, x, model = args
                z_s = model.f_theta(z_prev, h, x)

                return z_s

            z0 = jnp.zeros((self.d_inner,))
            cond_fn = dfx.steady_state_event(rtol=1e-4, atol=1e-3, norm=optx.rms_norm)
            event = dfx.Event(cond_fn)
            terms = dfx.ODETerm(depth_step)
            sol = dfx.diffeqsolve(
                terms,
                self.solver,
                t0=0.0,
                t1=self.max_time,
                dt0=self.dt,
                y0=z0,
                args=(h, x_t, self),
                max_steps=self.max_iters,
                event=event,
                adjoint=self.adjoint,
            )
            z_star = sol.ys[-1]

            # Compute new hidden state
            lambda_val = self.compute_lambda(z_star, x_t)  # (d_state,)
            u_val = self.compute_u(z_star, x_t)  # (d_state,)

            h_new = lambda_val * h + u_val

            y = self.out_net(z_star)

            return h_new, y

        h_init = jnp.zeros(self.d_state)
        _, y = jax.lax.scan(scan_fn, h_init, x)

        return y

    def __call__(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq d_model"]:
        seq_len, _ = x.shape

        z0 = jnp.zeros((seq_len, self.d_inner))
        cond_fn = dfx.steady_state_event(rtol=1e-4, atol=1e-3, norm=optx.rms_norm)
        event = dfx.Event(cond_fn)
        terms = dfx.ODETerm(_depth_step)
        sol = dfx.diffeqsolve(
            terms,
            self.solver,
            t0=0.0,
            t1=self.max_time,
            dt0=self.dt,
            y0=z0,
            args=(x, self),
            max_steps=self.max_iters,
            event=event,
            adjoint=self.adjoint,
        )
        z_star = sol.ys[-1]
        y_star = jax.vmap(self.out_net)(z_star)

        return y_star
