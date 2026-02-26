from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import flatten_util, lax
from jaxtyping import Array, Float, PRNGKeyArray


@jax.jit
def binary_op(
    left: Tuple[Array, Array],
    right: Tuple[Array, Array],
) -> Tuple[Array, Array]:
    """
    (a1, b1) ∘ (a2, b2) = (a1·a2, a2·b1 + b2)
    """
    a1, b1 = left
    a2, b2 = right
    return (a1 * a2, a2 * b1 + b2)


class ImplicitSNN(eqx.Module):
    d_model: int
    d_state: int
    d_latent: int
    max_iters: int
    tol: float

    embedding: eqx.nn.Embedding
    f_net: eqx.Module  # f_θ: maps (z, h, x) → z
    f_net2: eqx.Module  # f_θ: maps (z, h, x) → z
    lambda_net: eqx.Module  # Λ: maps (z, x) → decay factor (diagonal)
    u_net: eqx.Module  # u: maps (z, x) → input contribution
    out_net: eqx.nn.Linear  # Output: maps (z, h) → y

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_state: int = 16,
        d_latent: int = 32,
        max_iters: int = 20,
        tol: float = 1e-5,
        *,
        key: PRNGKeyArray,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_latent = d_latent
        self.max_iters = max_iters
        self.tol = tol

        keys = jrandom.split(key, 6)

        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])

        # f_θ(z, h, x) → z
        # Input: z (d_latent) + h (d_state) + x (d_model)
        self.f_net = eqx.nn.Linear(
            d_state + d_model,
            d_latent // 2,
            key=keys[1],
        )

        self.f_net2 = eqx.nn.Linear(
            d_state + d_model,
            d_latent // 2,
            key=keys[5],
        )

        # Λ(z, x) → (d_state,) decay factors
        self.lambda_net = eqx.nn.Linear(d_latent + d_model, d_state, key=keys[2])

        # u(z, x) → (d_state,) input contribution
        self.u_net = eqx.nn.Linear(d_latent + d_model, d_state, key=keys[3])

        # Output projection
        self.out_net = eqx.nn.Linear(d_latent + d_state, d_model, key=keys[4])

    def compute_lambda(
        self, z: Float[Array, " d_latent"], x: Float[Array, " d_model"]
    ) -> Float[Array, " d_state"]:
        """Λ(z, x) - the decay/retention factor."""
        zx = jnp.concatenate([z, x], axis=-1)
        # Sigmoid to keep in (0, 1) for stability
        return jax.nn.sigmoid(jax.vmap(self.lambda_net)(zx))

    def compute_u(
        self, z: Float[Array, " d_latent"], x: Float[Array, " d_model"]
    ) -> Float[Array, " d_state"]:
        """u(z, x) - the input contribution."""
        zx = jnp.concatenate([z, x], axis=-1)
        return jax.vmap(self.u_net)(zx)

    # TODO: Change for diffrax
    def f_theta(
        self,
        z: Float[Array, " d_latent"],
        h: Float[Array, " d_state"],
        x: Float[Array, " d_model"],
    ) -> Float[Array, " d_latent"]:
        """f_θ(z, h, x) → z - the implicit function."""
        hx = jnp.concatenate([h, x])

        rho = 30.0
        alpha = jax.nn.sigmoid(self.f_net(hx))
        beta = 15.6
        gamma = 0.26
        sigma = jax.nn.sigmoid(self.f_net2(hx))
        u, v = jnp.split(z, 2)

        thresh = jax.nn.tanh(rho * (v - u))
        du = (1 - alpha * jnp.exp(beta * v) * (1 - gamma * (0.3 - u))) + sigma * thresh
        dv = (-1 + alpha * jnp.exp(beta * u) * (1 + gamma * (0.3 - v))) + sigma * thresh
        dz = jnp.concat([du, dv])
        z = z + 0.001 * dz

        return z

    # ==================== Parallel mode (training) ====================

    def parallel_scan_h(
        self,
        lambda_vals: Float[Array, "seq d_state"],
        u_vals: Float[Array, "seq d_state"],
    ) -> Float[Array, "seq d_state"]:
        """
        Compute h sequence via parallel scan.

        h_t = λ_t · h_{t-1} + u_t

        This is the standard SSM recurrence, parallelizable via associative scan.
        """

        # Elements: (λ_t, u_t)
        # After scan: (cumulative λ, h_t)
        _, h = lax.associative_scan(binary_op, (lambda_vals, u_vals), axis=0)

        return h

    def debug(self, x):
        x = jax.vmap(self.embedding)(x)  # (seq, d_model)
        seq_len = x.shape[0]

        def body_fn(state, _):
            z, _ = state

            # Compute λ, u
            lambda_vals = self.compute_lambda(z, x)
            u_vals = self.compute_u(z, x)

            # Parallel scan for h
            h = self.parallel_scan_h(lambda_vals, u_vals)

            # Shift h
            h_prev = jnp.concatenate([jnp.zeros((1, self.d_state)), h[:-1]], axis=0)

            # Update z
            z_new = jax.vmap(self.f_theta)(z, h_prev, x)

            return (z_new, z), z_new

        # Initialize
        z_init = jnp.zeros((seq_len, self.d_latent))
        z_prev_init = jnp.full((seq_len, self.d_latent), jnp.inf)

        _, z_star = jax.lax.scan(
            body_fn, (z_init, z_prev_init), None, length=self.max_iters
        )

        return z_star

    def forward_parallel(
        self,
        x: Float[Array, " seq"],
    ) -> Float[Array, "seq d_model"]:
        """
        Parallel forward pass for training.

        Algorithm:
        1. Initialize z for all positions
        2. Iterate until convergence:
           a. Compute λ, u from current z (parallel over positions)
           b. Compute h via parallel scan
           c. Update z = f_θ(z, h_shifted, x) (parallel over positions)
        3. Compute output from final z, h

        Note: h_shifted means h_{t-1} for position t, so we shift h right.
        """
        x = jax.vmap(self.embedding)(x)  # (seq, d_model)
        seq_len = x.shape[0]

        def cond_fn(state):
            i, z, z_prev = state
            converged = jnp.linalg.norm(z - z_prev) < self.tol
            return (i < self.max_iters) & ~converged

        def body_fn(state):
            i, z, _ = state

            # Compute λ, u
            lambda_vals = self.compute_lambda(z, x)
            u_vals = self.compute_u(z, x)

            # Parallel scan for h
            h = self.parallel_scan_h(lambda_vals, u_vals)

            # Shift h
            h_prev = jnp.concatenate([jnp.zeros((1, self.d_state)), h[:-1]], axis=0)

            # Update z
            z_new = jax.vmap(self.f_theta)(z, h_prev, x)

            return (i + 1, z_new, z)

        # Initialize
        z_init = jnp.zeros((seq_len, self.d_latent))
        z_prev_init = jnp.full((seq_len, self.d_latent), jnp.inf)

        # Run fixed-point iteration
        _, z_star, _ = eqx.internal.while_loop(
            cond_fn,
            body_fn,
            (0, z_init, z_prev_init),
            max_steps=self.max_iters,
            kind="bounded",
        )

        # Final h computation with converged z
        lambda_vals = self.compute_lambda(z_star, x)
        u_vals = self.compute_u(z_star, x)
        h_star = self.parallel_scan_h(lambda_vals, u_vals)

        # Output
        zh = jnp.concatenate([z_star, h_star], axis=-1)
        y_star = jax.vmap(self.out_net)(zh)

        return y_star

    # ==================== Sequential mode (inference) ====================

    def find_fixed_point(
        self,
        h_prev: Float[Array, " d_state"],
        x: Float[Array, " d_model"],
    ) -> Float[Array, " d_latent"]:
        """
        Find z* such that z* = f_θ(z*, h_prev, x)
        using fixed-point iteration.
        """

        def cond_fn(state):
            i, z, z_prev = state
            converged = jnp.linalg.norm(z - z_prev) < self.tol
            return (i < self.max_iters) & ~converged

        def body_fn(state):
            i, z, _ = state
            z_new = self.f_theta(z, h_prev, x)
            return (i + 1, z_new, z)

        # Initialize z
        z_init = jnp.zeros(self.d_latent)
        z_prev_init = jnp.full(self.d_latent, jnp.inf)

        _, z_star, _ = eqx.internal.while_loop(
            cond_fn,
            body_fn,
            (0, z_init, z_prev_init),
            max_steps=self.max_iters,
            kind="bounded",
        )

        return z_star

    def step(
        self,
        h_prev: Float[Array, " d_state"],
        x: Float[Array, " d_model"],
    ) -> Tuple[Float[Array, " d_state"], Float[Array, " d_model"]]:
        """
        Single step of the implicit SSM.

        1. Find z* via fixed-point iteration
        2. Compute h* = Λ(z*, x) · h_prev + u(z*, x)
        3. Compute output y
        """
        # Find fixed point z*
        z_star = self.find_fixed_point(h_prev, x)

        # Compute new hidden state
        lambda_val = self.compute_lambda(z_star[jnp.newaxis, :], x[jnp.newaxis, :])[
            0
        ]  # (d_state,)
        u_val = self.compute_u(z_star[jnp.newaxis, :], x[jnp.newaxis, :])[
            0
        ]  # (d_state,)

        h_new = lambda_val * h_prev + u_val

        # Compute output
        zh = jnp.concatenate([z_star, h_new])
        y = self.out_net(zh)

        return h_new, y

    def forward_sequential(
        self,
        x: Float[Array, " seq"],
    ) -> Float[Array, "seq d_model"]:
        """
        Sequential forward pass for inference.

        Processes one token at a time, maintaining state.
        """
        x = jax.vmap(self.embedding)(x)

        def scan_fn(h, x_t):
            h_new, y_t = self.step(h, x_t)
            return h_new, y_t

        h_init = jnp.zeros(self.d_state)
        _, y = lax.scan(scan_fn, h_init, x)

        return y

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        mode: str = "parallel",
    ) -> Float[Array, "seq d_model"]:
        """
        Forward pass over sequence.
        Note: This is inherently sequential due to the implicit nature.
        """

        if mode == "parallel":
            return self.forward_parallel(x)
        else:
            return self.forward_sequential(x)


if __name__ == "__main__":
    import time

    key = jrandom.key(42)
    keys = jrandom.split(key, 2)

    d_model, d_state, d_latent = 64, 16, 32
    seq_len = 1080

    model = ImplicitSNN(
        vocab_size=16,
        d_model=d_model,
        d_state=d_state,
        d_latent=d_latent,
        max_iters=10,
        key=keys[0],
    )

    # Test input
    x = jrandom.randint(keys[1], (seq_len,), 0, 15)

    # Test parallel mode
    print("Testing parallel mode...")
    y_parallel = model(x, mode="parallel")
    print(f"  Input: {x.shape}, Output: {y_parallel.shape}")

    # Test sequential mode
    print("\nTesting sequential mode...")
    y_sequential = model(x, mode="sequential")
    print(f"  Input: {x.shape}, Output: {y_sequential.shape}")

    # Compare outputs (should be close but not exact due to different convergence paths)
    diff = jnp.linalg.norm(y_parallel - y_sequential) / jnp.linalg.norm(y_sequential)
    print(f"\nRelative difference: {diff:.6f}")

    # Test gradients in parallel mode
    def loss_fn(model, x, mode):
        return jnp.mean(model(x, mode=mode) ** 2)

    print("\nTesting gradients (parallel mode)...")
    grads_par = eqx.filter_grad(loss_fn)(model, x, "parallel")
    grads_par = flatten_util.ravel_pytree(grads_par)[0]
    print("\nTesting gradients (sequential mode)...")
    grads_seq = eqx.filter_grad(loss_fn)(model, x, "sequential")
    grads_seq = flatten_util.ravel_pytree(grads_seq)[0]

    grad_diff = jnp.linalg.norm(grads_par - grads_seq) / jnp.linalg.norm(grads_seq)
    print("  Gradients computed successfully!")
    print(f"\nRelative difference: {grad_diff:.6f}")

    # Benchmark
    print("\nBenchmarking...")

    # JIT compile
    forward_parallel_jit = eqx.filter_jit(lambda m, x: m(x, mode="parallel"))
    forward_sequential_jit = eqx.filter_jit(lambda m, x: m(x, mode="sequential"))

    # Warmup
    _ = forward_parallel_jit(model, x)
    _ = forward_sequential_jit(model, x)

    # Time parallel
    start = time.time()
    for _ in range(100):
        _ = forward_parallel_jit(model, x).block_until_ready()
    parallel_time = (time.time() - start) / 100

    # Time sequential
    start = time.time()
    for _ in range(100):
        _ = forward_sequential_jit(model, x).block_until_ready()
    sequential_time = (time.time() - start) / 100

    print(f"  Parallel:   {parallel_time * 1000:.3f} ms")
    print(f"  Sequential: {sequential_time * 1000:.3f} ms")
    print(f"  Speedup:    {sequential_time / parallel_time:.2f}x")
