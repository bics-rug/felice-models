from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, DTypeLike, Float


class Boomerang(eqx.Module):
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)

    u0: float = eqx.field(static=True)
    v0: float = eqx.field(static=True)

    alpha: float = eqx.field(static=True)  # I_n0 / I_bias ratio
    beta: float = eqx.field(static=True)  # k / U_t (inverse thermal scale)
    gamma: float = eqx.field(static=True)  # coupling coefficient
    rho: float = eqx.field(static=True)  # tanh steepness
    sigma: float = eqx.field(static=True)  # bias scaling (s * I_bias)

    dtype: DTypeLike = eqx.field(static=True)

    def __init__(
        self,
        *,
        atol: float = 1e-6,
        rtol: float = 1e-4,
        alpha: float = 0.0129,
        beta: float = 15.6,
        gamma: float = 0.26,
        rho: float = 30.0,
        sigma: float = 0.6,
        dtype: DTypeLike = jnp.float32,
    ):
        r"""Initialize the WereRabbit neuron model.

        Args:
            key: JAX random key for weight initialization.
            n_neurons: Number of neurons in this layer.
            in_size: Number of input connections (excluding recurrent connections).
            wmask: Binary mask defining connectivity pattern of shape (in_plus_neurons, neurons).
            rtol: Relative tolerance for the spiking fixpoint calculation.
            atol: Absolute tolerance for the spiking fixpoint calculation.
            alpha: Current scaling parameter $\alpha = I_{n0}/I_{bias}$ (default: 0.0129)
            beta: Exponential slope $\beta = \kappa/U_t$ (default: 15.6)
            gamma: Coupling parameter $\gamma = 26e^{-2}$
            rho: Steepness of the tanh function $\rho$ (default: 5)
            sigma: Fixpoint distance scaling $\sigma$ (default: 0.6)
            wlim: Limit for weight initialization. If None, uses init_weights.
            wmean: Mean value for weight initialization.
            init_weights: Optional initial weight values. If None, weights are randomly initialized.
            fan_in_mode: Mode for fan-in based weight initialization ('sqrt', 'linear').
            dtype: Data type for arrays (default: float32).
        """
        self.dtype = dtype

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

        self.rtol = rtol
        self.atol = atol

        def fn(y, _):
            return self.vector_field(y[0], y[1])

        solver: optx.AbstractRootFinder = optx.Newton(rtol=1e-8, atol=1e-8)
        y0 = (jnp.array(0.3), jnp.array(0.3))
        u0, v0 = optx.root_find(fn, solver, y0).value
        self.u0 = u0.item()
        self.v0 = v0.item()

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        """Initialize the neuron state variables.

        Args:
            n_neurons: Number of neurons to initialize.

        Returns:
            Initial state array of shape (neurons, 3) containing [u, v],
            where u and v are the predator/prey membrane voltages.
        """

        u = jnp.full((n_neurons,), self.u0, dtype=self.dtype)
        v = jnp.full((n_neurons,), self.v0, dtype=self.dtype)
        x = jnp.stack([u, v], axis=1)
        return x

    def vector_field(
        self, u: Float[Array, "..."], v: Float[Array, "..."]
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        sigma = self.sigma
        rho = self.rho

        z = jax.nn.tanh(rho * (v - u))
        du = (1 - alpha * jnp.exp(beta * v) * (1 - gamma * (0.3 - u))) + sigma * z
        dv = (-1 + alpha * jnp.exp(beta * u) * (1 + gamma * (0.3 - v))) + sigma * z

        return du, dv

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 2"]:
        """Compute time derivatives of the neuron state variables.

        This implements the WereRabbit dynamics

            - du/dt: Predator dynamics
            - dv/dt: WerePrey dynamics

        Args:
            t: Current simulation time (unused but required by framework).
            y: State array of shape (neurons, 2) containing [u, v].
            args: Additional arguments (unused but required by framework).

        Returns:
            Time derivatives of shape (neurons, 2) containing [du/dt, dv/dt].
        """
        u = y[:, 0]
        v = y[:, 1]

        du, dv = self.vector_field(u, v)
        dxdt = jnp.stack([du, dv], axis=1)

        return dxdt

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 2"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, " neurons"]:
        """Compute spike condition for event detection.

        A spike is triggered when the system reach to a fixpoint.

        INFO:
            `has_spiked` is use to the system don't detect a continuos
            spike when reach a fixpoint.

        Args:
            t: Current simulation time (unused but required by the framework).
            y: State array of shape (neurons, 3) containing [u, v, has_spiked].
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Spike condition array of shape (neurons,). Positive values indicate spike.
        """
        _atol = self.atol
        _rtol = self.rtol
        _norm = optx.rms_norm

        vf = self.dynamics(t, y, {})

        @jax.vmap
        def calculate_norm(vf, y):
            return _atol + _rtol * _norm(y) - _norm(vf)

        base_cond = calculate_norm(vf, y).repeat(2)

        return base_cond
