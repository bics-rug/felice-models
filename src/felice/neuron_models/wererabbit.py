from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, DTypeLike, Float


class WereRabbit(eqx.Module):
    r"""
    WereRabbit Neuron Model

    The WereRabbit model implements a predator-prey dynamic with bistable 
    switching behavior controlled by a "moon phase" parameter $z$.
    
    The dynamics are governed by:

    $$
    \begin{align}
        z &= tanh(\rho (u-v)) \\
        \frac{du}{dt} &= z - z \alpha e^{\beta v} [1 + \gamma (0.5 - u)] - \sigma \\
        \frac{dv}{dt} &= -z - z \alpha e^{\beta u} [1 + \gamma (0.5 - v)] - \sigma
    \end{align}
    $$

    where $z$ represents the "moon phase" that switches the predator-prey roles.
    
    Attributes:
        alpha: Current scaling parameter $\alpha = I_{n0}/I_{bias}$ (default: 0.0129)
        beta: Exponential slope $\beta = \kappa/U_t$ (default: 15.6)
        gamma: Coupling parameter $\gamma = 26e^{-2}$
        rho: Steepness of the tanh function $\rho$ (default: 5)
        sigma: Fixpoint distance scaling $\sigma$ (default: 0.6)

        rtol: Relative tolerance for the spiking fixpoint calculation.
        atol: Absolute tolerance for the spiking fixpoint calculation.

        weight_u: Input weight for the predator.
        weight_v: Input weight for the prey.
    """

    dtype: DTypeLike = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)

    alpha: float = eqx.field(static=True)  # I_n0 / I_bias ratio
    beta: float = eqx.field(static=True)  # k / U_t (inverse thermal scale)
    gamma: float = eqx.field(static=True)  # coupling coefficient
    rho: float = eqx.field(static=True)  # tanh steepness
    sigma: float = eqx.field(static=True)  # bias scaling (s * I_bias)

    def __init__(
        self,
        *,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        alpha: float = 0.0129,
        beta: float = 15.6,
        gamma: float = 0.26,
        rho: float = 5.0,
        sigma: float = 0.6,
        dtype: DTypeLike = jnp.float32,
    ):
        r"""Initialize the WereRabbit neuron model.

        Args:
            rtol: Relative tolerance for the spiking fixpoint calculation.
            atol: Absolute tolerance for the spiking fixpoint calculation.
            alpha: Current scaling parameter $\alpha = I_{n0}/I_{bias}$ (default: 0.0129)
            beta: Exponential slope $\beta = \kappa/U_t$ (default: 15.6)
            gamma: Coupling parameter $\gamma = 26e^{-2}$
            rho: Steepness of the tanh function $\rho$ (default: 5)
            sigma: Fixpoint distance scaling $\sigma$ (default: 0.6)
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

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 2"]:
        """Initialize the neuron state variables.

        Args:
            n_neurons: Number of neurons to initialize.

        Returns:
            Initial state array of shape (neurons, 3) containing [u, v, has_spiked],
            where u and v are the predator/prey membrane voltages, has_spiked is a
            variable that is 1 whenever the neuron spike and 0 otherwise .
        """
        x1 = jnp.zeros((n_neurons,), dtype=self.dtype)
        x2 = jnp.zeros((n_neurons,), dtype=self.dtype)
        return jnp.stack([x1, x2], axis=1)

    def vector_field(self, y: Float[Array, "neurons 2"]) -> Float[Array, "neurons 2"]:
        """Compute vector field of the neuron state variables.

        This implements the WereRabbit dynamics

            - du/dt: Predator dynamics
            - dv/dt: WerePrey dynamics

        Args:
            y: State array of shape (neurons, 2) containing [u, v].

        Returns:
            Time derivatives of shape (neurons, 2) containing [du/dt, dv/dt].
        """
        u = y[:, 0]
        v = y[:, 1]

        z = jax.nn.tanh(self.rho * (u - v))
        du = (
            z * (1 - self.alpha * jnp.exp(self.beta * v) * (1 + self.gamma * (0.5 - u)))
            - self.sigma
        )
        dv = (
            z
            * (-1 + self.alpha * jnp.exp(self.beta * u) * (1 + self.gamma * (0.5 - v)))
            - self.sigma
        )

        dv = jnp.where(jnp.allclose(z, 0.0), dv * jnp.sign(v), dv)
        du = jnp.where(jnp.allclose(z, 0.0), du * jnp.sign(u), du)

        return jnp.stack([du, dv], axis=1)

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
            y: State array of shape (neurons, 3) containing [u, v, has_spiked].
            args: Additional arguments (unused but required by framework).

        Returns:
            Time derivatives of shape (neurons, 3) containing [du/dt, dv/dt, 0].
        """
        dxdt = self.vector_field(y)

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
            return _atol + _rtol * _norm(y[:-1]) - _norm(vf[:-1])

        base_cond = calculate_norm(vf, y)

        return base_cond
