from typing import Any, Dict, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float


class FHNRS(eqx.Module):
    r"""FitzHugh-Nagumo neuron model

    Model for FitzHugh-Nagumo neuron, with a hardware implementation proposed by
    Ribar-Sepulchre. This implementation uses a dual-timescale dynamics with fast
    and slow currents to produce oscillatory spiking behavior.

    The dynamics are governed by:

    $$
    \begin{align}
        C\frac{dv}{dt} &= I_{app} - I_{passive} - I_{fast} - I_{slow} \\
        \frac{dv_{slow}}{dt} &= \frac{v - v_{slow}}{\tau_{slow}} \\
        \frac{dI_{app}}{dt} &= -\frac{I_{app}}{\tau_{syn}}
    \end{align}
    $$

    where the currents are:

    - $I_{passive} = g_{max}(v - E_{rev})$
    - $I_{fast} = a_{fast} \tanh(v - v_{off,fast})$
    - $I_{slow} = a_{slow} \tanh(v_{slow} - v_{off,slow})$

    References:
        - Ribar, L., & Sepulchre, R. (2019). Neuromodulation of neuromorphic circuits. IEEE Transactions on Circuits and Systems I: Regular Papers, 66(8), 3028-3040.

    Attributes:
        reset_grad_preserve: Preserve the gradient when the neuron spikes by doing a soft reset.
        gmax_pasive: Maximal conductance of the passive current.
        Erev_pasive: Reversal potential for the passive current.
        a_fast: Amplitude parameter for the fast current dynamics.
        voff_fast: Voltage offset for the fast current activation.
        tau_fast: Time constant for the fast current (typically zero for instantaneous).
        a_slow: Amplitude parameter for the slow current dynamics.
        voff_slow: Voltage offset for the slow current activation.
        tau_slow: Time constant for the slow recovery variable.
        vthr: Voltage threshold for spike generation.
        C: Membrane capacitance.
        tsyn: Synaptic time constant for input current decay.
        weights: Synaptic weight matrix of shape (in_plus_neurons, neurons).
    """

    # Pasive parameters
    gmax_pasive: float = eqx.field(static=True)
    Erev_pasive: float = eqx.field(static=True)

    # Fast current
    a_fast: float = eqx.field(static=True)
    voff_fast: float = eqx.field(static=True)
    tau_fast: float = eqx.field(static=True)

    # Slow current
    a_slow: float = eqx.field(static=True)
    voff_slow: float = eqx.field(static=True)
    tau_slow: float = eqx.field(static=True)

    # Neuron threshold
    vthr: float = eqx.field(static=True)
    C: float = eqx.field(static=True, default=1.0)

    # Input synaptic time constant
    tsyn: float = eqx.field(static=True)

    dtype: DTypeLike = eqx.field(static=True)

    def __init__(
        self,
        *,
        tsyn: Union[int, float, jnp.ndarray] = 1.0,
        C: Union[int, float, jnp.ndarray] = 1.0,
        gmax_pasive: Union[int, float, jnp.ndarray] = 1.0,
        Erev_pasive: Union[int, float, jnp.ndarray] = 0.0,
        a_fast: Union[int, float, jnp.ndarray] = -2.0,
        voff_fast: Union[int, float, jnp.ndarray] = 0.0,
        tau_fast: Union[int, float, jnp.ndarray] = 0.0,
        a_slow: Union[int, float, jnp.ndarray] = 2.0,
        voff_slow: Union[int, float, jnp.ndarray] = 0.0,
        tau_slow: Union[int, float, jnp.ndarray] = 50.0,
        vthr: Union[int, float, jnp.ndarray] = 2.0,
        dtype: DTypeLike = jnp.float32,
    ):
        """Initialize the FitzHugh-Nagumo neuron model.

        Args:
            tsyn: Synaptic time constant for input current decay. Can be scalar or per-neuron array.
            C: Membrane capacitance. Can be scalar or per-neuron array.
            gmax_pasive: Maximal conductance of passive current. Can be scalar or per-neuron array.
            Erev_pasive: Reversal potential for passive current. Can be scalar or per-neuron array.
            a_fast: Amplitude of fast current. Can be scalar or per-neuron array.
            voff_fast: Voltage offset for fast current activation. Can be scalar or per-neuron array.
            tau_fast: Time constant for fast current (typically 0 for instantaneous). Can be scalar or per-neuron array.
            a_slow: Amplitude of slow current. Can be scalar or per-neuron array.
            voff_slow: Voltage offset for slow current activation. Can be scalar or per-neuron array.
            tau_slow: Time constant for slow recovery variable. Can be scalar or per-neuron array.
            vthr: Voltage threshold for spike generation. Can be scalar or per-neuron array.
            dtype: Data type for arrays (default: float32).
        """
        self.dtype = dtype

        self.tsyn = tsyn
        self.C = C
        self.gmax_pasive = gmax_pasive
        self.Erev_pasive = Erev_pasive
        self.a_fast = a_fast
        self.voff_fast = voff_fast
        self.tau_fast = tau_fast
        self.a_slow = a_slow
        self.voff_slow = voff_slow
        self.tau_slow = tau_slow
        self.vthr = vthr

    def init_state(self, n_neurons: int) -> Float[Array, "neurons 3"]:
        """Initialize the neuron state variables.

        Args:
            n_neurons: Number of neurons to initialize.

        Returns:
            Initial state array of shape (neurons, 3) containing [v, v_slow, i_app],
            where v is membrane voltage, v_slow is the slow recovery variable,
            and i_app is the applied synaptic current.
        """
        return jnp.zeros((n_neurons, 3), dtype=self.dtype)

    def IV_inst(self, v: Float[Array, "..."], Vrest: float = 0) -> Float[Array, "..."]:
        """Compute instantaneous I-V relationship with fast and slow currents at rest.

        Args:
            v: Membrane voltage.
            Vrest: Resting voltage for both fast and slow currents (default: 0).

        Returns:
            Total current at voltage v with both fast and slow currents evaluated at Vrest.
        """
        I_pasive = self.gmax_pasive * (v - self.Erev_pasive)
        I_fast = self.a_fast * jnp.tanh(Vrest - self.voff_fast)
        I_slow = self.a_slow * jnp.tanh(Vrest - self.voff_slow)

        return I_pasive + I_fast + I_slow

    def IV_fast(self, v: Float[Array, "..."], Vrest: float = 0) -> Float[Array, "..."]:
        """Compute I-V relationship with fast current at voltage v and slow current at rest.

        Args:
            v: Membrane voltage for passive and fast currents.
            Vrest: Resting voltage for slow current (default: 0).

        Returns:
            Total current with fast dynamics responding to v and slow current at Vrest.
        """
        I_pasive = self.gmax_pasive * (v - self.Erev_pasive)
        I_fast = self.a_fast * jnp.tanh(v - self.voff_fast)
        I_slow = self.a_slow * jnp.tanh(Vrest - self.voff_slow)

        return I_pasive + I_fast + I_slow

    def IV_slow(self, v: Float[Array, "..."], Vrest: float = 0) -> Float[Array, "..."]:
        """Compute steady-state I-V relationship with all currents at voltage v.

        Args:
            v: Membrane voltage for all currents.
            Vrest: Unused parameter for API consistency (default: 0).

        Returns:
            Total steady-state current with all currents responding to v.
        """
        I_pasive = self.gmax_pasive * (v - self.Erev_pasive)
        I_fast = self.a_fast * jnp.tanh(v - self.voff_fast)
        I_slow = self.a_slow * jnp.tanh(v - self.voff_slow)

        return I_pasive + I_fast + I_slow

    def dynamics(
        self,
        t: float,
        y: Float[Array, "neurons 3"],
        args: Dict[str, Any],
    ) -> Float[Array, "neurons 3"]:
        """Compute time derivatives of the neuron state variables.

        This implements the FitzHugh-Nagumo dynamics with passive, fast, and slow currents:
        - dv/dt: Fast membrane voltage dynamics
        - dv_slow/dt: Slow recovery variable dynamics
        - di_app/dt: Synaptic current decay

        Args:
            t: Current simulation time (unused but required by framework).
            y: State array of shape (neurons, 3) containing [v, v_slow, i_app].
            args: Additional arguments (unused but required by framework).

        Returns:
            Time derivatives of shape (neurons, 3) containing [dv/dt, dv_slow/dt, di_app/dt].
        """
        v = y[:, 0]
        v_slow = y[:, 1]
        i_app = y[:, 2]

        I_pasive = self.gmax_pasive * (v - self.Erev_pasive)
        I_fast = self.a_fast * jnp.tanh(v - self.voff_fast)
        I_slow = self.a_slow * jnp.tanh(v_slow - self.voff_slow)

        i_sum = I_pasive + I_fast + I_slow

        dv_dt = (i_app - i_sum) / self.C
        dvslow_dt = (v - v_slow) / self.tau_slow
        di_dt = -i_app / self.tsyn

        return jnp.stack([dv_dt, dvslow_dt, di_dt], axis=1)

    def spike_condition(
        self,
        t: float,
        y: Float[Array, "neurons 3"],
        **kwargs: Dict[str, Any],
    ) -> Float[Array, " neurons"]:
        """Compute spike condition for event detection.

        A spike is triggered when this function crosses zero (v >= vthr).

        Args:
            t: Current simulation time (unused but required by event detection).
            y: State array of shape (neurons, 3) containing [v, v_slow, i_app].
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Spike condition array of shape (neurons,). Positive values indicate v > vthr.
        """
        return y[:, 0] - self.vthr
