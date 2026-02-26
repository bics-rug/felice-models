import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from wigglystuff import Slider2D

    return Slider2D, mo


@app.cell
def _():
    import diffrax
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    return diffrax, jax, jnp, np, plt


@app.cell
def _(diffrax, jax, jnp):
    def vector_field(t, state, args):
        u, v = state
        alpha, beta, gamma, kappa, sigma, delta = args

        z = jax.nn.tanh(kappa * (u - v))

        # Prey dynamics
        du = z * (1 - alpha * jnp.exp(beta * v) * (1 + gamma * (0.5 - u))) - sigma

        # Predator dynamics
        dv = z * (-1 + alpha * jnp.exp(beta * u) * u * (1 + gamma * (0.5 - v))) - sigma

        return jnp.array([du, dv])

    def vector_field_prod(t, state, args):
        u, v = state
        alpha, beta, gamma, kappa, sigma, delta = args

        z = jax.nn.tanh(kappa * (u - v))

        # Prey dynamics
        du = (
            z * (1 - alpha * jnp.exp(beta * v) * (1 + gamma * (0.5 - u))) - sigma
            # + sigma * jnp.maximum(0, delta - u) / (delta + 1e-16)
        )

        # Predator dynamics
        dv = (
            z * (-1 + alpha * jnp.exp(beta * u) * (1 + gamma * (0.5 - v))) - sigma
            # + sigma * jnp.maximum(0, delta - v) / (delta + 1e-16)
        )

        dv = jnp.where(jnp.allclose(z, 0.0), dv * jnp.sign(v), dv)
        du = jnp.where(jnp.allclose(z, 0.0), du * jnp.sign(u), du)

        return jnp.array([du, dv])

    def vector_field_exp(t, state, args):
        u, v = state
        alpha, beta, gamma, kappa, sigma, delta = args

        z = jax.nn.tanh(kappa * (u - v))

        # Prey dynamics
        du = (
            z * (1 - alpha * jnp.exp(beta * v) * (1 + gamma * (0.5 - u)))
            - sigma
            + sigma * jnp.exp(-u / delta)
        )

        # Predator dynamics
        dv = (
            z * (-1 + alpha * jnp.exp(beta * u) * u * (1 + gamma * (0.5 - v)))
            - sigma
            + sigma * jnp.exp(-v / delta)
        )

        return jnp.array([du, dv])

    def physical_vector_field(t, state, args):
        x1, x2 = state
        alpha, beta, gamma, kappa, sigma, delta = args

        In0 = 129e-15  # fixed by design
        C = 0.1e-12  # fixed by design
        kk = 0.39  # fixed by tech
        Ut = 0.025  # temperature dependent

        Ibias = In0 / alpha

        Ia = Ibias * sigma
        x3 = jax.nn.tanh(kappa * (x1 - x2))

        dx1 = (
            x3 * Ibias
            - (In0 * jnp.exp(kk * x2 / Ut)) * (x3 + 26e-2 * (0.5 - x1) * x3)
            - Ia
        ) / C
        dx2 = (
            -x3 * Ibias
            + In0 * jnp.exp(kk * x1 / Ut) * (x3 + 26e-2 * (0.5 - x2) * x3)
            - Ia
        ) / C

        return jnp.array([dx1, dx2])

    def compute_nullclines(vector_field, u_range, v_range, args, resolution=200):
        """
        Compute nullclines
        du/dt = 0 (u-nullcline)
        dv/dt = 0 (v-nullcline)
        """
        alpha, beta, gamma, kappa, sigma, delta = args
        u_vals = jnp.linspace(u_range[0], u_range[1], resolution)
        v_vals = jnp.linspace(v_range[0], v_range[1], resolution)
        U, V = jnp.meshgrid(u_vals, v_vals)

        dU, dV = vector_field(0, [U, V], args)

        return U, V, dU, dV

    def solve(dyn, y0, p, T, n=1000):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(dyn),
            diffrax.Tsit5(),
            t0=0.0,
            t1=T,
            dt0=0.01,
            y0=y0,
            args=p,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, T, n)),
            stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-8),
            max_steps=50000,
        )
        return sol.ts, sol.ys

    return compute_nullclines, solve, vector_field_prod


@app.cell
def _(Slider2D, mo):
    # alpha = 0.5      # I_n0 / I_bias ratio
    # beta = 0.39/0.025       # k / U_t (inverse thermal scale)
    # gamma = 0.26     # coupling coefficient
    # kappa = 5.0      # tanh steepness
    # sigma = 0.6      # bias scaling (s * I_bias normalized)
    # y0 = jnp.array([0.2, 0.4])
    # ts, ys = solve(vector_field, y0, params, 140)

    alpha = mo.ui.slider(
        0.0004, 0.012, 0.00001, 0.00129, label="alpha", orientation="vertical"
    )
    beta = mo.ui.slider(
        0.0, 30, 0.00001, 0.39 / 0.025, label="beta", orientation="vertical"
    )
    gamma = mo.ui.slider(0, 1, 0.01, 0.26, label="gamma", orientation="vertical")
    kappa = mo.ui.slider(0, 10, 0.1, 5.0, label="kappa", orientation="vertical")
    sigma = mo.ui.slider(0, 1, 0.01, 0.6, label="sigma", orientation="vertical")
    delta = mo.ui.slider(0, 0.1, 0.001, 0.02, label="delta", orientation="vertical")

    # v0 = mo.ui.slider(0, 1.0, 0.01, 0.3, label="v0")
    # u0 = mo.ui.slider(0, 1.0, 0.01, 0.2, label="u0", orientation="vertical")

    state0 = mo.ui.anywidget(
        Slider2D(
            width=150,
            height=150,
            x_bounds=(-1.0, 1.5),
            y_bounds=(-1.0, 1.5),
        )
    )

    mo.hstack(
        [
            mo.plain_text("""
    alpha: I_n0 / I_bias ratio
    beta: k / U_t ratio
    gamma: coupling coefficient
    kappa: tanh steepness
    sigma: bias scaling (s * I_bias)
    """),
            mo.hstack(
                [state0, alpha, beta, gamma, kappa, sigma, delta], justify="start"
            ),
        ]
    )
    return alpha, beta, delta, gamma, kappa, sigma, state0


@app.cell
def _(
    alpha,
    beta,
    compute_nullclines,
    delta,
    gamma,
    jnp,
    kappa,
    np,
    plt,
    sigma,
    solve,
    state0,
    vector_field_prod,
):
    params = (
        alpha.value,
        beta.value,
        gamma.value,
        kappa.value,
        sigma.value,
        delta.value,
    )
    ic_neuro = [state0.x, state0.y]

    u_range = [-1.0, 1.5]
    v_range = [-1.0, 1.5]

    u_sparse = jnp.linspace(u_range[0], u_range[1], 20)
    v_sparse = jnp.linspace(v_range[0], v_range[1], 20)

    Us, Vs = jnp.meshgrid(u_sparse, v_sparse)

    def plot_vf(ax, vector_field):
        U, V, dU, dV = compute_nullclines(vector_field, u_range, v_range, params)
        dUs, dVs = vector_field(0, [Us, Vs], params)

        # Normalize for visualization
        magnitude = np.sqrt(dUs**2 + dVs**2)
        magnitude[magnitude == 0] = 1
        dUs_norm = dUs / magnitude
        dVs_norm = dVs / magnitude

        # Nullclines
        ax.contour(U, V, dU, levels=[0], colors="blue", linewidths=2, linestyles="-")
        ax.contour(U, V, dV, levels=[0], colors="red", linewidths=2, linestyles="-")

        ax.quiver(Us, Vs, dUs_norm, dVs_norm, magnitude, cmap="viridis", alpha=0.6)

        # Trajectories
        color = plt.cm.plasma(0.2)

        ts, ys = solve(vector_field, jnp.array(ic_neuro), params, 50)
        ax.plot(ys[:, 0], ys[:, 1], "-", color=color, linewidth=1.5, alpha=0.8)
        ax.plot(ic_neuro[0], ic_neuro[1], "o", color=color, markersize=6)

        ax.set_xlabel("u (Prey)")
        ax.set_ylabel("v (Predator)")
        ax.set_title("Wererabbit: Phase Portrait")
        ax.legend(["u-nullcline (du/dt=0)", "v-nullcline (dv/dt=0)"], loc="upper right")
        ax.set_xlim(u_range)
        ax.set_ylim(v_range)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)

    def plot_trj(ax, vector_field):
        ts, ys = solve(vector_field, jnp.array(ic_neuro), params, 50)
        ax.plot(ts, ys[:, 0], "b-", linewidth=2, label="u (Prey)")
        ax.plot(ts, ys[:, 1], "r-", linewidth=2, label="v (Predator)")

        ax.set_xlabel("Time τ")
        ax.set_ylabel("Population")
        ax.set_title(
            f"Wererabbit: Time Series (IC: u₀={ic_neuro[0]:.2f}, v₀={ic_neuro[1]:.2f})"
        )
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    fig = plt.figure(figsize=(10, 4))

    # --- Plot 1: Wererabbit Phase Portrait ---
    ax1 = fig.add_subplot(1, 2, 1)
    plot_vf(ax1, vector_field_prod)

    ax2 = fig.add_subplot(1, 2, 2)
    plot_trj(ax2, vector_field_prod)

    # ax3 = fig.add_subplot(3, 2, 3)
    # plot_vf(ax3, vector_field_prod)

    # ax4 = fig.add_subplot(3, 2, 4)
    # plot_trj(ax4, vector_field_prod)

    # ax5 = fig.add_subplot(3, 2, 5)
    # plot_vf(ax5, vector_field_exp)

    # ax6 = fig.add_subplot(3, 2, 6)
    # plot_trj(ax6, vector_field_exp)

    plt.tight_layout()
    fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
