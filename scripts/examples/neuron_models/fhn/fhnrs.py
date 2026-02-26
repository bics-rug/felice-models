import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import diffrax as dfx
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from jax import jit

    return dfx, jit, jnp, mo, np, plt


@app.cell
def _(dfx, jnp):
    def vector_field(t, y, args):
        v, vslow = y

        ipasive = args["gmax"] * (v - args["Erev"])
        ifast = args["af"] * jnp.tanh(v - args["Ef"])
        islow = args["as"] * jnp.tanh(vslow - args["Es"])

        dv = (-ipasive - ifast - islow) / args["C"]
        dvs = (v - vslow) / args["ts"]

        return jnp.array([dv, dvs])

    term = dfx.ODETerm(vector_field)
    return term, vector_field


@app.cell
def _(mo):
    p1 = mo.ui.slider(0.0, 5.0, value=1.0, step=0.1, label="gmax")
    p2 = mo.ui.slider(-1.0, 1.0, value=0.0, step=0.1, label="Erev")
    p3 = mo.ui.slider(-5.0, 5.0, value=-2.0, step=0.05, label="af")
    p4 = mo.ui.slider(-1.0, 1.0, value=0.0, step=0.05, label="Ef")
    p5 = mo.ui.slider(-5.0, 5.0, value=2.0, step=0.05, label="as")
    p6 = mo.ui.slider(-1.0, 1.0, value=0.0, step=0.05, label="Es")
    p7 = mo.ui.slider(1.0, 100.0, value=50.0, step=0.1, label="ts")
    p8 = mo.ui.slider(0.0, 1.0, value=1.0, step=0.01, label="C")

    mo.hstack(
        [
            mo.vstack([p1, p2, p3, p4], justify="start", gap=1),
            mo.vstack([p5, p6, p7, p8], justify="start", gap=1),
        ]
    )
    return p1, p2, p3, p4, p5, p6, p7, p8


@app.cell
def _(mo):
    mo.md("""
    ### Initial Conditions & Simulation
    """)
    return


@app.cell
def _(mo):
    x0 = mo.ui.slider(-5.0, 5.0, value=2.0, step=0.1, label="x₀")
    y0 = mo.ui.slider(-5.0, 5.0, value=0.0, step=0.1, label="y₀")
    t_max = mo.ui.slider(10, 100, value=30, step=5, label="t_max")

    mo.hstack([x0, y0, t_max], justify="start", gap=2)
    return t_max, x0, y0


@app.cell
def _(dfx, jit, jnp, p1, p2, p3, p4, p5, p6, p7, p8, t_max, term, x0, y0):
    @jit
    def solve_ode(y_init, args, t_end):
        solver = dfx.Tsit5()
        saveat = dfx.SaveAt(ts=jnp.linspace(0, t_end, 2000))
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0,
            t1=t_end,
            dt0=0.01,
            y0=y_init,
            args=args,
            saveat=saveat,
            max_steps=100000,
        )
        return sol.ts, sol.ys

    args = {
        "gmax": p1.value,
        "Erev": p2.value,
        "af": p3.value,
        "Ef": p4.value,
        "as": p5.value,
        "Es": p6.value,
        "ts": p7.value,
        "C": p8.value,
    }
    y_init = jnp.array([x0.value, y0.value])

    t, ys = solve_ode(y_init, args, float(t_max.value))
    x_sol = ys[:, 0]
    y_sol = ys[:, 1]
    return args, t, x_sol, y_sol


@app.cell
def _(args, jnp, np, plt, t, vector_field, x0, x_sol, y0, y_sol):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time series

    ax1.plot(t, x_sol, "b-", lw=1.5, label="x(t)")
    ax1.plot(t, y_sol, "r-", lw=1.5, label="y(t)")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("State")
    ax1.set_title("Transient Response")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Phase plane bounds
    # pad = 1.0
    xmin, xmax = -4, 4
    ymin, ymax = -2.5, 2.5

    # Vector field
    X, Y = jnp.meshgrid(jnp.linspace(xmin, xmax, 20), jnp.linspace(ymin, ymax, 20))
    U, V = jnp.zeros_like(X), np.zeros_like(Y)

    state = jnp.stack([X, Y], axis=0)
    deriv = vector_field(0.0, state, args)
    dx, dy = deriv[0], deriv[1]
    mag = jnp.sqrt(dx**2 + dy**2)
    U = jnp.where(mag > 0, dx / mag, U)
    V = jnp.where(mag > 0, dy / mag, V)

    ax2.quiver(X, Y, U, V, alpha=0.4, color="gray", scale=25)

    # Nullclines
    Xf, Yf = jnp.meshgrid(jnp.linspace(xmin, xmax, 150), jnp.linspace(ymin, ymax, 150))
    DX, DY = jnp.zeros_like(Xf), jnp.zeros_like(Yf)
    state = jnp.stack([Xf, Yf], axis=0)
    deriv = vector_field(0.0, state, args)
    DX, DY = deriv[0], deriv[1]
    ax2.contour(
        Xf,
        Yf,
        DX,
        levels=[0],
        colors="blue",
        linestyles="--",
        linewidths=1.5,
        alpha=0.7,
    )
    ax2.contour(
        Xf, Yf, DY, levels=[0], colors="red", linestyles="--", linewidths=1.5, alpha=0.7
    )

    # Trajectory
    ax2.plot(x_sol, y_sol, "b-", lw=2)
    ax2.plot(x0.value, y0.value, "go", ms=10, label="Start")
    ax2.plot(x_sol[-1], y_sol[-1], "r*", ms=12, label="End")

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Phase Plane")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)

    plt.tight_layout()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
