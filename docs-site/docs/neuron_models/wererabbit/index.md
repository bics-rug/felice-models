# WereRabbit

The wererabbit neuron model is a two coupled oscillator that follows a predator- prey dynamic with a switching in the diagonal of the phaseplane. When the z in equation 1c represents the “moon phase”, when ever it cross that threshold, the rabbit (prey) becomes the predator.

## Circuit equation

$$
\begin{align}
    C\frac{du}{dt} &= z I_{bias} - I_{n0} e^{\kappa v / U_t} [z + 26e^{-2} (0.5 - u) z] - I_a \\
    C\frac{dv}{dt} &= -z I_{bias} + I_{n0} e^{\kappa u / U_t} [z + 26e^{-2} (0.5 - v) z] - I_a \\
    z &= tanh(\rho (u-v))\\
    I_a &= \sigma I_{bias} \\
\end{align}
$$

| **Parameter** | **Symbol** | **Definition** | **Value** |
|-----------|--------|------------|-------|
| Capacitance | C | Circuit capacitance | $0.1\,pF$ |
| Bias current | $I_{bias}$ | DC bias current for the fixpoint location | $100\,pA$
Leakage current | $I_{n0}$ | Transistor leakage current | $0.129\,pA$
Subthreshold slope | $\kappa$ | Transistor subthreshold slope factor | $0.39$
Thermal voltage | $U_t$ | Thermal voltage at room temperature | $25\,mV$
Bias scale | $\sigma$ | Scaling factor for the distance between fixpoints | $0.6$
Steepness | $\rho$ | Tanh steepness for the moonphase | $5$s

## Abstraction
To simplify the analysis of the model for simulation purposes, we can introduce a dimensionless time variable $\tau=tI_{bias}/C$, transforming the derivate of the equations in $\frac{d}{dt}=\frac{I_{bias}}{C}\frac{d}{d\tau}$. Substituting this time transformation on equation~\ref{eq:wererabbit:circ}

$$
\begin{equation}
C\frac{I_{bias}}{C}\frac{du}{d\tau} = z I_{bias} - I_{n0} e^{\kappa v / U_t} [z + 26e^{-2} (0.5 - u) z] - \sigma I_{bias}
\end{equation}
$$

And dividing by $I_{bias}$ on both sides:

$$
\begin{equation}
    \frac{du}{d\tau} = z - \frac{I_{n0}}{I_{bias}} e^{\kappa v / U_t} [z + 26e^{-2} (0.5 - u) z] - \sigma
\end{equation}
$$

Obtaining the following set of equations:

$$
\begin{align}
    z &= tanh(\kappa (u-v)) \\
    \frac{du}{dt} &= z - z \alpha e^{\beta v} [1 + \gamma (0.5 - u)] - \sigma \\
    \frac{dv}{dt} &= -z - z \alpha e^{\beta u} [1 + \gamma (0.5 - v)] - \sigma
\end{align}
$$

| **Parameter** | **Definition** | **Value** |
|---------------|----------------|-----------|
| $\tau$ | $tI_{bias}/C$ | -- |
| $\alpha$ | $I_{n0}/I_{bias}$ | $0.0129$ |
| $\beta$ | $\kappa/U_t$ | 15.6  |
| $\gamma$ | -- | $26e^{-2}$ |
| $\rho$ | Tanh steepness for the moonphase | 5  |
| $\sigma$ | Scaling factor for the distance between fixpoints | 0.6 |

## Examples

See the following interactive notebook for a practical example:

- [Basic Usage Example](wererabbit.ipynb) - Introduction to the WereRabbit model
