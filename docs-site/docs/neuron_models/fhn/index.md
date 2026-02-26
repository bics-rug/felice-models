# FitzHugh-Nagumo

## Circuit equation

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

## Examples

See the following interactive notebook for a practical example:

- [Basic Usage Example](fhn.ipynb) - Introduction to the FitzHugh-Nagumo model