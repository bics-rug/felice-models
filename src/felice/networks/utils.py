from typing import Tuple

import jax
from jaxtyping import Array


@jax.jit
def binary_op(
    left: Tuple[Array, Array], right: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    a1, b1 = left
    a2, b2 = right
    return (a1 * a2, a2 * b1 + b2)
