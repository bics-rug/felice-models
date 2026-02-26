from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class ReasoningDataset:
    """
    Task types:
    1. Simple comparison: IF A > B THEN x ELSE y
    2. Accumulated conditions: SET A, ADD B, IF SUM > threshold THEN x ELSE y
    """

    # Token vocabulary
    VOCAB = {
        "PAD": 0,
        "SET_A": 1,
        "SET_B": 2,
        "SET_C": 3,
        "IF_A>B": 4,
        "IF_A<B": 5,
        "IF_SUM>": 6,
        "THEN": 7,
        "ELSE": 8,
        "QUERY": 9,
        "ADD": 10,
    }
    NUM_OFFSET: int = 11
    VOCAB_SIZE: int = 31
    NUM_OUTPUT: int = 16

    def generate_simple_comparison(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        """
        Generate: [SET_A, a, SET_B, b, IF_A>B, THEN, x, ELSE, y, QUERY]
        Target at QUERY position: x if a > b else y
        """
        keys = jax.random.split(key, 4)

        a = jax.random.randint(keys[0], (), 0, self.NUM_OUTPUT - 1)
        b = jax.random.randint(keys[1], (), 0, self.NUM_OUTPUT - 1)
        x = jax.random.randint(keys[2], (), 0, self.NUM_OUTPUT - 1)
        y = jax.random.randint(keys[3], (), 0, self.NUM_OUTPUT - 1)

        # TODO: Generate other sequences
        input_seq = jnp.array(
            [
                self.VOCAB["SET_A"],
                self.NUM_OFFSET + a,
                self.VOCAB["SET_B"],
                self.NUM_OFFSET + b,
                self.VOCAB["IF_A>B"],
                self.VOCAB["THEN"],
                self.NUM_OFFSET + x,
                self.VOCAB["ELSE"],
                self.NUM_OFFSET + y,
                self.VOCAB["QUERY"],
            ]
        )

        result = jnp.where(a > b, x, y)
        target = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, result])
        mask = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        return input_seq, target, mask

    def generate_accumulation_condition(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        """
        Generate: [SET_A, a, ADD, b, ADD, c, IF_SUM>, threshold, THEN, x, ELSE, y, QUERY]
        Requires accumulating a + b + c and comparing to threshold.
        """
        keys = jax.random.split(key, 6)

        a = jax.random.randint(keys[0], (), 0, 10)
        b = jax.random.randint(keys[1], (), 0, 10)
        c = jax.random.randint(keys[2], (), 0, 10)
        threshold = jax.random.randint(keys[3], (), 5, 25)
        x = jax.random.randint(keys[4], (), 0, 15)
        y = jax.random.randint(keys[5], (), 0, 15)

        # TODO: Generate other sequences
        input_seq = jnp.array(
            [
                self.VOCAB["SET_A"],
                self.NUM_OFFSET + a,
                self.VOCAB["ADD"],
                self.NUM_OFFSET + b,
                self.VOCAB["ADD"],
                self.NUM_OFFSET + c,
                self.VOCAB["IF_SUM>"],
                self.NUM_OFFSET + threshold,
                self.VOCAB["THEN"],
                self.NUM_OFFSET + x,
                self.VOCAB["ELSE"],
                self.NUM_OFFSET + y,
                self.VOCAB["QUERY"],
            ]
        )

        total = a + b + c
        result = jnp.where(total > threshold, x, y)

        target = jnp.zeros(13, dtype=jnp.int32).at[-1].set(result)
        mask = jnp.zeros(13, dtype=jnp.int32).at[-1].set(1)

        return input_seq, target, mask

    def generate_batch(
        self, key: PRNGKeyArray, batch_size: int, task_type: str = "simple"
    ) -> Tuple[Array, Array, Array]:
        keys = jax.random.split(key, batch_size)

        if task_type == "simple":
            gen_fn = self.generate_simple_comparison
        else:  # accumulation
            gen_fn = self.generate_accumulation_condition

        inputs, targets, masks = [], [], []
        for k in keys:
            inp, tgt, msk = gen_fn(k)
            inputs.append(inp)
            targets.append(tgt)
            masks.append(msk)

        return jnp.stack(inputs), jnp.stack(targets), jnp.stack(masks)
