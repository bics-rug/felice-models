from typing import Generic, Protocol, TypeVar

import equinox as eqx
import jax
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray

from .implicit import Boomerang, Implicit
from .ssm import Mamba

__all__ = ["Mamba", "Implicit", "Boomerang", "SequenceClassifier"]

T = TypeVar("T")


class HasSequential(Protocol):
    def sequential(self, x): ...


class SequenceClassifier(eqx.Module, Generic[T]):
    embedding: eqx.nn.Embedding
    model: eqx.Module

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        model_class: T,
        key=PRNGKeyArray,
        **model_kwargs: dict,
    ):
        keys = jrandom.split(key, 2)
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])
        self.model = model_class(d_model=d_model, key=keys[1], **model_kwargs)

    def __call__(self, input_ids: Float[Array, " seq"]) -> Float[Array, "seq d_model"]:
        x = jax.vmap(self.embedding)(input_ids)
        y = self.model(x)
        return y

    def sequential(
        self: "SequenceClassifier[HasSequential]", input_ids: Float[Array, " seq"]
    ) -> Float[Array, "seq d_model"]:
        x = jax.vmap(self.embedding)(input_ids)
        return self.model.sequential(x)
