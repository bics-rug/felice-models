import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from einops import repeat
from jaxtyping import Array, Float, PRNGKeyArray

from .utils import binary_op


class SelectiveSSM(eqx.Module):
    d_model: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)

    x_proj: eqx.nn.Linear
    dt_proj: eqx.nn.Linear

    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, " d_inner"]

    def __init__(
        self,
        d_model: int,
        dt_rank: int,
        d_state: int = 16,
        *,
        key=PRNGKeyArray,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        keys = jrandom.split(key, 5)

        self.x_proj = eqx.nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, use_bias=False, key=keys[3]
        )
        self.dt_proj = eqx.nn.Linear(
            self.dt_rank, self.d_model, use_bias=True, key=keys[4]
        )

        # S4D-Real initialization
        A = repeat(
            jnp.arange(1, d_state + 1, dtype=jnp.float32),
            "n -> d n",
            d=self.d_model,
        )
        self.A_log = jnp.log(A)

        self.D = jnp.ones(self.d_model)

    def __call__(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq d_model"]:
        x_dbl = jax.vmap(self.x_proj)(x)
        dt, B, C = jnp.split(
            x_dbl, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )

        dt = jax.vmap(self.dt_proj)(dt)
        dt = jax.nn.softplus(dt)

        A = -jnp.exp(self.A_log)  # (d_inner, d_state)
        dA = jnp.exp(jnp.einsum("ld,dn->ldn", dt, A))
        dB_x = jnp.einsum("ld,ln,ld->ldn", dt, B, x)

        _, h = jax.lax.associative_scan(binary_op, (dA, dB_x), axis=0)
        y = jnp.einsum("ldn,ln->ld", h, C)

        y = y + x * self.D

        return y


class Mamba(eqx.Module):
    d_model: int = eqx.field(static=True)
    d_conv: int = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)

    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    conv1d: eqx.nn.Conv1d

    ssm: SelectiveSSM

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: int = 32,
        d_conv: int = 4,
        *,
        key=PRNGKeyArray,
    ):
        self.d_model = d_model
        self.d_conv = d_conv
        self.d_inner = d_inner

        keys = jrandom.split(key, 4)

        self.in_proj = eqx.nn.Linear(self.d_model, self.d_inner * 2, key=keys[0])
        self.conv1d = eqx.nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            key=keys[1],
        )

        self.ssm = SelectiveSSM(
            d_model=d_inner,
            dt_rank=math.ceil(self.d_model / 16),
            d_state=d_state,
            key=keys[2],
        )
        self.out_proj = eqx.nn.Linear(self.d_inner, self.d_model, key=keys[3])

    def __call__(self, x: Float[Array, "seq d_model"]) -> Float[Array, "seq d_model"]:
        seq_len, _ = x.shape

        # Projec the input into the convolution and residual
        xz = jax.vmap(self.in_proj)(x)
        x, z = jnp.split(xz, 2, axis=-1)

        # 1D Convolution
        x = x.T
        x = self.conv1d(x)[:, :seq_len]
        x = x.T
        x = jax.nn.silu(x)

        y = self.ssm(x)
        y = y * jax.nn.silu(z)

        logits = jax.vmap(self.out_proj)(y)
        return logits
