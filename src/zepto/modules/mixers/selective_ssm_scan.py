"""Mamba-2 selective SSM scan (token-unrolled)."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.compose.context import Compose
from zepto.semantic import (
    Add,
    Cast,
    Concat,
    Exp,
    Log,
    Multiply,
    ReduceSum,
    RepeatKV,
    Reshape,
    Split,
    Transpose,
)
from zepto.semantic.metadata import DType

from zepto.modules._internal._concat import concat_leading, split_leading
from zepto.modules._internal._helpers import (
    RMSNORM_COMPUTE_DTYPE,
    activation_dtype,
    add_bias_parameter,
    scale_by_parameter,
)


class SelectiveSSMScan(Module):
    """Selective state-space scan with grouped B/C sharing."""

    module_kind = "SelectiveSSMScan"

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        state_size: int,
        num_groups: int,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0 or state_size <= 0 or num_groups <= 0:
            raise ValueError("dimensions must be positive")
        if num_heads % num_groups != 0:
            raise ValueError("num_heads must be divisible by num_groups")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.decay_rate = Parameter(shape=(num_heads,), semantic_type="weight")
        self.d_skip = Parameter(shape=(num_heads,), semantic_type="weight")
        self.dt_bias = Parameter(shape=(num_heads,), semantic_type="bias")
        self._one = Tensor(shape=(1,), semantic_type="one", requires_grad=False)
        self._minus_one = Tensor(
            shape=(1,), semantic_type="minus_one", requires_grad=False
        )

    def _softplus(self, x: Tensor) -> Tensor:
        exp_x = Exp()(x)
        return Log()(Add()(self._one, exp_x))  # type: ignore[return-value]

    def _expand_groups(self, grouped: Tensor, seq_len: int) -> Tensor:
        """Expand (S, G, N) to (S, h, N) by repeating each group over heads."""
        parts = split_leading(grouped, seq_len)
        expanded: list[Tensor] = []
        for part in parts:
            g = Reshape(shape=(self.num_groups, self.state_size))(part)
            h = RepeatKV(n_rep=self.heads_per_group, axis=0)(g)
            expanded.append(Reshape(shape=(1, self.num_heads, self.state_size))(h))
        return concat_leading(tuple(expanded))

    def forward(
        self,
        x: Tensor,
        delta: Tensor,
        b: Tensor,
        c: Tensor,
        scan_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        seq_len, num_heads, head_dim = x.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("x shape mismatch")
        if delta.shape != (seq_len, num_heads):
            raise ValueError("delta must be (S, h)")
        if b.shape[1] != self.num_groups or c.shape[1] != self.num_groups:
            raise ValueError("B/C group count mismatch")

        restore_dtype = activation_dtype(x)
        b_h = self._expand_groups(b, seq_len)
        c_h = self._expand_groups(c, seq_len)

        x_parts = split_leading(x, seq_len)
        delta_parts = split_leading(delta, seq_len)
        b_parts = split_leading(b_h, seq_len)
        c_parts = split_leading(c_h, seq_len)

        if scan_state_in is None:
            ctx = Compose.current()
            if ctx is None:
                raise RuntimeError("SelectiveSSMScan requires an active Compose context")
            state = ctx.input(
                Tensor(
                    shape=(num_heads, head_dim, self.state_size),
                    semantic_type="zero",
                    dtype=RMSNORM_COMPUTE_DTYPE,
                    requires_grad=False,
                )
            )
        else:
            expected = (num_heads, head_dim, self.state_size)
            if scan_state_in.shape != expected:
                raise ValueError("scan_state_in shape mismatch")
            state = Cast(to_dtype=RMSNORM_COMPUTE_DTYPE)(scan_state_in)  # type: ignore[assignment]

        outputs: list[Tensor] = []
        for t in range(seq_len):
            x_t = Reshape(shape=(num_heads, head_dim))(x_parts[t])
            b_t = Reshape(shape=(num_heads, 1, self.state_size))(b_parts[t])
            c_t = Reshape(shape=(num_heads, 1, self.state_size))(c_parts[t])

            delta_raw = add_bias_parameter(
                Reshape(shape=(num_heads,))(delta_parts[t]), self.dt_bias
            )
            delta_pos = self._softplus(
                Cast(to_dtype=RMSNORM_COMPUTE_DTYPE)(delta_raw)  # type: ignore[arg-type]
            )
            delta_broadcast = Reshape(shape=(num_heads, 1, 1))(delta_pos)
            scaled_delta = scale_by_parameter(delta_pos, self.decay_rate)  # type: ignore[arg-type]
            neg_a = Multiply()(scaled_delta, self._minus_one)  # type: ignore[call-arg]
            a_bar = Exp()(Reshape(shape=(num_heads, 1, 1))(neg_a))
            b_bar = Multiply()(delta_broadcast, b_t)  # type: ignore[call-arg]

            x_col = Reshape(shape=(num_heads, head_dim, 1))(x_t)
            update = Multiply()(x_col, b_bar)  # type: ignore[call-arg]
            decayed = Multiply()(state, a_bar)  # type: ignore[call-arg]
            state = Add()(decayed, update)  # type: ignore[assignment]

            c_broadcast = Reshape(shape=(num_heads, 1, self.state_size))(c_t)
            weighted = Multiply()(state, c_broadcast)  # type: ignore[call-arg]
            y = ReduceSum(axis=2)(weighted)
            y = Reshape(shape=(num_heads, head_dim))(y)
            x_scaled = Transpose(permutation=(1, 0))(x_t)
            x_scaled = scale_by_parameter(x_scaled, self.d_skip)  # type: ignore[arg-type]
            x_scaled = Transpose(permutation=(1, 0))(x_scaled)  # type: ignore[assignment]
            y = Add()(y, x_scaled)  # type: ignore[assignment]
            outputs.append(Reshape(shape=(1, num_heads, head_dim))(y))

        output = concat_leading(tuple(outputs))
        output = Cast(to_dtype=restore_dtype)(output)  # type: ignore[assignment]
        state_out = Cast(to_dtype=restore_dtype)(state)  # type: ignore[assignment]
        return output, state_out


__all__ = ["SelectiveSSMScan"]
