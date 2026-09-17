"""Gated DeltaNet recurrence (S-unrolled at compose time)."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.compose.context import Compose
from zepto.semantic import Add, Cast, Concat, Exp, MatMul, Multiply, Reshape, Split, Subtract
from zepto.semantic.metadata import DType

from ._concat import concat_leading, split_leading
from ._helpers import RMSNORM_COMPUTE_DTYPE, activation_dtype


class GatedDeltaScan(Module):
    """Token-unrolled gated delta-rule linear attention scan."""

    module_kind = "GatedDeltaScan"

    def __init__(
        self,
        num_heads: int,
        key_head_dim: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or key_head_dim <= 0 or value_head_dim <= 0:
            raise ValueError("head counts and dims must be positive")
        self.num_heads = num_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        beta: Tensor,
        log_decay: Tensor,
        scan_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if query.shape != key.shape:
            raise ValueError("query and key must share shape")
        if value.shape[0] != query.shape[0] or value.shape[1] != query.shape[1]:
            raise ValueError("value head/seq dims must match query")
        seq_len, num_heads, key_dim = query.shape
        if num_heads != self.num_heads or key_dim != self.key_head_dim:
            raise ValueError("query shape does not match scan configuration")
        if value.shape[2] != self.value_head_dim:
            raise ValueError("value head dim mismatch")

        restore_dtype = activation_dtype(query)
        q_parts = split_leading(query, seq_len)
        k_parts = split_leading(key, seq_len)
        v_parts = split_leading(value, seq_len)
        beta_parts = split_leading(beta, seq_len)
        decay_parts = split_leading(log_decay, seq_len)

        if scan_state_in is None:
            ctx = Compose.current()
            if ctx is None:
                raise RuntimeError("GatedDeltaScan requires an active Compose context")
            state = ctx.input(
                Tensor(
                    shape=(num_heads, key_dim, self.value_head_dim),
                    semantic_type="zero",
                    dtype=RMSNORM_COMPUTE_DTYPE,
                    requires_grad=False,
                )
            )
        else:
            if scan_state_in.shape != (num_heads, key_dim, self.value_head_dim):
                raise ValueError("scan_state_in shape mismatch")
            state = Cast(to_dtype=RMSNORM_COMPUTE_DTYPE)(scan_state_in)  # type: ignore[assignment]

        outputs: list[Tensor] = []
        for t in range(seq_len):
            q_t = Reshape(shape=(num_heads, key_dim))(q_parts[t])
            k_t = Reshape(shape=(num_heads, key_dim))(k_parts[t])
            v_t = Reshape(shape=(num_heads, self.value_head_dim))(v_parts[t])
            beta_t = Reshape(shape=(num_heads, 1))(beta_parts[t])
            decay_t = Reshape(shape=(num_heads, 1, 1))(decay_parts[t])

            alpha = Exp()(decay_t)
            decayed = Multiply()(state, alpha)  # type: ignore[call-arg]

            k_row = Reshape(shape=(num_heads, 1, key_dim))(k_t)
            prediction = MatMul()(k_row, decayed)  # type: ignore[call-arg]
            prediction = Reshape(shape=(num_heads, self.value_head_dim))(prediction)
            delta = Multiply()(beta_t, Subtract()(v_t, prediction))  # type: ignore[call-arg]

            k_col = Reshape(shape=(num_heads, key_dim, 1))(k_t)
            delta_row = Reshape(shape=(num_heads, 1, self.value_head_dim))(delta)
            outer = MatMul()(k_col, delta_row)  # type: ignore[call-arg]
            state = Add()(decayed, outer)  # type: ignore[assignment]

            q_row = Reshape(shape=(num_heads, 1, key_dim))(q_t)
            out_t = MatMul()(q_row, state)  # type: ignore[call-arg]
            outputs.append(Reshape(shape=(1, num_heads, self.value_head_dim))(out_t))

        output = concat_leading(tuple(outputs))
        output = Cast(to_dtype=restore_dtype)(output)  # type: ignore[return-value]
        state_out = Cast(to_dtype=restore_dtype)(state)  # type: ignore[assignment]
        return output, state_out  # type: ignore[return-value]


__all__ = ["GatedDeltaScan"]
