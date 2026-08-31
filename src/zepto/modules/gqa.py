"""Grouped-query attention module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, MatMul, RepeatKV, Reshape, Transpose
from .linear import Linear
from .qk_norm import QKNormRMSNorm
from .rope_apply import RoPEApply
from .softmax import Softmax, attention_softmax_scale


class GroupedQueryAttention(Module):
    """Masked grouped-query self-attention (decoder-only, eager decomposition).

    Pipeline: Q/K/V projections → head reshape/transpose → optional QK-Norm →
    optional RoPE → RepeatKV → ``QK^T`` → mask add → scaled softmax → ``PV`` →
    merge → output projection. Matches HuggingFace Llama/Apertus eager layout.
    """

    module_kind = "GroupedQueryAttention"

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        head_dim: int | None = None,
        qk_norm: QKNormRMSNorm | None = None,
        rope: RoPEApply | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("hidden_size, num_heads, and num_kv_heads must be positive")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        resolved_head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        if num_heads * resolved_head_dim != hidden_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must equal "
                f"num_heads * head_dim ({num_heads} * {resolved_head_dim})"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = resolved_head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.kv_projection_size = num_kv_heads * resolved_head_dim

        self.q_proj = Linear(hidden_size, hidden_size)
        self.k_proj = Linear(hidden_size, self.kv_projection_size)
        self.v_proj = Linear(hidden_size, self.kv_projection_size)
        self.o_proj = Linear(hidden_size, hidden_size)
        self.softmax = Softmax(scale=attention_softmax_scale(resolved_head_dim))

        if qk_norm is not None:
            self.qk_norm = qk_norm
        if rope is not None:
            self.rope = rope

    def forward(
        self,
        hidden_states: Tensor,
        causal_mask: Tensor,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        if len(hidden_states.shape) != 2:
            raise ValueError(
                "GroupedQueryAttention expects rank-2 hidden states (S, d), "
                f"got {hidden_states.shape}"
            )
        seq_len, hidden = hidden_states.shape
        if hidden != self.hidden_size:
            raise ValueError(
                f"hidden size mismatch: expected {self.hidden_size}, got {hidden}"
            )

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query_heads = Transpose(permutation=(1, 0, 2))(
            Reshape(shape=(seq_len, self.num_heads, self.head_dim))(query)
        )
        key_heads = Transpose(permutation=(1, 0, 2))(
            Reshape(shape=(seq_len, self.num_kv_heads, self.head_dim))(key)
        )
        value_heads = Transpose(permutation=(1, 0, 2))(
            Reshape(shape=(seq_len, self.num_kv_heads, self.head_dim))(value)
        )

        qk_norm = getattr(self, "qk_norm", None)
        if qk_norm is not None:
            query_heads, key_heads = qk_norm(query_heads, key_heads)  # type: ignore[misc]

        rope = getattr(self, "rope", None)
        if rope is not None:
            if rope_cos is None or rope_sin is None:
                raise ValueError(
                    "GroupedQueryAttention RoPE requires rope_cos and rope_sin"
                )
            query_heads = rope(query_heads, rope_cos, rope_sin)
            key_heads = rope(key_heads, rope_cos, rope_sin)

        key_heads = RepeatKV(n_rep=self.num_kv_groups, axis=0)(key_heads)
        value_heads = RepeatKV(n_rep=self.num_kv_groups, axis=0)(value_heads)

        key_transposed = Transpose(permutation=(0, 2, 1))(key_heads)
        scores = MatMul()(query_heads, key_transposed)
        masked_scores = Add()(scores, causal_mask)
        attention_weights = self.softmax(masked_scores)
        context = MatMul()(attention_weights, value_heads)

        merged = Reshape(shape=(seq_len, self.hidden_size))(
            Transpose(permutation=(1, 0, 2))(context)
        )
        return self.o_proj(merged)  # type: ignore[return-value]
