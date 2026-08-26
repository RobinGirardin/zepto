"""Grouped-query attention module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.functional import add, matmul, repeat_kv, reshape, transpose
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

        self._qk_norm = qk_norm
        self._rope = rope
        self.softmax = Softmax(scale=attention_softmax_scale(resolved_head_dim))

    def forward(
        self,
        hidden_states: GraphTensor,
        causal_mask: GraphTensor,
        rope_cos: GraphTensor | None = None,
        rope_sin: GraphTensor | None = None,
    ) -> GraphTensor:
        if len(hidden_states.metadata.shape) != 2:
            raise ValueError(
                "GroupedQueryAttention expects rank-2 hidden states (S, d), "
                f"got {hidden_states.metadata.shape}"
            )
        seq_len, hidden = hidden_states.metadata.shape
        if hidden != self.hidden_size:
            raise ValueError(
                f"hidden size mismatch: expected {self.hidden_size}, got {hidden}"
            )

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query_heads = transpose(
            reshape(query, (seq_len, self.num_heads, self.head_dim)),
            (1, 0, 2),
        )
        key_heads = transpose(
            reshape(key, (seq_len, self.num_kv_heads, self.head_dim)),
            (1, 0, 2),
        )
        value_heads = transpose(
            reshape(value, (seq_len, self.num_kv_heads, self.head_dim)),
            (1, 0, 2),
        )

        if self._qk_norm is not None:
            query_heads, key_heads = self._qk_norm.apply(query_heads, key_heads)

        if self._rope is not None:
            if rope_cos is None or rope_sin is None:
                raise ValueError("GroupedQueryAttention RoPE requires rope_cos and rope_sin")
            query_heads = self._rope(query_heads, rope_cos, rope_sin)
            key_heads = self._rope(key_heads, rope_cos, rope_sin)

        key_heads = repeat_kv(key_heads, self.num_kv_groups, axis=0)
        value_heads = repeat_kv(value_heads, self.num_kv_groups, axis=0)

        key_transposed = transpose(key_heads, (0, 2, 1))
        scores = matmul(query_heads, key_transposed)
        masked_scores = add(scores, causal_mask)
        attention_weights = self.softmax(masked_scores)
        context = matmul(attention_weights, value_heads)

        merged = reshape(
            transpose(context, (1, 0, 2)),
            (seq_len, self.hidden_size),
        )
        return self.o_proj(merged)
