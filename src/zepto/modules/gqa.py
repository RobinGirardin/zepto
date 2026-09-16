"""Grouped-query attention module (Llama/Apertus preset)."""

from __future__ import annotations

from zepto.compose import Tensor

from .attention import FlexibleAttention
from .attention_config import llama_gqa
from .qk_norm import QKNormRMSNorm
from .rope_apply import RoPEApply


class GroupedQueryAttention(FlexibleAttention):
    """Masked grouped-query self-attention (decoder-only, eager decomposition).

    Backward-compatible preset over ``FlexibleAttention`` with ``llama_gqa``
    configuration. Preserves the historical constructor and ``causal_mask``
    parameter name for Apertus integration.
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
        if hidden_size <= 0 or num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError(
                "hidden_size, num_heads, and num_kv_heads must be positive"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )
        resolved_head_dim = (
            head_dim if head_dim is not None else hidden_size // num_heads
        )
        if num_heads * resolved_head_dim != hidden_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must equal "
                f"num_heads * head_dim ({num_heads} * {resolved_head_dim})"
            )
        config = llama_gqa(
            hidden_size, num_heads, num_kv_heads, head_dim=head_dim
        )
        super().__init__(config, qk_norm=qk_norm, rope=rope)

    def forward(
        self,
        hidden_states: Tensor,
        causal_mask: Tensor,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        return super().forward(
            hidden_states, causal_mask, rope_cos, rope_sin
        )
