"""Apertus decoder block module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add
from .ffn import FFN
from .gqa import GroupedQueryAttention
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply


class ApertusDecoderBlock(Module):
    """Pre-RMSNorm Apertus block: attention residual → FFN residual."""

    module_kind = "ApertusDecoderBlock"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        head_dim: int | None = None,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        resolved_head_dim = (
            head_dim if head_dim is not None else hidden_size // num_heads
        )

        self.pre_attn_norm = RMSNorm(hidden_size)
        rope = RoPEApply(resolved_head_dim)
        qk = QKNormRMSNorm(resolved_head_dim) if qk_norm else None
        self.attn = GroupedQueryAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim=resolved_head_dim,
            qk_norm=qk,
            rope=rope,
        )
        self.pre_ffn_norm = RMSNorm(hidden_size)
        self.ffn = FFN(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        causal_mask: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        residual = hidden_states
        normed = self.pre_attn_norm(hidden_states)
        attention_out = self.attn(
            normed,
            causal_mask,
            rope_cos,
            rope_sin,
        )
        hidden_states = Add()(residual, attention_out)  # type: ignore[assignment]

        residual = hidden_states
        normed = self.pre_ffn_norm(hidden_states)
        ffn_out = self.ffn(normed)
        return Add()(residual, ffn_out)  # type: ignore[return-value]
