"""Laguna XS decoder block: gated GQA + dense or sparse MoE FFN."""

from __future__ import annotations

from collections.abc import Callable

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .attention import FlexibleAttention
from .attention_config import AttentionConfig
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply
from .swiglu import SwiGLU


class LagunaDecoderBlock(Module):
    """Pre-norm attention residual; layer 0 dense SwiGLU else Laguna sparse MoE."""

    module_kind = "LagunaDecoderBlock"

    def __init__(
        self,
        layer_index: int,
        *,
        attention: AttentionConfig,
        hidden_size: int,
        rope: RoPEApply | None,
        dense_swiglu_intermediate: int = 8192,
        moe_factory: Callable[[], Module] | None = None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.pre_attn_norm = RMSNorm(hidden_size)
        self.pre_ffn_norm = RMSNorm(hidden_size)
        qk = QKNormRMSNorm(attention.head_dim)
        self.attn = FlexibleAttention(attention, qk_norm=qk, rope=rope)
        if layer_index == 0:
            self.ffn: Module = SwiGLU(hidden_size, dense_swiglu_intermediate)
        else:
            if moe_factory is None:
                raise ValueError("Laguna layers > 0 require moe_factory")
            self.ffn = moe_factory()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rope_cos: Tensor | None,
        rope_sin: Tensor | None,
    ) -> Tensor:
        residual = hidden_states
        normed = self.pre_attn_norm(hidden_states)
        attn_out = self.attn(normed, attention_mask, rope_cos, rope_sin)
        hidden = Add()(residual, attn_out)  # type: ignore[assignment]
        residual = hidden
        ffn_out = self.ffn(self.pre_ffn_norm(hidden))  # type: ignore[misc]
        return Add()(residual, ffn_out)  # type: ignore[return-value]


__all__ = ["LagunaDecoderBlock"]
