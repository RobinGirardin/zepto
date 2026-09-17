"""Pre-norm SwiGLU decoder block with ``FlexibleAttention``."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .attention import FlexibleAttention
from .attention_config import AttentionConfig
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply
from .swiglu import SwiGLU


class SwiGLUDecoderBlock(Module):
    """Pre-RMSNorm attention residual → pre-RMSNorm SwiGLU residual."""

    module_kind = "SwiGLUDecoderBlock"

    def __init__(
        self,
        attention: AttentionConfig,
        *,
        swiglu_intermediate: int,
        rope: RoPEApply | None = None,
        qk_norm: QKNormRMSNorm | None = None,
    ) -> None:
        super().__init__()
        hidden = attention.hidden_size
        self.pre_attn_norm = RMSNorm(hidden)
        self.pre_ffn_norm = RMSNorm(hidden)
        self.attn = FlexibleAttention(attention, qk_norm=qk_norm, rope=rope)
        self.ffn = SwiGLU(hidden, swiglu_intermediate)

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
        normed = self.pre_ffn_norm(hidden)
        ffn_out = self.ffn(normed)
        return Add()(residual, ffn_out)  # type: ignore[return-value]


__all__ = ["SwiGLUDecoderBlock"]
