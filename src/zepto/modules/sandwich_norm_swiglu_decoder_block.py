"""Decoder block with pre/post RMSNorm around attention and SwiGLU (Gemma/Muse layout)."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .attention import FlexibleAttention
from .attention_config import AttentionConfig
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply
from .swiglu import SwiGLU


@dataclass(frozen=True, slots=True)
class SandwichNormStyle:
    """Per-norm flags for the four sandwich RMSNorm slots."""

    center: bool = False
    elementwise_affine: bool = True


MUSE_LANGUAGE_SANDWICH_NORM = SandwichNormStyle(
    center=True, elementwise_affine=True
)
GEMMA4_LANGUAGE_SANDWICH_NORM = SandwichNormStyle(
    center=False, elementwise_affine=True
)


def _sandwich_rms(hidden_size: int, style: SandwichNormStyle) -> RMSNorm:
    return RMSNorm(
        hidden_size,
        center=style.center,
        elementwise_affine=style.elementwise_affine,
    )


class SandwichNormSwiGLUDecoderBlock(Module):
    """Pre/post RMSNorm around attention and FFN with residual adds (HF sandwich)."""

    module_kind = "SandwichNormSwiGLUDecoderBlock"

    def __init__(
        self,
        attention: AttentionConfig,
        *,
        swiglu_intermediate: int,
        norm_style: SandwichNormStyle = GEMMA4_LANGUAGE_SANDWICH_NORM,
        rope: RoPEApply | None = None,
        qk_norm: QKNormRMSNorm | None = None,
    ) -> None:
        super().__init__()
        hidden = attention.hidden_size
        self.pre_attn_norm = _sandwich_rms(hidden, norm_style)
        self.post_attn_norm = _sandwich_rms(hidden, norm_style)
        self.pre_ffn_norm = _sandwich_rms(hidden, norm_style)
        self.post_ffn_norm = _sandwich_rms(hidden, norm_style)
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
        attn_out = self.attn(
            self.pre_attn_norm(hidden_states),
            attention_mask,
            rope_cos,
            rope_sin,
        )
        hidden = Add()(residual, self.post_attn_norm(attn_out))  # type: ignore[assignment]
        residual = hidden
        ffn_out = self.ffn(self.pre_ffn_norm(hidden))
        return Add()(residual, self.post_ffn_norm(ffn_out))  # type: ignore[return-value]


__all__ = [
    "GEMMA4_LANGUAGE_SANDWICH_NORM",
    "MUSE_LANGUAGE_SANDWICH_NORM",
    "SandwichNormStyle",
    "SandwichNormSwiGLUDecoderBlock",
]
