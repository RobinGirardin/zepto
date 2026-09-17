"""Vision transformer encoder block (norm → attn → FFN)."""

from __future__ import annotations

from typing import Literal

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .attention import FlexibleAttention
from .ffn import FFN
from .geglu import GeGLU
from .gelu import GELUErf, GELUTanh
from .layer_norm import LayerNorm
from .linear import Linear
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply
from .vision_attention_config import VisionAttentionConfig


class VisionEncoderBlock(Module):
    """Pre-norm vision block using :class:`FlexibleAttention` and external masks."""

    module_kind = "VisionEncoderBlock"

    def __init__(
        self,
        config: VisionAttentionConfig,
        *,
        ffn_intermediate: int,
        norm_kind: Literal["layer", "rms"] = "layer",
        ffn_activation: Literal["gelu_tanh", "gelu", "geglu"] = "gelu_tanh",
    ) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size

        if norm_kind == "layer":
            self.pre_attn_norm = LayerNorm(d)
            self.pre_ffn_norm = LayerNorm(d)
        else:
            self.pre_attn_norm = RMSNorm(d)
            self.pre_ffn_norm = RMSNorm(d)

        qk = QKNormRMSNorm(config.head_dim) if config.value_norm == "rms" else None
        v_norm = RMSNorm(config.head_dim) if config.value_norm == "rms" else None
        rope = RoPEApply(config.head_dim) if config.position == "rope" else None

        self.attn = FlexibleAttention(
            config.to_attention_config(),
            qk_norm=qk,
            rope=rope,
            v_norm=v_norm,
        )

        if ffn_activation == "geglu":
            self.ffn: Module = GeGLU(d, ffn_intermediate)
        elif ffn_activation == "gelu":
            self.ffn = FFN(d, ffn_intermediate, activation=GELUErf())
        else:
            self.ffn = FFN(d, ffn_intermediate, activation=GELUTanh())

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        normed = self.pre_attn_norm(hidden_states)
        attn_out = self.attn(normed, attention_mask, rope_cos, rope_sin)
        hidden_states = Add()(residual, attn_out)  # type: ignore[assignment]

        residual = hidden_states
        normed = self.pre_ffn_norm(hidden_states)
        ffn_out = self.ffn(normed)
        return Add()(residual, ffn_out)  # type: ignore[return-value]


class VisionMLP(Module):
    """Two-layer vision MLP without residual (Muse-style explicit linears)."""

    module_kind = "VisionMLP"

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Literal["gelu_tanh", "gelu"] = "gelu",
    ) -> None:
        super().__init__()
        self.up = Linear(hidden_size, intermediate_size)
        self.down = Linear(intermediate_size, hidden_size)
        self._activation = GELUErf() if activation == "gelu" else GELUTanh()

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.up(x)
        activated = self._activation(hidden)
        return self.down(activated)  # type: ignore[return-value]


__all__ = ["VisionEncoderBlock", "VisionMLP"]
