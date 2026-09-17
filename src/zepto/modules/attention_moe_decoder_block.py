"""GPT-OSS-shaped attention + MoE decoder block."""

from __future__ import annotations

from collections.abc import Callable

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .attention import FlexibleAttention
from .gpt_oss_moe_block import GptOssMoEBlock
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply
from .rope_config import gpt_oss_layer_binding


class AttentionMoEDecoderBlock(Module):
    """RMSNorm → GQA → residual → RMSNorm → MoE → residual."""

    module_kind = "AttentionMoEDecoderBlock"

    def __init__(
        self,
        layer_index: int,
        *,
        seq_len: int,
        moe_factory: Callable[[], GptOssMoEBlock],
    ) -> None:
        super().__init__()
        binding = gpt_oss_layer_binding(layer_index=layer_index)
        attn_cfg = binding.attention
        self.pre_attn_norm = RMSNorm(attn_cfg.hidden_size)
        self.pre_moe_norm = RMSNorm(attn_cfg.hidden_size)
        self.attn = FlexibleAttention(
            attn_cfg,
            rope=RoPEApply(attn_cfg.head_dim),
        )
        self.moe = moe_factory()
        self._layer_index = layer_index

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        residual = hidden_states
        normed = self.pre_attn_norm(hidden_states)
        attn_out = self.attn(normed, attention_mask, rope_cos, rope_sin)
        hidden = Add()(residual, attn_out)  # type: ignore[assignment]
        residual = hidden
        moe_out = self.moe(self.pre_moe_norm(hidden))  # type: ignore[misc]
        return Add()(residual, moe_out)  # type: ignore[return-value]


__all__ = ["AttentionMoEDecoderBlock"]
