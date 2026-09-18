"""Gemma 4 31B vision-language full model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from zepto.compose import Module, Tensor
from zepto.semantic import Multiply

from zepto.modules.attention.decoder_attention_context import DecoderAttentionContext
from zepto.modules.layers.embedding import Embedding
from zepto.modules.multimodal.multimodal_language_model import embed_multimodal_sequence
from zepto.modules.multimodal.multimodal_sequence_builder import MultimodalSequenceBuilder
from zepto.modules.output.output_presets import gemma4_lm_output
from zepto.modules.layers.qk_norm import QKNormRMSNorm
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_apply import RoPEApply
from zepto.modules.position.rope_config import gemma4_layer_binding
from zepto.modules.blocks.sandwich_norm_swiglu_decoder_block import (
    GEMMA4_LANGUAGE_SANDWICH_NORM,
    SandwichNormSwiGLUDecoderBlock,
)
from zepto.modules.vision.vision_presets import Gemma4VisionPath


@dataclass(frozen=True, slots=True)
class Gemma4Config:
    """Architecture constants for Gemma 4 31B IT."""

    hidden_size: int = 5376
    swiglu_intermediate: int = 21504
    num_layers: int = 60
    vocab_size: int = 262144
    embed_scale: float | None = None


GEMMA4_31B_IT = Gemma4Config(embed_scale=math.sqrt(5376))


def _rope_apply_for_binding(layer_index: int) -> RoPEApply | None:
    binding = gemma4_layer_binding(layer_index=layer_index)
    if binding.rope is None:
        return None
    cfg = binding.rope
    rotary = cfg.resolved_rotary_dim
    if rotary == cfg.head_dim:
        return RoPEApply(cfg.head_dim)
    return RoPEApply(cfg.head_dim, rotary_dim=rotary)


class Gemma4(Module):
    """Vision tower + language trunk with sandwich RMSNorm blocks.

    Text-only ``forward(token_ids)`` skips vision compute but retains tower weights
    when ``include_vision=True``. Tied embedding/LM head per checkpoint.
    """

    module_kind = "Gemma4"

    def __init__(
        self,
        *,
        config: Gemma4Config = GEMMA4_31B_IT,
        seq_len: int,
        num_layers: int | None = None,
        include_vision: bool = True,
        vision_num_layers: int = 2,
        grid_h: int = 28,
        grid_w: int = 40,
    ) -> None:
        super().__init__()
        cfg = config
        layers = num_layers if num_layers is not None else cfg.num_layers
        if layers <= 0 or seq_len <= 0:
            raise ValueError("num_layers and seq_len must be positive")
        self.seq_len = seq_len
        self.config = cfg
        self.vision: Gemma4VisionPath | None = None
        self.sequence_builder: MultimodalSequenceBuilder | None = None
        if include_vision:
            self.vision = Gemma4VisionPath(
                grid_h=grid_h, grid_w=grid_w, num_layers=vision_num_layers
            )
            self.sequence_builder = MultimodalSequenceBuilder(
                cfg.hidden_size, cfg.vocab_size
            )
            self.embedding = self.sequence_builder.embedding
        else:
            self.embedding = Embedding(cfg.hidden_size, cfg.vocab_size)
        self.attn_context = DecoderAttentionContext(
            seq_len,
            layer_binding=lambda i: gemma4_layer_binding(layer_index=i),
            num_layers=layers,
        )
        self.blocks: list[SandwichNormSwiGLUDecoderBlock] = []
        for index in range(layers):
            binding = gemma4_layer_binding(layer_index=index)
            attn = binding.attention
            qk = QKNormRMSNorm(attn.head_dim)
            block = SandwichNormSwiGLUDecoderBlock(
                attn,
                swiglu_intermediate=cfg.swiglu_intermediate,
                norm_style=GEMMA4_LANGUAGE_SANDWICH_NORM,
                rope=_rope_apply_for_binding(index),
                qk_norm=qk,
            )
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self._embed_scale: Tensor | None = None
        if cfg.embed_scale is not None:
            self._embed_scale = Tensor(
                shape=(1,),
                semantic_type="embed_scale",
                requires_grad=False,
            )
        self.lm_output = gemma4_lm_output(
            tied_weight=self.embedding.weight,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    @classmethod
    def gemma4_31b_it(cls, seq_len: int) -> Gemma4:
        return cls(config=GEMMA4_31B_IT, seq_len=seq_len)

    def _scale_embed(self, hidden: Tensor) -> Tensor:
        if self._embed_scale is None:
            return hidden
        return Multiply()(hidden, self._embed_scale)  # type: ignore[return-value]

    def _language_trunk(self, hidden: Tensor) -> Tensor:
        for index, block in enumerate(self.blocks):
            ctx = self.attn_context.for_layer(index)
            hidden = block(
                hidden,
                ctx.mask,
                ctx.cos,
                ctx.sin,
            )  # type: ignore[assignment]
        return self.lm_output(self.final_norm(hidden))  # type: ignore[return-value]

    def forward(
        self,
        token_ids: Tensor,
        placeholder_indices: Tensor | None = None,
        vision_pixels: Tensor | None = None,
    ) -> Tensor:
        hidden = embed_multimodal_sequence(
            self,
            token_ids,
            placeholder_indices=placeholder_indices,
            vision_pixels=vision_pixels,
        )
        return self._language_trunk(self._scale_embed(hidden))

    def vision_forward(self, vision_pixels: Tensor) -> Tensor:
        if self.vision is None:
            raise ValueError("vision_forward requires include_vision=True")
        return self.vision(vision_pixels)  # type: ignore[return-value]


__all__ = ["GEMMA4_31B_IT", "Gemma4", "Gemma4Config"]
