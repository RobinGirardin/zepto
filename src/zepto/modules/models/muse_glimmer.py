"""Muse Glimmer 30B vision-language full model."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor

from zepto.modules._internal._batch import expect_token_ids_rank
from zepto.modules.attention.decoder_attention_context import DecoderAttentionContext
from zepto.modules.layers.embedding import Embedding
from zepto.modules.multimodal.multimodal_language_model import embed_multimodal_sequence
from zepto.modules.multimodal.multimodal_sequence_builder import MultimodalSequenceBuilder
from zepto.modules.output.output_presets import muse_glimmer_lm_output
from zepto.modules.layers.qk_norm import QKNormRMSNorm
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_apply import RoPEApply
from zepto.modules.position.rope_config import muse_glimmer_layer_binding
from zepto.modules.blocks.sandwich_norm_swiglu_decoder_block import (
    MUSE_LANGUAGE_SANDWICH_NORM,
    SandwichNormSwiGLUDecoderBlock,
)
from zepto.modules.vision.vision_presets import MuseGlimmerVisionTower


@dataclass(frozen=True, slots=True)
class MuseGlimmerConfig:
    """Architecture constants for Muse Glimmer 30B."""

    hidden_size: int = 6656
    swiglu_intermediate: int = 19968
    num_layers: int = 52
    vocab_size: int = 202048


MUSE_GLIMMER_30B = MuseGlimmerConfig()


def _rope_apply_for_binding(layer_index: int) -> RoPEApply | None:
    binding = muse_glimmer_layer_binding(layer_index=layer_index)
    if binding.rope is None:
        return None
    cfg = binding.rope
    return RoPEApply(cfg.head_dim)


class MuseGlimmer(Module):
    """Vision tower + language trunk with centered sandwich RMSNorm blocks."""

    module_kind = "MuseGlimmer"

    def __init__(
        self,
        *,
        config: MuseGlimmerConfig = MUSE_GLIMMER_30B,
        seq_len: int,
        num_layers: int | None = None,
        include_vision: bool = True,
        vision_num_layers: int = 2,
        grid_h: int = 14,
        grid_w: int = 14,
    ) -> None:
        super().__init__()
        cfg = config
        layers = num_layers if num_layers is not None else cfg.num_layers
        if layers <= 0 or seq_len <= 0:
            raise ValueError("num_layers and seq_len must be positive")
        self.seq_len = seq_len
        self.config = cfg
        self.vision: MuseGlimmerVisionTower | None = None
        self.sequence_builder: MultimodalSequenceBuilder | None = None
        if include_vision:
            self.vision = MuseGlimmerVisionTower(
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
            layer_binding=lambda i: muse_glimmer_layer_binding(layer_index=i),
            num_layers=layers,
        )
        self.embed_norm = RMSNorm(
            cfg.hidden_size, center=True, elementwise_affine=False
        )
        self.blocks: list[SandwichNormSwiGLUDecoderBlock] = []
        for index in range(layers):
            binding = muse_glimmer_layer_binding(layer_index=index)
            attn = binding.attention
            qk = QKNormRMSNorm(attn.head_dim)
            block = SandwichNormSwiGLUDecoderBlock(
                attn,
                swiglu_intermediate=cfg.swiglu_intermediate,
                norm_style=MUSE_LANGUAGE_SANDWICH_NORM,
                rope=_rope_apply_for_binding(index),
                qk_norm=qk,
            )
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_output = muse_glimmer_lm_output(
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size
        )

    @classmethod
    def muse_glimmer_30b(cls, seq_len: int) -> MuseGlimmer:
        return cls(config=MUSE_GLIMMER_30B, seq_len=seq_len)

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
        expect_token_ids_rank(token_ids)
        hidden = embed_multimodal_sequence(
            self,
            token_ids,
            placeholder_indices=placeholder_indices,
            vision_pixels=vision_pixels,
        )
        return self._language_trunk(self.embed_norm(hidden))

    def vision_forward(self, vision_pixels: Tensor) -> Tensor:
        if self.vision is None:
            raise ValueError("vision_forward requires include_vision=True")
        return self.vision(vision_pixels)  # type: ignore[return-value]


__all__ = ["MUSE_GLIMMER_30B", "MuseGlimmer", "MuseGlimmerConfig"]
