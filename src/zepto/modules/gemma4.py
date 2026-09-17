"""Gemma 4 31B vision-language full model."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor

from .decoder_attention_context import DecoderAttentionContext
from .embedding import Embedding
from .multimodal_language_model import embed_multimodal_sequence
from .multimodal_sequence_builder import MultimodalSequenceBuilder
from .output_presets import gemma4_lm_output
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply
from .rope_config import gemma4_layer_binding
from .swiglu_decoder_block import SwiGLUDecoderBlock
from .vision_presets import Gemma4VisionPath


@dataclass(frozen=True, slots=True)
class Gemma4Config:
    """Architecture constants for Gemma 4 31B IT."""

    hidden_size: int = 5376
    swiglu_intermediate: int = 21504
    num_layers: int = 60
    vocab_size: int = 262144


GEMMA4_31B_IT = Gemma4Config()


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
    """Vision tower + language trunk; v1 uses two pre-norms per block (Granite-shaped).

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
        self.blocks: list[SwiGLUDecoderBlock] = []
        for index in range(layers):
            binding = gemma4_layer_binding(layer_index=index)
            attn = binding.attention
            qk = QKNormRMSNorm(attn.head_dim)
            block = SwiGLUDecoderBlock(
                attn,
                swiglu_intermediate=cfg.swiglu_intermediate,
                rope=_rope_apply_for_binding(index),
                qk_norm=qk,
            )
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_output = gemma4_lm_output(
            tied_weight=self.embedding.weight,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )

    @classmethod
    def gemma4_31b_it(cls, seq_len: int) -> Gemma4:
        return cls(config=GEMMA4_31B_IT, seq_len=seq_len)

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
        return self._language_trunk(hidden)

    def vision_forward(self, vision_pixels: Tensor) -> Tensor:
        if self.vision is None:
            raise ValueError("vision_forward requires include_vision=True")
        return self.vision(vision_pixels)  # type: ignore[return-value]


__all__ = ["GEMMA4_31B_IT", "Gemma4", "Gemma4Config"]
