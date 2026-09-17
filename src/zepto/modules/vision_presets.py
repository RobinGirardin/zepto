"""Checkpoint-faithful vision tower factories (Qwen3-VL, Muse Glimmer, Gemma 4)."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor
from zepto.semantic import Add

from .axial_rope_materialize import AxialRoPEMaterialize
from .interpolated_position_grid import InterpolatedPositionGrid
from .learned_position_embedding_2d import LearnedPositionEmbedding2D
from .layer_norm import LayerNorm
from .multimodal_projector import gemma4_vision_projector, muse_glimmer_perception_adapter
from .patch_embed import GemmaLinearPatchEmbed, LinearPatchEmbed, qwen3_vl_patch_embed
from .patch_merger import PatchMerger2x2
from .pixel_shuffle import PixelShuffle2x2
from .position_aware_pool import PositionAwareAveragePool2x2
from .rope_materialize import RoPEMaterialize
from .vision_attention_config import (
    gemma4_vision_attention,
    muse_glimmer_vision_schedule,
    qwen3_vl_vision_attention,
)
from .vision_encoder_block import VisionEncoderBlock
from .vision_masks import vision_mask_for_config


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Frozen vision tower geometry and depth (per-checkpoint presets)."""

    patch_size: int
    hidden_size: int
    num_layers: int
    grid_h: int
    grid_w: int
    grid_t: int = 1


class Qwen3VLVisionTower(Module):
    """Conv3D tubelets → interp positions → vision blocks → 2×2 merger."""

    module_kind = "Qwen3VLVisionTower"

    def __init__(
        self,
        *,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        num_layers: int = 27,
        src_pos_h: int = 16,
        src_pos_w: int = 16,
    ) -> None:
        super().__init__()
        self.grid_t = grid_t
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_layers = num_layers
        self.seq_len = grid_t * grid_h * grid_w

        self.patch_embed = qwen3_vl_patch_embed(
            grid_t=grid_t, grid_h=grid_h, grid_w=grid_w
        )
        self.positions = InterpolatedPositionGrid(
            src_h=src_pos_h,
            src_w=src_pos_w,
            embed_dim=1152,
            out_h=grid_h,
            out_w=grid_w,
        )
        attn_cfg = qwen3_vl_vision_attention()
        self.blocks = [
            VisionEncoderBlock(
                attn_cfg,
                ffn_intermediate=4304,
                norm_kind="layer",
                ffn_activation="gelu_tanh",
            )
            for _ in range(num_layers)
        ]
        self.merger = PatchMerger2x2(
            hidden_size=1152,
            out_dim=5120,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        self.rope = RoPEMaterialize(self.seq_len, attn_cfg.head_dim)

    def forward(self, pixels: Tensor) -> Tensor:
        tokens = self.patch_embed(pixels)
        pos = self.positions()
        tokens = Add()(tokens, pos)  # type: ignore[assignment]
        mask = vision_mask_for_config(
            qwen3_vl_vision_attention(), self.seq_len
        )()
        rope_cos, rope_sin = self.rope()
        for block in self.blocks:
            tokens = block(tokens, mask, rope_cos, rope_sin)  # type: ignore[assignment]
        return self.merger(tokens)  # type: ignore[return-value]


class MuseGlimmerVisionTower(Module):
    """Linear patch → interp positions → scheduled vision → shuffle → adapter."""

    module_kind = "MuseGlimmerVisionTower"

    def __init__(
        self,
        *,
        grid_h: int = 14,
        grid_w: int = 14,
        num_layers: int = 50,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_layers = num_layers
        self.seq_len = grid_h * grid_w

        self.patch_embed = LinearPatchEmbed(
            patch_size=14,
            in_channels=3,
            temporal_frames=2,
            out_dim=1536,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        self.positions = InterpolatedPositionGrid(
            src_h=14,
            src_w=14,
            embed_dim=1536,
            out_h=grid_h,
            out_w=grid_w,
        )
        schedule = muse_glimmer_vision_schedule(num_layers)
        self.attn_schedule = schedule
        self.blocks = [
            VisionEncoderBlock(
                schedule[i],
                ffn_intermediate=8960,
                norm_kind="layer",
                ffn_activation="gelu",
            )
            for i in range(num_layers)
        ]
        self.post_norm = LayerNorm(1536)
        self.shuffle = PixelShuffle2x2(1536, grid_h=grid_h, grid_w=grid_w)
        self.adapter = muse_glimmer_perception_adapter()
        self.rope = RoPEMaterialize(self.seq_len, schedule[0].head_dim)

    def forward(self, pixels: Tensor) -> Tensor:
        tokens = self.patch_embed(pixels)
        pos = self.positions()
        tokens = Add()(tokens, pos)  # type: ignore[assignment]
        rope_cos, rope_sin = self.rope()
        for idx, block in enumerate(self.blocks):
            mask = vision_mask_for_config(self.attn_schedule[idx], self.seq_len)()
            tokens = block(tokens, mask, rope_cos, rope_sin)  # type: ignore[assignment]
        tokens = self.post_norm(tokens)
        tokens = self.shuffle(tokens)
        return self.adapter(tokens)  # type: ignore[return-value]


class Gemma4VisionPath(Module):
    """Linear 16×16 patch → axial positions → vision → pool → projector."""

    module_kind = "Gemma4VisionPath"

    def __init__(
        self,
        *,
        grid_h: int = 28,
        grid_w: int = 40,
        num_layers: int = 27,
    ) -> None:
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_layers = num_layers
        self.seq_len = grid_h * grid_w
        if (grid_h // 2) * (grid_w // 2) != 280:
            raise ValueError("Gemma4VisionPath requires pooled token count 280")

        self.patch_embed = GemmaLinearPatchEmbed(
            patch_size=16,
            in_channels=3,
            out_dim=1152,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        self.positions = LearnedPositionEmbedding2D(
            max_grid_h=grid_h,
            max_grid_w=grid_w,
            embed_dim=1152,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        attn_cfg = gemma4_vision_attention()
        self.blocks = [
            VisionEncoderBlock(
                attn_cfg,
                ffn_intermediate=1152 * 4,
                norm_kind="rms",
                ffn_activation="geglu",
            )
            for _ in range(num_layers)
        ]
        self.pool = PositionAwareAveragePool2x2(
            hidden_size=1152,
            out_dim=5376,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        self.projector = gemma4_vision_projector()
        self.rope = AxialRoPEMaterialize(grid_h, grid_w, attn_cfg.head_dim)

    def forward(self, pixels: Tensor) -> Tensor:
        tokens = self.patch_embed(pixels)
        pos = self.positions()
        tokens = Add()(tokens, pos)  # type: ignore[assignment]
        mask = vision_mask_for_config(gemma4_vision_attention(), self.seq_len)()
        rope_cos, rope_sin = self.rope()
        for block in self.blocks:
            tokens = block(tokens, mask, rope_cos, rope_sin)  # type: ignore[assignment]
        pooled = self.pool(tokens)
        return self.projector(pooled)  # type: ignore[return-value]


__all__ = [
    "Gemma4VisionPath",
    "MuseGlimmerVisionTower",
    "Qwen3VLVisionTower",
    "VisionConfig",
]
