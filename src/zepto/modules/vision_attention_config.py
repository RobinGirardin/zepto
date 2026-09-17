"""Vision self-attention configuration and checkpoint presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .attention_config import AttentionConfig

VisionMaskPolicy = Literal["bidirectional", "sliding_1d", "window_2d"]
VisionPositionPolicy = Literal["rope", "none"]
VisionValueNormPolicy = Literal["none", "rms"]


@dataclass(frozen=True, slots=True)
class VisionAttentionConfig:
    """Frozen vision attention recipe (always non-causal via external mask modules)."""

    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    mask: VisionMaskPolicy = "bidirectional"
    window_size: int | None = None

    position: VisionPositionPolicy = "rope"
    value_norm: VisionValueNormPolicy = "none"

    qkv_bias: bool = False
    o_bias: bool = False

    @property
    def q_proj_size(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_proj_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def num_kv_groups(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("head counts and head_dim must be positive")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.mask in ("sliding_1d", "window_2d"):
            if self.window_size is None or self.window_size <= 0:
                raise ValueError(f"{self.mask} requires a positive window_size")
        elif self.window_size is not None:
            raise ValueError(f"{self.mask} requires window_size to be None")

    def to_attention_config(self) -> AttentionConfig:
        """Map to :class:`FlexibleAttention` config (mask schedule is external)."""
        return AttentionConfig(
            hidden_size=self.hidden_size,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mask="full",
            position=self.position,
            qkv_bias=self.qkv_bias,
            o_bias=self.o_bias,
        )


def qwen3_vl_vision_attention() -> VisionAttentionConfig:
    return VisionAttentionConfig(
        hidden_size=1152,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=72,
    )


def muse_glimmer_vision_attention(*, layer_index: int) -> VisionAttentionConfig:
    global_layer = layer_index % 4 == 3
    return VisionAttentionConfig(
        hidden_size=1536,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=96,
        mask="bidirectional" if global_layer else "sliding_1d",
        window_size=None if global_layer else 112,
    )


def gemma4_vision_attention() -> VisionAttentionConfig:
    return VisionAttentionConfig(
        hidden_size=1152,
        num_q_heads=16,
        num_kv_heads=16,
        head_dim=72,
        value_norm="rms",
    )


def muse_glimmer_vision_schedule(num_layers: int = 50) -> list[VisionAttentionConfig]:
    return [
        muse_glimmer_vision_attention(layer_index=i) for i in range(num_layers)
    ]


__all__ = [
    "VisionAttentionConfig",
    "VisionMaskPolicy",
    "VisionPositionPolicy",
    "VisionValueNormPolicy",
    "gemma4_vision_attention",
    "muse_glimmer_vision_attention",
    "muse_glimmer_vision_schedule",
    "qwen3_vl_vision_attention",
]
