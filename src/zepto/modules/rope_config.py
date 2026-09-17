"""RoPE configuration, validation, and checkpoint preset factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .attention_config import (
    AttentionConfig,
    gemma4_text_attention_layer,
    gpt_oss_attention_layer,
    laguna_xs_attention_layer,
    muse_glimmer_text_attention_layer,
)

RopeType = Literal[
    "default",
    "yarn",
    "llama3",
    "proportional",
    "mrope",
]


@dataclass(frozen=True, slots=True)
class RoPEConfig:
    """Frozen RoPE recipe. Runtime shapes depend on rotary_dim and rope_type."""

    head_dim: int
    rope_theta: float = 10_000.0
    rope_type: RopeType = "default"
    rotary_dim: int | None = None
    attention_scaling: float = 1.0

    # YaRN (rope_type == "yarn")
    factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_position_embeddings: int = 4096

    # Llama 3 (rope_type == "llama3")
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0

    # mRoPE (rope_type == "mrope")
    num_sections: int = 3
    mrope_section: tuple[int, ...] | None = None

    @property
    def resolved_rotary_dim(self) -> int:
        return self.rotary_dim if self.rotary_dim is not None else self.head_dim

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        rotary = self.resolved_rotary_dim
        if rotary <= 0 or rotary % 2 != 0 or rotary > self.head_dim:
            raise ValueError("rotary_dim must be positive even and ≤ head_dim")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.attention_scaling <= 0:
            raise ValueError("attention_scaling must be positive")
        if self.rope_type == "yarn":
            if self.factor < 1.0:
                raise ValueError(
                    "yarn requires factor >= 1 and positive original_max_position_embeddings"
                )
            if self.original_max_position_embeddings <= 0:
                raise ValueError(
                    "yarn requires factor >= 1 and positive original_max_position_embeddings"
                )
        if self.rope_type == "llama3":
            if self.low_freq_factor <= 0 or self.high_freq_factor <= 0:
                raise ValueError("llama3 factors must be positive")
        if self.rope_type == "mrope":
            if self.num_sections < 1:
                raise ValueError("mrope requires num_sections >= 1")
        if self.mrope_section is not None:
            if sum(self.mrope_section) != rotary // 2:
                raise ValueError("mrope_section must sum to rotary_dim // 2")


def llama_rope(*, head_dim: int, rope_theta: float = 10_000.0) -> RoPEConfig:
    return RoPEConfig(head_dim=head_dim, rope_theta=rope_theta, rope_type="default")


def gpt_oss_rope(*, head_dim: int = 64) -> RoPEConfig:
    return RoPEConfig(
        head_dim=head_dim,
        rope_type="yarn",
        factor=32.0,
        original_max_position_embeddings=4096,
        rope_theta=10_000.0,
    )


def laguna_rope_layer(*, layer_index: int, head_dim: int = 128) -> RoPEConfig:
    if layer_index % 4 == 0:
        return RoPEConfig(
            head_dim=head_dim,
            rope_type="yarn",
            rotary_dim=64,
            rope_theta=500_000.0,
            factor=32.0,
            original_max_position_embeddings=4096,
        )
    return RoPEConfig(head_dim=head_dim, rope_theta=10_000.0, rope_type="default")


def gemma4_rope_layer(*, layer_index: int, head_dim: int) -> RoPEConfig:
    if layer_index % 6 == 5:
        return RoPEConfig(
            head_dim=head_dim,
            rope_type="proportional",
            rotary_dim=head_dim // 4,
            rope_theta=1_000_000.0,
        )
    return RoPEConfig(head_dim=head_dim, rope_theta=10_000.0, rope_type="default")


def muse_glimmer_rope_layer(*, layer_index: int, head_dim: int = 128) -> RoPEConfig:
    if layer_index % 4 == 3:
        raise ValueError("muse global layer has NoPE; use muse_glimmer_layer_binding instead")
    return RoPEConfig(head_dim=head_dim, rope_theta=500_000.0, rope_type="default")


def qwen3_vl_mrope(*, head_dim: int = 256, rotary_dim: int = 64) -> RoPEConfig:
    return RoPEConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope_type="mrope",
        num_sections=3,
        mrope_section=(8, 12, 12),
    )


def apertus_rope(*, head_dim: int, rope_theta: float = 12_000_000.0) -> RoPEConfig:
    return RoPEConfig(
        head_dim=head_dim,
        rope_theta=rope_theta,
        rope_type="llama3",
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )


def granite_rope(*, head_dim: int = 128, rope_theta: float = 50_000_000.0) -> RoPEConfig:
    """Use checkpoint config value (50M); model card prose may differ."""
    return RoPEConfig(head_dim=head_dim, rope_theta=rope_theta, rope_type="default")


@dataclass(frozen=True, slots=True)
class LayerRoPEBinding:
    """Pairs one layer's attention policy with its RoPE recipe."""

    attention: AttentionConfig
    rope: RoPEConfig | None


def bind_rope(attention: AttentionConfig, rope: RoPEConfig | None) -> LayerRoPEBinding:
    if attention.position == "none":
        if rope is not None:
            raise ValueError("NoPE layer must have rope=None")
        return LayerRoPEBinding(attention=attention, rope=None)
    if rope is None:
        raise ValueError("RoPE layer requires a RoPEConfig")
    if rope.head_dim != attention.head_dim:
        raise ValueError(
            f"rope.head_dim ({rope.head_dim}) must match attention.head_dim "
            f"({attention.head_dim})"
        )
    return LayerRoPEBinding(attention=attention, rope=rope)


def gpt_oss_layer_binding(*, layer_index: int) -> LayerRoPEBinding:
    attn = gpt_oss_attention_layer(layer_index=layer_index)
    return bind_rope(attn, gpt_oss_rope(head_dim=attn.head_dim))


def laguna_layer_binding(*, layer_index: int) -> LayerRoPEBinding:
    attn = laguna_xs_attention_layer(layer_index=layer_index)
    return bind_rope(attn, laguna_rope_layer(layer_index=layer_index, head_dim=attn.head_dim))


def gemma4_layer_binding(*, layer_index: int) -> LayerRoPEBinding:
    attn = gemma4_text_attention_layer(layer_index=layer_index)
    if attn.position == "none":
        return bind_rope(attn, None)
    return bind_rope(
        attn, gemma4_rope_layer(layer_index=layer_index, head_dim=attn.head_dim)
    )


def muse_glimmer_layer_binding(*, layer_index: int) -> LayerRoPEBinding:
    attn = muse_glimmer_text_attention_layer(layer_index=layer_index)
    if attn.position == "none":
        return bind_rope(attn, None)
    return bind_rope(
        attn, muse_glimmer_rope_layer(layer_index=layer_index, head_dim=attn.head_dim)
    )


__all__ = [
    "LayerRoPEBinding",
    "RoPEConfig",
    "RopeType",
    "apertus_rope",
    "bind_rope",
    "gemma4_layer_binding",
    "gemma4_rope_layer",
    "gpt_oss_layer_binding",
    "gpt_oss_rope",
    "granite_rope",
    "laguna_layer_binding",
    "laguna_rope_layer",
    "llama_rope",
    "muse_glimmer_layer_binding",
    "muse_glimmer_rope_layer",
    "qwen3_vl_mrope",
]
