"""Attention configuration dataclass and checkpoint preset factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MaskPolicy = Literal["full", "sliding"]
PositionPolicy = Literal["rope", "none"]
KvSharingPolicy = Literal["separate", "v_equals_k"]
OutputGatePolicy = Literal["none", "sigmoid", "softplus"]
SoftmaxPolicy = Literal["standard", "sink"]


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Frozen per-layer attention policy and projection geometry."""

    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    mask: MaskPolicy = "full"
    window_size: int | None = None

    position: PositionPolicy = "rope"
    kv_sharing: KvSharingPolicy = "separate"
    output_gate: OutputGatePolicy = "none"
    softmax: SoftmaxPolicy = "standard"

    qkv_bias: bool = False
    o_bias: bool = False
    q_post_norm_scale: float | None = None

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
        if self.num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.mask == "sliding":
            if self.window_size is None or self.window_size <= 0:
                raise ValueError(
                    "sliding mask requires window_size to be a positive integer"
                )
        elif self.mask == "full" and self.window_size is not None:
            raise ValueError("full mask requires window_size to be None")
        if self.q_post_norm_scale is not None and self.q_post_norm_scale <= 0:
            raise ValueError("q_post_norm_scale must be positive when set")


def llama_gqa(
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    head_dim: int | None = None,
) -> AttentionConfig:
    """Llama/Apertus: ``q_proj_size == hidden_size == o_proj`` input width."""
    resolved_head_dim = (
        head_dim if head_dim is not None else hidden_size // num_q_heads
    )
    return AttentionConfig(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=resolved_head_dim,
    )


def gated_gqa(
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    *,
    head_dim: int,
    gate: Literal["sigmoid", "softplus"] = "sigmoid",
    **kwargs: object,
) -> AttentionConfig:
    """Qwen3-Next / Muse / Laguna gated dot-product attention preset."""
    _ = kwargs
    return AttentionConfig(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        output_gate=gate,
    )


def gpt_oss_attention_layer(*, layer_index: int) -> AttentionConfig:
    """GPT-OSS 20B attention layer (alternating 128-token sliding / full)."""
    sliding = layer_index % 2 == 0
    return AttentionConfig(
        hidden_size=2880,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=64,
        mask="sliding" if sliding else "full",
        window_size=128 if sliding else None,
        softmax="sink",
        qkv_bias=True,
        o_bias=True,
    )


def gemma4_text_attention_layer(*, layer_index: int) -> AttentionConfig:
    """Gemma 4 31B text layer (5 local sliding + 1 global per sextet)."""
    global_layer = layer_index % 6 == 5
    if global_layer:
        return AttentionConfig(
            hidden_size=5376,
            num_q_heads=32,
            num_kv_heads=4,
            head_dim=512,
            mask="full",
            kv_sharing="v_equals_k",
        )
    return AttentionConfig(
        hidden_size=5376,
        num_q_heads=32,
        num_kv_heads=16,
        head_dim=256,
        mask="sliding",
        window_size=1024,
    )


def muse_glimmer_text_attention_layer(*, layer_index: int) -> AttentionConfig:
    """Muse Glimmer 30B text layer (3 local + 1 global per quartet)."""
    global_layer = layer_index % 4 == 3
    return AttentionConfig(
        hidden_size=6656,
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=128,
        mask="full" if global_layer else "sliding",
        window_size=None if global_layer else 2048,
        position="none" if global_layer else "rope",
        output_gate="sigmoid",
        q_post_norm_scale=3.87,
    )


def laguna_xs_attention_layer(*, layer_index: int) -> AttentionConfig:
    """Laguna XS 2.1 layer (1 full + 3 local per quartet)."""
    full_layer = layer_index % 4 == 0
    if full_layer:
        return AttentionConfig(
            hidden_size=2048,
            num_q_heads=48,
            num_kv_heads=8,
            head_dim=128,
            mask="full",
            output_gate="softplus",
        )
    return AttentionConfig(
        hidden_size=2048,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        window_size=512,
        output_gate="softplus",
    )


def granite_attention_layer() -> AttentionConfig:
    """IBM Granite 4.2 30B full causal GQA (supports batched inputs)."""
    return AttentionConfig(
        hidden_size=4096,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


@dataclass(frozen=True, slots=True)
class AttentionLayerTemplate:
    """Repeatable attention config block for model schedules."""

    config: AttentionConfig
    repeat: int = 1


def repeat_pattern(templates: list[AttentionLayerTemplate]) -> list[AttentionConfig]:
    """Expand a list of templates into a flat per-layer config schedule."""
    result: list[AttentionConfig] = []
    for template in templates:
        if template.repeat <= 0:
            raise ValueError("template repeat must be positive")
        result.extend([template.config] * template.repeat)
    return result


__all__ = [
    "AttentionConfig",
    "AttentionLayerTemplate",
    "MaskPolicy",
    "OutputGatePolicy",
    "PositionPolicy",
    "KvSharingPolicy",
    "SoftmaxPolicy",
    "gemma4_text_attention_layer",
    "gated_gqa",
    "gpt_oss_attention_layer",
    "granite_attention_layer",
    "laguna_xs_attention_layer",
    "llama_gqa",
    "muse_glimmer_text_attention_layer",
    "repeat_pattern",
]
