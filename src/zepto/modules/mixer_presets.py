"""Checkpoint-faithful hybrid layer schedules."""

from __future__ import annotations

from .attention_config import AttentionConfig, gated_gqa
from .layer_spec import LayerSpec
from .mixer_config import GatedDeltaNetConfig, Mamba2MixerConfig

_QWEN_HIDDEN = 5120
_QWEN_QK_HEADS = 16
_QWEN_V_HEADS = 48
_QWEN_HEAD = 128
_QWEN_FFN = 17408

_NEMOTRON_HIDDEN = 2688
_NEMOTRON_MAMBA = Mamba2MixerConfig(
    hidden_size=_NEMOTRON_HIDDEN,
    num_heads=64,
    head_dim=64,
    state_size=128,
    num_groups=8,
)
_NEMOTRON_ATTN = AttentionConfig(
    hidden_size=_NEMOTRON_HIDDEN,
    num_q_heads=32,
    num_kv_heads=2,
    head_dim=128,
    position="none",
)
_NEMOTRON_MAMBA_LAYERS = frozenset(
    {
        1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29, 31, 33, 36, 38, 40,
        42, 45, 47, 49, 51,
    }
)
_NEMOTRON_ATTN_LAYERS = frozenset({6, 13, 20, 27, 34, 43})


def qwen35_language_layer_specs() -> tuple[LayerSpec, ...]:
    """64 layers: repeat ×16 [DeltaNet, DeltaNet, DeltaNet, gated GQA + SwiGLU]."""
    delta = GatedDeltaNetConfig(
        hidden_size=_QWEN_HIDDEN,
        num_qk_heads=_QWEN_QK_HEADS,
        num_v_heads=_QWEN_V_HEADS,
        head_dim=_QWEN_HEAD,
    )
    attn = gated_gqa(
        _QWEN_HIDDEN,
        24,
        4,
        head_dim=256,
        gate="sigmoid",
    )
    cycle: list[LayerSpec] = [
        LayerSpec(mixer="gated_delta_net", gated_delta=delta, ffn="swiglu", swiglu_intermediate=_QWEN_FFN),
        LayerSpec(mixer="gated_delta_net", gated_delta=delta, ffn="swiglu", swiglu_intermediate=_QWEN_FFN),
        LayerSpec(mixer="gated_delta_net", gated_delta=delta, ffn="swiglu", swiglu_intermediate=_QWEN_FFN),
        LayerSpec(mixer="attention", attention=attn, ffn="swiglu", swiglu_intermediate=_QWEN_FFN),
    ]
    return tuple(cycle * 16)


def nemotron_h_layer_specs() -> tuple[LayerSpec, ...]:
    """52 Nemotron 3.5 Lightning layers (Mamba / attention / MoE)."""
    specs: list[LayerSpec] = []
    for index in range(52):
        layer_no = index + 1
        if layer_no in _NEMOTRON_MAMBA_LAYERS:
            specs.append(LayerSpec(mixer="mamba2", mamba2=_NEMOTRON_MAMBA))
        elif layer_no in _NEMOTRON_ATTN_LAYERS:
            specs.append(LayerSpec(mixer="attention", attention=_NEMOTRON_ATTN))
        else:
            specs.append(
                LayerSpec(
                    mixer="moe",
                    moe_hidden_size=_NEMOTRON_HIDDEN,
                )
            )
    return tuple(specs)


__all__ = ["nemotron_h_layer_specs", "qwen35_language_layer_specs"]
