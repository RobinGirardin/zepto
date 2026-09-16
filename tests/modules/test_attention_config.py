"""Tests for AttentionConfig validation and checkpoint presets."""

from __future__ import annotations

import pytest

from zepto.modules.attention_config import (
    AttentionConfig,
    AttentionLayerTemplate,
    gemma4_text_attention_layer,
    gated_gqa,
    gpt_oss_attention_layer,
    granite_attention_layer,
    laguna_xs_attention_layer,
    llama_gqa,
    muse_glimmer_text_attention_layer,
    repeat_pattern,
)


def test_llama_gqa_standard_shape() -> None:
    cfg = llama_gqa(4096, 32, 8)
    assert cfg.q_proj_size == 4096
    assert cfg.kv_proj_size == 1024
    assert cfg.mask == "full"
    assert cfg.window_size is None


def test_llama_gqa_width_split_allowed() -> None:
    cfg = llama_gqa(5120, 24, 4, head_dim=256)
    assert cfg.q_proj_size == 6144
    assert cfg.kv_proj_size == 1024


def test_sliding_requires_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        AttentionConfig(
            hidden_size=2048,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            mask="sliding",
        )


def test_full_rejects_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        AttentionConfig(
            hidden_size=2048,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            mask="full",
            window_size=128,
        )


def test_gpt_oss_preset_sliding_and_full() -> None:
    sliding = gpt_oss_attention_layer(layer_index=0)
    assert sliding.q_proj_size == 4096
    assert sliding.kv_proj_size == 512
    assert sliding.mask == "sliding"
    assert sliding.window_size == 128
    assert sliding.softmax == "sink"
    assert sliding.qkv_bias is True

    full = gpt_oss_attention_layer(layer_index=1)
    assert full.mask == "full"
    assert full.window_size is None


def test_gemma4_local_and_global_presets() -> None:
    local = gemma4_text_attention_layer(layer_index=0)
    assert local.q_proj_size == 8192
    assert local.kv_proj_size == 4096
    assert local.mask == "sliding"
    assert local.window_size == 1024

    global_layer = gemma4_text_attention_layer(layer_index=5)
    assert global_layer.q_proj_size == 16384
    assert global_layer.kv_proj_size == 2048
    assert global_layer.kv_sharing == "v_equals_k"
    assert global_layer.mask == "full"


def test_muse_glimmer_presets() -> None:
    local = muse_glimmer_text_attention_layer(layer_index=0)
    assert local.q_proj_size == 4096
    assert local.mask == "sliding"
    assert local.window_size == 2048
    assert local.output_gate == "sigmoid"

    global_layer = muse_glimmer_text_attention_layer(layer_index=3)
    assert global_layer.position == "none"
    assert global_layer.mask == "full"


def test_laguna_presets() -> None:
    full = laguna_xs_attention_layer(layer_index=0)
    assert full.q_proj_size == 6144
    assert full.output_gate == "softplus"
    assert full.mask == "full"

    local = laguna_xs_attention_layer(layer_index=1)
    assert local.q_proj_size == 8192
    assert local.mask == "sliding"
    assert local.window_size == 512


def test_granite_preset() -> None:
    cfg = granite_attention_layer()
    assert cfg.q_proj_size == 4096
    assert cfg.hidden_size == 4096
    assert cfg.num_q_heads == 32


def test_gated_gqa_preset() -> None:
    cfg = gated_gqa(6656, 32, 2, head_dim=128, gate="sigmoid")
    assert cfg.output_gate == "sigmoid"
    assert cfg.q_proj_size == 4096


def test_repeat_pattern() -> None:
    templates = [
        AttentionLayerTemplate(llama_gqa(4096, 32, 8), repeat=2),
        AttentionLayerTemplate(gated_gqa(6656, 32, 2, head_dim=128), repeat=1),
    ]
    schedule = repeat_pattern(templates)
    assert len(schedule) == 3
    assert schedule[0].q_proj_size == 4096
    assert schedule[2].output_gate == "sigmoid"
