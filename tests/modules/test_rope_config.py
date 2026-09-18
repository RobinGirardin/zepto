"""RoPEConfig validation and preset factory tests."""

from __future__ import annotations

import pytest

from zepto.modules.position.rope_config import (
    RoPEConfig,
    gemma4_rope_layer,
    gpt_oss_rope,
    laguna_layer_binding,
    laguna_rope_layer,
    llama_rope,
    muse_glimmer_layer_binding,
    muse_glimmer_rope_layer,
)


def test_llama_rope_defaults() -> None:
    cfg = llama_rope(head_dim=128)
    assert cfg.resolved_rotary_dim == 128
    assert cfg.rope_type == "default"


def test_gpt_oss_rope_yarn() -> None:
    cfg = gpt_oss_rope()
    assert cfg.rope_type == "yarn"
    assert cfg.factor == 32.0
    assert cfg.original_max_position_embeddings == 4096


def test_laguna_rope_layer_schedule() -> None:
    partial = laguna_rope_layer(layer_index=0)
    assert partial.rotary_dim == 64
    assert partial.rope_type == "yarn"
    full = laguna_rope_layer(layer_index=1)
    assert full.rotary_dim is None
    assert full.resolved_rotary_dim == 128


def test_gemma4_proportional_global() -> None:
    cfg = gemma4_rope_layer(layer_index=5, head_dim=512)
    assert cfg.resolved_rotary_dim == 128
    assert cfg.rope_type == "proportional"


def test_muse_glimmer_rope_layer() -> None:
    muse_glimmer_rope_layer(layer_index=0)
    with pytest.raises(ValueError, match="NoPE"):
        muse_glimmer_rope_layer(layer_index=3)


def test_layer_bindings() -> None:
    assert laguna_layer_binding(layer_index=0).rope is not None
    assert muse_glimmer_layer_binding(layer_index=3).rope is None


def test_odd_rotary_dim_raises() -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        RoPEConfig(head_dim=128, rotary_dim=63)
