"""Qwen3.8 family option validation (hybrid identities, residual width free)."""

from __future__ import annotations

import pytest

from zepto.empirical.models import get_model_family
from zepto.empirical.models.qwen38_core import Qwen38FamilyCore

_WORKED = {
    "num_cycles": 1,
    "hidden_size": 256,
    "delta_qk_heads": 4,
    "delta_v_group": 3,
    "delta_head_dim": 32,
    "attn_kv_heads": 2,
    "attn_gqa_group": 6,
    "attn_head_dim": 64,
    "ffn_mult": 5.25,
    "vocab_size": 1024,
}


def test_derive_worked_example() -> None:
    family = Qwen38FamilyCore()
    opts = family.derive_options(_WORKED)
    assert opts["num_layers"] == 4
    assert opts["delta_num_v_heads"] == 12
    assert opts["attn_num_q_heads"] == 12
    assert opts["intermediate_size"] == 1344
    assert opts["hidden_size"] == 256
    assert opts["hidden_size"] != opts["attn_num_q_heads"] * opts["attn_head_dim"]
    family.validate_options(opts)


def test_explicit_heads_allow_free_hidden_width() -> None:
    family = Qwen38FamilyCore()
    opts = family.derive_options(
        {
            "hidden_size": 256,
            "intermediate_size": 1344,
            "num_layers": 4,
            "vocab_size": 1024,
            "delta_num_qk_heads": 4,
            "delta_num_v_heads": 12,
            "delta_head_dim": 32,
            "attn_num_q_heads": 12,
            "attn_num_kv_heads": 2,
            "attn_head_dim": 64,
        }
    )
    family.validate_options(opts)
    assert opts["hidden_size"] == 256
    assert opts["attn_num_q_heads"] * opts["attn_head_dim"] == 768


def test_validate_incomplete_cycles() -> None:
    family = Qwen38FamilyCore()
    with pytest.raises(ValueError, match="complete 3:1 cycles"):
        family.validate_options(
            {
                "hidden_size": 256,
                "intermediate_size": 1344,
                "num_layers": 2,
                "vocab_size": 1024,
                "delta_num_qk_heads": 4,
                "delta_num_v_heads": 12,
                "delta_head_dim": 32,
                "attn_num_q_heads": 12,
                "attn_num_kv_heads": 2,
                "attn_head_dim": 64,
            }
        )


def test_get_model_family_qwen38() -> None:
    assert get_model_family("qwen38").model_id == "qwen38"


def test_infer_factory_is_text_only_hybrid() -> None:
    family = get_model_family("qwen38")
    opts = family.derive_options(_WORKED)
    module = family.build_zepto_infer_module_factory(opts, seq_len=32)(None)
    assert module.vision is None
    assert module.mtp is None
    mixers = [spec.mixer for spec in module._layer_specs]
    assert mixers == ["gated_delta_net", "gated_delta_net", "gated_delta_net", "attention"]
    assert mixers.count("gated_delta_net") == 3
