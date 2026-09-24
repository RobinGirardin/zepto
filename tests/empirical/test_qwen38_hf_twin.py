"""HF Qwen3.8 twin: scored_window MPE and text-only causal-LM contract."""

from __future__ import annotations

import importlib.util

import pytest

from zepto.empirical.models import get_model_family
from zepto.empirical.models.qwen38 import (
    SCORED_WINDOW_MIN_MPE,
    Qwen38Family,
    qwen38_hf_max_position_embeddings,
)

pytest.importorskip("transformers")

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
except ImportError:  # pragma: no cover
    Qwen3_5ForCausalLM = None
    Qwen3_5TextConfig = None

if Qwen3_5TextConfig is None:
    pytest.skip("transformers Qwen3_5TextConfig is unavailable", allow_module_level=True)

_WORKED_OPTS = {
    "vocab_size": 100,
    "hidden_size": 256,
    "intermediate_size": 1344,
    "num_layers": 4,
    "delta_num_qk_heads": 4,
    "delta_num_v_heads": 12,
    "delta_head_dim": 32,
    "attn_num_q_heads": 12,
    "attn_num_kv_heads": 2,
    "attn_head_dim": 64,
}


def test_scored_window_mpe_matches_smoke() -> None:
    assert (
        qwen38_hf_max_position_embeddings(8, twin_mode="scored_window")
        == SCORED_WINDOW_MIN_MPE
    )


def test_hf_scored_window_build() -> None:
    if importlib.util.find_spec("fla") or importlib.util.find_spec("causal_conv1d"):
        pytest.skip("fla/causal_conv1d would replace GDN with a fused kernel")
    family = Qwen38Family()
    model = family.build_hf_model(
        _WORKED_OPTS,
        seq_len=8,
        precision="fp32",
        device=__import__("torch").device("cpu"),
        twin_mode="scored_window",
    )
    cfg = model.config
    assert cfg.max_position_embeddings == 32
    hidden_act = getattr(cfg, "hidden_act", None) or getattr(cfg, "hidden_activation", None)
    assert hidden_act == "silu"
    assert cfg.tie_word_embeddings is False
    assert cfg.use_cache is False
    assert list(cfg.layer_types) == ["linear_attention"] * 3 + ["full_attention"]
    assert cfg.linear_num_key_heads == _WORKED_OPTS["delta_num_qk_heads"]
    assert cfg.linear_num_value_heads == _WORKED_OPTS["delta_num_v_heads"]
    assert type(model) is Qwen3_5ForCausalLM
    assert "ForConditionalGeneration" not in type(model).__name__


def test_hf_build_rejects_apertus_parity() -> None:
    family = Qwen38Family()
    with pytest.raises(ValueError, match="twin_mode"):
        family.build_hf_model(
            _WORKED_OPTS,
            seq_len=8,
            precision="fp32",
            device=__import__("torch").device("cpu"),
            twin_mode="apertus_parity",  # type: ignore[arg-type]
        )


def test_hf_build_rejects_mixed_precision() -> None:
    family = Qwen38Family()
    with pytest.raises(ValueError, match="unknown precision"):
        family.build_hf_model(
            _WORKED_OPTS,
            seq_len=8,
            precision="mixed",
            device=__import__("torch").device("cpu"),
        )


def test_get_model_family_qwen38() -> None:
    assert get_model_family("qwen38").model_id == "qwen38"
