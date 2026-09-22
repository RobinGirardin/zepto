"""HF Apertus twin: llama3 RoPE and max_position_embeddings parity."""

from __future__ import annotations

import warnings
from dataclasses import fields

import pytest

from zepto.empirical.models.apertus import (
    APERTUS_ORIGINAL_MPE,
    SCORED_WINDOW_MIN_MPE,
    ApertusFamily,
    apertus_hf_max_position_embeddings,
)
from zepto.empirical.runner import RunConfig

pytest.importorskip("transformers")

_GOLDEN_OPTS = {
    "vocab_size": 100,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_layers": 2,
    "num_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 4,
}


def test_run_config_default_twin_mode_is_scored_window() -> None:
    field = next(f for f in fields(RunConfig) if f.name == "twin_mode")
    assert field.default == "scored_window"


def test_apertus_hf_max_position_embeddings_short_seq() -> None:
    assert (
        apertus_hf_max_position_embeddings(8, twin_mode="apertus_parity")
        == APERTUS_ORIGINAL_MPE + 1
    )


def test_apertus_hf_max_position_embeddings_beyond_original() -> None:
    assert apertus_hf_max_position_embeddings(10_000, twin_mode="apertus_parity") == 10_000


def test_scored_window_mpe_matches_smoke() -> None:
    assert (
        apertus_hf_max_position_embeddings(8, twin_mode="scored_window")
        == SCORED_WINDOW_MIN_MPE
    )


def test_hf_scored_window_build() -> None:
    family = ApertusFamily()
    # Transformers warns when original_mpe (8192) is not < MPE=32; smoke accepts it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = family.build_hf_model(
            _GOLDEN_OPTS,
            seq_len=8,
            precision="fp32",
            device=__import__("torch").device("cpu"),
            twin_mode="scored_window",
        )
    cfg = model.config
    assert cfg.max_position_embeddings == 32
    assert cfg.rope_parameters["rope_type"] == "llama3"
    assert cfg.rope_parameters["original_max_position_embeddings"] == 8192


def test_hf_apertus_parity_config_llama3_no_mpe_warning() -> None:
    from transformers import ApertusConfig

    family = ApertusFamily()
    model = family.build_hf_model(
        _GOLDEN_OPTS,
        seq_len=8,
        precision="fp32",
        device=__import__("torch").device("cpu"),
        twin_mode="apertus_parity",
    )
    cfg = model.config
    assert cfg.max_position_embeddings == 8193
    assert cfg.rope_parameters["rope_type"] == "llama3"
    assert cfg.rope_parameters["original_max_position_embeddings"] == 8192
    # Constructing config directly should not warn when MPE > original MPE.
    ApertusConfig(
        vocab_size=_GOLDEN_OPTS["vocab_size"],
        hidden_size=_GOLDEN_OPTS["hidden_size"],
        intermediate_size=_GOLDEN_OPTS["intermediate_size"],
        num_hidden_layers=_GOLDEN_OPTS["num_layers"],
        num_attention_heads=_GOLDEN_OPTS["num_heads"],
        num_key_value_heads=_GOLDEN_OPTS["num_kv_heads"],
        max_position_embeddings=apertus_hf_max_position_embeddings(
            8, twin_mode="apertus_parity"
        ),
    )


def test_hf_build_rejects_mixed_precision() -> None:
    family = ApertusFamily()
    with pytest.raises(ValueError, match="unknown precision"):
        family.build_hf_model(
            _GOLDEN_OPTS,
            seq_len=8,
            precision="mixed",
            device=__import__("torch").device("cpu"),
        )
