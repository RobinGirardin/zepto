"""HF Apertus twin: llama3 RoPE and max_position_embeddings parity."""

from __future__ import annotations

import pytest

from zepto.empirical.models.apertus import (
    APERTUS_ORIGINAL_MPE,
    ApertusFamily,
    apertus_hf_max_position_embeddings,
)

pytest.importorskip("transformers")


def test_apertus_hf_max_position_embeddings_short_seq() -> None:
    assert apertus_hf_max_position_embeddings(8) == APERTUS_ORIGINAL_MPE + 1


def test_apertus_hf_max_position_embeddings_beyond_original() -> None:
    assert apertus_hf_max_position_embeddings(10_000) == 10_000


def test_hf_apertus_parity_config_llama3_no_mpe_warning() -> None:
    from transformers import ApertusConfig

    opts = {
        "vocab_size": 100,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 2,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 4,
    }
    family = ApertusFamily()
    model = family.build_hf_model(
        opts,
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
        vocab_size=opts["vocab_size"],
        hidden_size=opts["hidden_size"],
        intermediate_size=opts["intermediate_size"],
        num_hidden_layers=opts["num_layers"],
        num_attention_heads=opts["num_heads"],
        num_key_value_heads=opts["num_kv_heads"],
        max_position_embeddings=apertus_hf_max_position_embeddings(8),
    )
