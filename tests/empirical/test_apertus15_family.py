"""Apertus 1.5 empirical family: vocab split and registry isolation."""

from __future__ import annotations

import pytest

from zepto.compose import Tensor, compose_graph
from zepto.empirical.models import get_model_family
from zepto.empirical.models.apertus import ApertusFamily
from zepto.empirical.models.apertus15_core import Apertus15FamilyCore as Apertus15Family


def test_derive_defaults_output_vocab_to_vocab_size() -> None:
    family = Apertus15Family()
    opts = family.derive_options(
        {
            "head_dim": 4,
            "intermediate_size": 64,
            "num_heads": 8,
            "num_kv_heads": 2,
            "num_layers": 2,
            "vocab_size": 200,
        }
    )
    assert opts["hidden_size"] == 32
    assert opts["output_vocab_size"] == 200
    family.validate_options(opts)


def test_derive_passes_through_output_vocab_size() -> None:
    family = Apertus15Family()
    opts = family.derive_options(
        {
            "head_dim": 4,
            "intermediate_size": 64,
            "num_heads": 8,
            "num_kv_heads": 2,
            "num_layers": 2,
            "vocab_size": 200,
            "output_vocab_size": 100,
        }
    )
    assert opts["output_vocab_size"] == 100
    family.validate_options(opts)


def test_derive_free_knobs_default_output_vocab() -> None:
    family = Apertus15Family()
    opts = family.derive_options(
        {
            "gqa_group": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
            "ffn_mult": 2.0,
            "num_layers": 2,
            "vocab_size": 200,
        }
    )
    assert opts["num_heads"] == 8
    assert opts["hidden_size"] == 32
    assert opts["output_vocab_size"] == 200
    family.validate_options(opts)


def test_derive_free_knobs_sampled_output_vocab() -> None:
    family = Apertus15Family()
    opts = family.derive_options(
        {
            "gqa_group": 4,
            "num_kv_heads": 2,
            "head_dim": 4,
            "ffn_mult": 2.0,
            "num_layers": 2,
            "vocab_size": 200,
            "output_vocab_size": 100,
        }
    )
    assert opts["output_vocab_size"] == 100
    family.validate_options(opts)


def test_validate_rejects_head_wider_than_embed() -> None:
    family = Apertus15Family()
    with pytest.raises(ValueError, match="output_vocab_size"):
        family.validate_options(
            {
                "hidden_size": 32,
                "head_dim": 4,
                "intermediate_size": 64,
                "num_heads": 8,
                "num_kv_heads": 2,
                "num_layers": 2,
                "vocab_size": 200,
                "output_vocab_size": 300,
            }
        )


def test_validate_rejects_non_positive_output_vocab() -> None:
    family = Apertus15Family()
    with pytest.raises(ValueError, match="output_vocab_size"):
        family.validate_options(
            {
                "hidden_size": 32,
                "head_dim": 4,
                "intermediate_size": 64,
                "num_heads": 8,
                "num_kv_heads": 2,
                "num_layers": 2,
                "vocab_size": 200,
                "output_vocab_size": 0,
            }
        )


def test_get_model_family_apertus15() -> None:
    family = get_model_family("apertus15")
    assert family.model_id == "apertus15"
    assert "output_vocab_size" in family.option_keys()


def test_get_model_family_apertus_unchanged() -> None:
    family = get_model_family("apertus")
    assert family.model_id == "apertus"
    assert isinstance(family, ApertusFamily)
    assert "output_vocab_size" not in family.option_keys()


def test_infer_factory_includes_output_vocab_size() -> None:
    family = get_model_family("apertus15")
    opts = family.derive_options(
        {
            "head_dim": 4,
            "intermediate_size": 64,
            "num_heads": 8,
            "num_kv_heads": 2,
            "num_layers": 2,
            "vocab_size": 200,
            "output_vocab_size": 100,
        }
    )
    factory = family.build_zepto_infer_module_factory(opts, seq_len=8)
    captured: dict = {}

    def wrapping(_ctx):
        model = factory(_ctx)
        captured["output_vocab_size"] = model.output_vocab_size
        captured["vocab_size"] = model.vocab_size
        captured["lm_head_vocab"] = model.lm_head.vocab_size
        captured["embed_vocab"] = model.embedding.vocab_size
        return model

    compose_graph(wrapping, (Tensor(shape=(8,)),))
    assert captured["output_vocab_size"] == 100
    assert captured["vocab_size"] == 200
    assert captured["lm_head_vocab"] == 100
    assert captured["embed_vocab"] == 200
