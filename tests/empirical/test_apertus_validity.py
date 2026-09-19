"""Apertus family option validation."""

from __future__ import annotations

import pytest

from zepto.empirical.models.apertus_core import ApertusFamilyCore as ApertusFamily


def test_derive_hidden_size() -> None:
    family = ApertusFamily()
    opts = family.derive_options(
        {
            "head_dim": 4,
            "intermediate_size": 64,
            "num_heads": 8,
            "num_kv_heads": 2,
            "num_layers": 2,
            "vocab_size": 100,
        }
    )
    assert opts["hidden_size"] == 32
    family.validate_options(opts)


def test_validate_head_dim_even() -> None:
    family = ApertusFamily()
    with pytest.raises(ValueError, match="even"):
        family.validate_options(
            {
                "hidden_size": 35,
                "head_dim": 5,
                "intermediate_size": 64,
                "num_heads": 7,
                "num_kv_heads": 1,
                "num_layers": 2,
                "vocab_size": 100,
            }
        )


def test_validate_gqa_ratio() -> None:
    family = ApertusFamily()
    with pytest.raises(ValueError, match="divisible"):
        family.validate_options(
            {
                "hidden_size": 28,
                "head_dim": 4,
                "intermediate_size": 64,
                "num_heads": 7,
                "num_kv_heads": 4,
                "num_layers": 2,
                "vocab_size": 100,
            }
        )
