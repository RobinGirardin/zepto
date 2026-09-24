"""Study ids match the architecture-only hash used by the old harness."""

from __future__ import annotations

from apertus_validity.schema import configuration_id, subject_id
from zepto.empirical.schema import configuration_id as harness_configuration_id
from zepto.empirical.schema import subject_id as harness_subject_id


def test_configuration_id_matches_harness() -> None:
    options = {
        "hidden_size": 1024,
        "intermediate_size": 5376,
        "num_heads": 8,
        "num_kv_heads": 4,
        "num_layers": 4,
        "vocab_size": 1024,
        "head_dim": 128,
    }
    assert configuration_id(options) == harness_configuration_id("apertus", options)


def test_subject_id_matches_harness() -> None:
    cfg = configuration_id({"hidden_size": 1024, "num_layers": 4})
    assert subject_id(cfg, seq_len=32, batch_size=1) == harness_subject_id(
        cfg, seq_len=32, batch_size=1
    )
