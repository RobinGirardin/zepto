"""Study ids match the architecture-only hash used by the harness."""

from __future__ import annotations

from qwen38_validity.schema import configuration_id, subject_id
from zepto.empirical.schema import configuration_id as harness_configuration_id
from zepto.empirical.schema import subject_id as harness_subject_id


def test_configuration_id_matches_harness() -> None:
    options = {
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
    assert configuration_id(options) == harness_configuration_id("qwen38", options)


def test_subject_id_matches_harness() -> None:
    cfg = configuration_id({"hidden_size": 256, "num_layers": 4})
    assert subject_id(cfg, seq_len=32, batch_size=1) == harness_subject_id(
        cfg, seq_len=32, batch_size=1
    )
