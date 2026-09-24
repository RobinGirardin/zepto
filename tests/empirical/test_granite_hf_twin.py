"""HF Granite twin: scored_window MPE and identity-scale contract."""

from __future__ import annotations

import pytest

from zepto.empirical.models import get_model_family
from zepto.empirical.models.granite import (
    SCORED_WINDOW_MIN_MPE,
    GraniteFamily,
    granite_hf_max_position_embeddings,
)

pytest.importorskip("transformers")

try:
    from transformers import GraniteConfig
except ImportError:  # pragma: no cover
    GraniteConfig = None

if GraniteConfig is None:
    pytest.skip("transformers GraniteConfig is unavailable", allow_module_level=True)

_GOLDEN_OPTS = {
    "vocab_size": 100,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_layers": 2,
    "num_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 4,
}


def test_scored_window_mpe_matches_smoke() -> None:
    assert (
        granite_hf_max_position_embeddings(8, twin_mode="scored_window")
        == SCORED_WINDOW_MIN_MPE
    )


def test_hf_scored_window_build() -> None:
    family = GraniteFamily()
    model = family.build_hf_model(
        _GOLDEN_OPTS,
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


def test_hf_build_rejects_apertus_parity() -> None:
    family = GraniteFamily()
    with pytest.raises(ValueError, match="twin_mode"):
        family.build_hf_model(
            _GOLDEN_OPTS,
            seq_len=8,
            precision="fp32",
            device=__import__("torch").device("cpu"),
            twin_mode="apertus_parity",  # type: ignore[arg-type]
        )


def test_hf_build_rejects_mixed_precision() -> None:
    family = GraniteFamily()
    with pytest.raises(ValueError, match="unknown precision"):
        family.build_hf_model(
            _GOLDEN_OPTS,
            seq_len=8,
            precision="mixed",
            device=__import__("torch").device("cpu"),
        )


def test_get_model_family_granite() -> None:
    assert get_model_family("granite").model_id == "granite"
