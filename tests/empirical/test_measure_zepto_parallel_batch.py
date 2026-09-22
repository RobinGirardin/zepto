"""Train/infer GOLDEN measurements use native parallel (B, S)."""

from __future__ import annotations

import pytest

from zepto.empirical import HARNESS_VERSION
from zepto.empirical.measure_zepto import measure_train_zepto
from zepto.empirical.models import get_model_family
from zepto.empirical.parity_ctx import build_invocation_context
from tests.integration.apertus.shared import GOLDEN


def _golden_opts() -> dict[str, int]:
    family = get_model_family("apertus")
    return family.derive_options(
        {
            "head_dim": GOLDEN["head_dim"],
            "intermediate_size": GOLDEN["intermediate_size"],
            "num_heads": GOLDEN["num_heads"],
            "num_kv_heads": GOLDEN["num_kv_heads"],
            "num_layers": GOLDEN["num_layers"],
            "vocab_size": GOLDEN["vocab_size"],
        }
    )


def test_harness_version_parallel_batch() -> None:
    assert HARNESS_VERSION == "0.3.0"


def test_train_batch_two_grows_activations() -> None:
    family = get_model_family("apertus")
    ctx = build_invocation_context("fp32", cuda_capability=(8, 0))
    opts = _golden_opts()
    r1 = measure_train_zepto(family, opts, seq_len=8, batch_size=1, ctx=ctx)
    r2 = measure_train_zepto(family, opts, seq_len=8, batch_size=2, ctx=ctx)
    assert r1.zepto_batch_representation == "parallel_batch"
    assert r2.zepto_batch_representation == "parallel_batch"
    assert r2.y_activations / r1.y_activations == pytest.approx(2.0, rel=0.10)
    assert r2.y_vram > r1.y_vram
