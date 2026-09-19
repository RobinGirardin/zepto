"""Zepto inference measurement (CPU): micro_sum batch horizon path."""

from __future__ import annotations

from zepto.empirical.measure_zepto import measure_infer_zepto
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


def test_infer_batch_two_uses_horizon_inputs_fn() -> None:
    """B>1 must call zepto_infer_inputs(step, ctx, state), not (step) only."""
    family = get_model_family("apertus")
    ctx = build_invocation_context("fp32", cuda_capability=(8, 0))
    opts = _golden_opts()
    b1 = measure_infer_zepto(family, opts, seq_len=8, batch_size=1, ctx=ctx)
    b2 = measure_infer_zepto(family, opts, seq_len=8, batch_size=2, ctx=ctx)
    assert b1[2] == b2[2] == "micro_sum"
    assert b2[0] == b1[0] * 2
    assert b1[0] > 0 and b1[1] > 0
