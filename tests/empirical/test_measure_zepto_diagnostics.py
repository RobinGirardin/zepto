"""Zepto empirical measurements expose workspace breakdown (CPU)."""

from __future__ import annotations

from zepto.empirical.measure_zepto import measure_infer_zepto, measure_train_zepto
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


def test_infer_zepto_runtime_workspace_nonzero_with_cuda_capability() -> None:
    family = get_model_family("apertus")
    ctx = build_invocation_context("fp32", cuda_capability=(8, 0))
    opts = _golden_opts()
    result = measure_infer_zepto(
        family, opts, seq_len=8, batch_size=1, ctx=ctx
    )
    assert result.y_runtime_workspace > 0
    assert result.y_activations > 0
    assert result.y_flop_no_opt == result.y_flop
    assert result.y_flop_fcm == result.y_flop_fcm_no_opt
    assert result.y_flop_fcm < result.y_flop


def test_train_zepto_no_opt_flops_below_full_horizon() -> None:
    family = get_model_family("apertus")
    ctx = build_invocation_context("fp32", cuda_capability=(8, 0))
    opts = _golden_opts()
    result = measure_train_zepto(
        family, opts, seq_len=8, batch_size=1, ctx=ctx
    )
    assert result.y_runtime_workspace > 0
    assert result.y_flop_no_opt > 0
    assert result.y_flop_no_opt < result.y_flop
    assert result.y_flop_fcm == result.y_flop_fcm_no_opt
    assert result.y_flop_fcm < result.y_flop_no_opt
    # G=1 train: peak-relevant workspace is backward's 2 handles, not 1+2+0.
    assert result.y_runtime_workspace >= 2 * 8_519_680
    assert result.y_runtime_workspace < 3 * 8_519_680
