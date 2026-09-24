"""Analytical replay of validity subject ab0f75a0da537ed6 (no GPU)."""

from __future__ import annotations

from zepto.analysis import AdamW, HorizonSpec, estimate_horizon, inputs_from_token_ids_and_labels
from zepto.empirical.parity_ctx import build_invocation_context
from zepto.modules import ApertusForCausalLM

# studies/apertus-validity/artifacts/main/subjects.csv + evaluation.csv
_SUBJECT = dict(
    hidden_size=2048,
    intermediate_size=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=4,
    vocab_size=1024,
    seq_len=32,
    head_dim=64,
)
_BATCH = 2
_TWIN_PEAK = 3_624_819_200
_OLD_ZEPTO_PEAK = 2_164_488_288


def _module(_ctx):
    return ApertusForCausalLM(**_SUBJECT)


def test_validity_subject_ab0f75a0_training_peak_within_delta() -> None:
    ctx = build_invocation_context("fp32", cuda_capability=(8, 0))
    spec = HorizonSpec.training(
        seq_len=_SUBJECT["seq_len"],
        batch=_BATCH,
        micro_batches=1,
        optimizer=AdamW,
    )
    report, sim = estimate_horizon(
        spec,
        _module,
        inputs_from_token_ids_and_labels(),
        ctx,
        return_simulation=True,
    )

    peak = report.peak_vram
    # Foreach workspace raises the param-dominated floor to ~5P. Residual
    # vs the eager HF twin is now a few tens of MiB (cuBLAS / tape), not
    # a missing P.
    assert abs(peak - _TWIN_PEAK) < abs(peak - _OLD_ZEPTO_PEAK)
    assert abs(peak - _TWIN_PEAK) / _TWIN_PEAK < 0.03

    train = report.per_step[0]
    assert len(report.per_step) == 2
    assert train.memory.breakdown.weight_grads > 0
    assert train.memory.breakdown.saved_for_backward > 0
    assert sim.state_initial.optimizer is not None
    assert sim.state_final.grad_accum is None
