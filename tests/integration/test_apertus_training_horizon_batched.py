"""Batched Apertus training-horizon estimate_horizon API."""

from __future__ import annotations

import pytest

from zepto.analysis import (
    AdamW,
    HorizonCostReport,
    HorizonSpec,
    estimate_horizon,
    inputs_from_token_ids,
    reference_invocation,
)
from zepto.modules.models.apertus import Apertus
from zepto.semantic.metadata import DType

_SEQ_LEN = 128
_BATCH = 2


def _tiny_apertus(_compose):
    return Apertus(
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        vocab_size=1024,
        seq_len=_SEQ_LEN,
    )


def _flash_ctx():
    return reference_invocation(
        requested_capabilities=frozenset({"fused", "flash"}),
        default_dtype=DType.FP16,
    )


def _training_report(batch: int) -> HorizonCostReport:
    spec = HorizonSpec.training(
        seq_len=_SEQ_LEN,
        batch=batch,
        micro_batches=1,
        optimizer=AdamW,
    )
    return estimate_horizon(spec, _tiny_apertus, inputs_from_token_ids(), _flash_ctx())


def test_apertus_training_horizon_batched_estimate_api() -> None:
    report_1 = _training_report(1)
    report_b = _training_report(_BATCH)

    assert len(report_1.per_step) == 3
    assert len(report_b.per_step) == 3
    assert report_1.state_final.optimizer is not None
    assert report_b.state_final.optimizer is not None

    params_1 = report_1.per_step[0].memory.breakdown.parameters
    params_b = report_b.per_step[0].memory.breakdown.parameters
    assert params_1 == params_b
    assert report_1.state_final.optimizer.bytes == report_b.state_final.optimizer.bytes

    # MICRO_FORWARD activations, not raw peak_vram: params + AdamW (8 B/param)
    # are constant in B and dominate total horizon peak on tiny Apertus.
    # Shared causal mask is (1, S, S). See ADR-0010.
    act_1 = report_1.per_step[0].memory.breakdown.activations
    act_b = report_b.per_step[0].memory.breakdown.activations
    assert act_1 > 0
    assert act_b / act_1 == pytest.approx(_BATCH, rel=0.05)
    assert report_b.peak_vram > report_1.peak_vram
