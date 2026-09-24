"""G>1 training peak: TRAIN × G on (b, S), then optimizer."""

from __future__ import annotations

import pytest

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    StepKind,
    estimate_horizon,
    reference_invocation,
    simulate_horizon,
)
from zepto.analysis.horizon.state import trainable_gradient_bytes
from tests.horizon.test_horizon import (
    _linear_inputs,
    _linear_module,
)

_CUBLAS_HANDLE = 8_519_680


def test_training_g2_has_no_grad_accum() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=128, batch=4, micro_batches=2, optimizer=AdamW)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)

    assert sim.state_initial.grad_accum is None
    assert sim.state_final.grad_accum is None
    for record in sim.timeline:
        assert record.state_after.grad_accum is None


def test_training_g2_each_micro_has_tape_and_persist_grads() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=128, batch=4, micro_batches=2, optimizer=AdamW)
    report, sim = estimate_horizon(
        spec, _linear_module, _linear_inputs, ctx, return_simulation=True
    )

    train_reports = [
        (record, step)
        for record, step in zip(sim.timeline, report.per_step, strict=True)
        if record.step.kind == StepKind.TRAIN
    ]
    assert len(train_reports) == 2
    for record, step in train_reports:
        grad_bytes = trainable_gradient_bytes(record.lowered)
        assert step.memory.breakdown.saved_for_backward > 0
        assert step.memory.breakdown.weight_grads == grad_bytes
        assert grad_bytes > 0

    optimizer = report.per_step[-1]
    assert optimizer.memory.breakdown.saved_for_backward == 0
    assert optimizer.memory.breakdown.weight_grads == 0


def test_training_g2_peak_matches_g1_at_same_b() -> None:
    ctx = reference_invocation()
    batch = 4
    seq_len = 128
    spec_g1 = HorizonSpec.training(
        seq_len=seq_len, batch=batch, micro_batches=1, optimizer=AdamW
    )
    spec_g2 = HorizonSpec.training(
        seq_len=seq_len, batch=batch, micro_batches=2, optimizer=AdamW
    )
    report_g1 = estimate_horizon(spec_g1, _linear_module, _linear_inputs, ctx)
    report_g2 = estimate_horizon(spec_g2, _linear_module, _linear_inputs, ctx)

    assert report_g2.peak_vram == report_g1.peak_vram
    assert (
        report_g2.per_step[0].memory.breakdown.activations
        == report_g1.per_step[0].memory.breakdown.activations
    )


def test_training_g2_peak_lt_g1_at_batch_g_times_b() -> None:
    ctx = reference_invocation()
    spec_g2 = HorizonSpec.training(
        seq_len=2048, batch=4, micro_batches=2, optimizer=AdamW
    )
    spec_wide = HorizonSpec.training(
        seq_len=2048, batch=8, micro_batches=1, optimizer=AdamW
    )
    report_g2 = estimate_horizon(spec_g2, _linear_module, _linear_inputs, ctx)
    report_wide = estimate_horizon(spec_wide, _linear_module, _linear_inputs, ctx)

    assert report_g2.peak_vram < report_wide.peak_vram


def test_training_g2_flops_are_g_full_graphs() -> None:
    ctx = reference_invocation()
    spec_g1 = HorizonSpec.training(seq_len=128, batch=4, micro_batches=1, optimizer=AdamW)
    spec_g2 = HorizonSpec.training(seq_len=128, batch=4, micro_batches=2, optimizer=AdamW)
    report_g1 = estimate_horizon(spec_g1, _linear_module, _linear_inputs, ctx)
    report_g2 = estimate_horizon(spec_g2, _linear_module, _linear_inputs, ctx)

    train_g1 = report_g1.per_step[0].flops
    adam = report_g1.per_step[1].flops.total_flops
    assert report_g2.total_flops == 2 * train_g1.total_flops + adam

    split = 2 * train_g1.forward_flops + train_g1.backward_flops
    assert report_g2.total_flops > split
    assert report_g2.total_flops != 3 * train_g1.forward_flops


def test_training_g2_cublas_merged_is_two_handles() -> None:
    ctx = reference_invocation(
        hardware="cuda",
        compute_capability=(8, 0),
    )
    spec = HorizonSpec.training(seq_len=128, micro_batches=2, optimizer=AdamW)
    report = estimate_horizon(spec, _linear_module, _linear_inputs, ctx)

    assert len(report.per_step) == 3
    assert report.per_step[0].memory.breakdown.runtime_workspace == 2 * _CUBLAS_HANDLE
    assert report.per_step[1].memory.breakdown.runtime_workspace == 2 * _CUBLAS_HANDLE
    assert report.per_step[2].memory.breakdown.runtime_workspace == 0
    assert report.memory.breakdown.runtime_workspace == 2 * _CUBLAS_HANDLE


def test_training_rejects_micro_batches_below_one() -> None:
    with pytest.raises(ValueError, match="micro_batches must be >= 1"):
        HorizonSpec.training(seq_len=8, micro_batches=0, optimizer=AdamW)
