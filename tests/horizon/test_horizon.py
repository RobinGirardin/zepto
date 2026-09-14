"""Phase 2 horizon simulation and accounting tests."""

from __future__ import annotations

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    StatePortRegistry,
    account_flops,
    account_memory,
    estimate,
    estimate_horizon,
    reference_invocation,
    simulate_horizon,
)
from zepto.analysis.horizon.spec import HorizonStep
from zepto.compose import Tensor, compose_graph
from zepto.modules.linear import Linear
from zepto.semantic.metadata import DType


def _linear_compose(step, _state):
    in_features = 64
    out_features = 32
    batch = step.batch
    seq = step.seq_len
    return (
        lambda _ctx: Linear(in_features, out_features),
        (Tensor(shape=(batch, seq, in_features)),),
    )


def test_one_step_degenerates_to_single_invocation() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec(steps=[HorizonStep("only", 128)])

    sim_report = estimate_horizon(_linear_compose, ctx, spec)
    direct = estimate(
        compose_graph(
            lambda _ctx: Linear(64, 32),
            (Tensor(shape=(1, 128, 64)),),
        ),
        ctx,
    )

    assert len(sim_report.per_step) == 1
    assert sim_report.total_flops == direct.flops.total_flops
    assert sim_report.peak_vram == direct.memory.peak_live_bytes


def test_prefill_decode_peak_includes_kv() -> None:
    ctx = reference_invocation()
    state = StatePortRegistry.empty().configure_kv(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        dtype=DType.FP16,
    )
    spec = HorizonSpec().prefill(8192).decode(3)

    report = estimate_horizon(
        _linear_compose,
        ctx,
        spec,
        state=state,
    )

    assert report.state_final.kv_caches
    assert report.state_final.kv_caches[0].seq_len == 8195
    assert report.memory.breakdown.state > 0
    assert report.peak_vram > report.per_step[0].memory.peak_live_bytes


def test_horizon_sum_all_deduplicates_parameters() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec(
        steps=[HorizonStep(f"step_{index}", 128) for index in range(3)]
    )

    report = estimate_horizon(_linear_compose, ctx, spec)
    param_bytes = report.per_step[0].memory.breakdown.parameters
    naive_sum = sum(step.memory.sum_all_bytes for step in report.per_step)

    assert report.memory.sum_all_bytes == naive_sum - param_bytes * (len(report.per_step) - 1)
    assert report.peak_vram == max(
        step.memory.peak_live_bytes for step in report.per_step
    )


def test_grad_accum_carries_buffers() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec().grad_accum(4, 2048)

    sim = simulate_horizon(_linear_compose, spec, ctx)
    assert sim.state_final.grad_accum is not None
    assert sim.state_final.grad_accum.micro_batches_seen == 4
    assert sim.state_final.grad_accum.parameter_bytes > 0

    report = account_memory(sim)
    assert isinstance(report.per_step, tuple)
    assert len(report.per_step) == 4
    assert report.breakdown.state >= 0


def test_optimizer_state_persisted() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec().training_step(1024, micro_batches=2, optimizer=AdamW)

    report = estimate_horizon(_linear_compose, ctx, spec)

    assert report.state_final.optimizer is not None
    assert report.state_final.optimizer.bytes > 0
    assert report.state_final.grad_accum is None

    final_step = report.per_step[-1]
    assert final_step.memory.breakdown.state > 0


def test_simulate_horizon_advances_kv_and_grad_state() -> None:
    ctx = reference_invocation()
    kv_state = StatePortRegistry.empty().configure_kv(
        num_layers=1,
        num_kv_heads=1,
        head_dim=16,
        dtype=DType.FP32,
    )
    spec = (
        HorizonSpec()
        .prefill(128)
        .decode(2)
        .grad_accum(2, 64)
        .backward_step(64)
    )

    sim = simulate_horizon(_linear_compose, spec, ctx, state=kv_state)
    assert len(sim.timeline) == 6
    assert sim.state_final.kv_caches[0].seq_len == 130

    hflops = account_flops(sim)
    assert hflops.total_flops > 0
    assert len(hflops.per_step) == 6
