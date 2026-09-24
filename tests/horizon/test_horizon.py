"""Phase 2 horizon simulation and accounting tests."""

from __future__ import annotations

import pytest

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    KVConfig,
    PrecisionPolicy,
    StepKind,
    account_flops,
    account_memory,
    estimate,
    estimate_horizon,
    inputs_from_shape,
    inputs_from_token_ids,
    inputs_from_token_ids_and_labels,
    reference_invocation,
    simulate_horizon,
)
from zepto.analysis.horizon.spec import HorizonStep
from zepto.analysis.horizon.state import (
    GradAccumState,
    parameter_bytes,
    trainable_gradient_bytes,
)
from zepto.compose import Tensor, compose_graph
from zepto.modules.layers.linear import Linear
from zepto.semantic.metadata import DType

_IN_FEATURES = 64
_OUT_FEATURES = 32


def _linear_module(_compose):
    return Linear(_IN_FEATURES, _OUT_FEATURES)


def _linear_inputs(step, _ctx, _state):
    return (Tensor(shape=(step.batch, step.seq_len, _IN_FEATURES)),)


def test_one_step_degenerates_to_single_invocation() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec(
        steps=[HorizonStep(kind=StepKind.GENERIC, seq_len=128, name="only")]
    )

    sim_report = estimate_horizon(spec, _linear_module, _linear_inputs, ctx)
    direct = estimate(
        compose_graph(
            lambda _ctx: Linear(_IN_FEATURES, _OUT_FEATURES),
            (Tensor(shape=(1, 128, _IN_FEATURES)),),
        ),
        ctx,
    )

    assert len(sim_report.per_step) == 1
    assert sim_report.total_flops == direct.flops.total_flops
    assert sim_report.peak_vram == direct.memory.peak_live_bytes


def test_prefill_decode_peak_includes_kv() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.inference(
        prefill=8192,
        decode_steps=3,
        kv=KVConfig(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            dtype=DType.FP16,
        ),
    )

    report = estimate_horizon(spec, _linear_module, _linear_inputs, ctx)

    assert report.state_final.kv_caches
    assert report.state_final.kv_caches[0].seq_len == 8195
    assert report.memory.breakdown.state > 0
    assert report.peak_vram > report.per_step[0].memory.peak_live_bytes


def test_horizon_sum_all_deduplicates_parameters() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.repeat(3, seq_len=128)

    report = estimate_horizon(spec, _linear_module, _linear_inputs, ctx)
    param_bytes = report.per_step[0].memory.breakdown.parameters
    naive_sum = sum(step.memory.sum_all_bytes for step in report.per_step)

    assert report.memory.sum_all_bytes == naive_sum - param_bytes * (len(report.per_step) - 1)
    assert report.peak_vram == max(
        step.memory.peak_live_bytes for step in report.per_step
    )


def test_grad_accum_carries_buffers() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec().grad_accum(4, 2048)

    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    assert sim.state_final.grad_accum is not None
    assert sim.state_final.grad_accum.micro_batches_seen == 4
    assert sim.state_final.grad_accum.parameter_bytes > 0

    report = account_memory(sim)
    assert isinstance(report.per_step, tuple)
    assert len(report.per_step) == 4
    assert report.breakdown.state >= 0


def test_optimizer_state_persisted() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=1024, micro_batches=2, optimizer=AdamW)

    report = estimate_horizon(spec, _linear_module, _linear_inputs, ctx)

    assert len(report.per_step) == 3
    assert spec.steps[0].kind == StepKind.TRAIN
    assert spec.steps[1].kind == StepKind.TRAIN
    assert spec.steps[-1].kind == StepKind.OPTIMIZER
    assert report.state_final.optimizer is not None
    assert report.state_final.optimizer.bytes > 0
    assert report.state_final.grad_accum is None

    last_graph = report.per_step[-2]
    assert last_graph.memory.breakdown.saved_for_backward > 0
    assert last_graph.memory.breakdown.weight_grads > 0
    assert last_graph.memory.breakdown.state > 0


def test_simulate_horizon_advances_kv_and_grad_state() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec(
        kv=KVConfig(
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
            dtype=DType.FP32,
        ),
    )
    spec.prefill(128).decode(2).grad_accum(2, 64).backward_step(64)

    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    assert len(sim.timeline) == 6
    assert sim.state_final.kv_caches[0].seq_len == 130

    hflops = account_flops(sim)
    assert hflops.total_flops > 0
    assert len(hflops.per_step) == 6


def test_training_horizon_backward_step_includes_double_cublas() -> None:
    ctx = reference_invocation(
        hardware="cuda",
        compute_capability=(8, 0),
    )
    spec = HorizonSpec.training(seq_len=128, micro_batches=1, optimizer=AdamW)
    report = estimate_horizon(spec, _linear_module, _linear_inputs, ctx)
    assert len(report.per_step) == 2
    train_step = report.per_step[0]
    assert train_step.memory.breakdown.runtime_workspace == 2 * 8_519_680
    optimizer_mem = report.per_step[1]
    assert optimizer_mem.memory.breakdown.runtime_workspace == 0
    assert report.memory.breakdown.runtime_workspace == 2 * 8_519_680


def test_inputs_from_shape_helper() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.repeat(1, seq_len=128)

    report = estimate_horizon(
        spec,
        _linear_module,
        inputs_from_shape(_IN_FEATURES),
        ctx,
    )
    assert report.total_flops > 0


def test_inputs_from_token_ids_is_rank_two() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.repeat(1, seq_len=32, batch=1)
    inputs_fn = inputs_from_token_ids()
    tokens = inputs_fn(spec.steps[0], ctx, None)  # type: ignore[arg-type]
    assert tokens[0].shape == (1, 32)
    assert tokens[0].semantic_type == "token_ids"


def test_inputs_from_token_ids_and_labels_is_rank_two() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.repeat(1, seq_len=32, batch=4)
    inputs_fn = inputs_from_token_ids_and_labels()
    tokens, labels = inputs_fn(spec.steps[0], ctx, None)  # type: ignore[arg-type]
    assert tokens.shape == (4, 32)
    assert labels.shape == (4, 32)
    assert tokens.semantic_type == "token_ids"
    assert labels.semantic_type == "labels"


def test_inputs_fn_batch_mismatch_raises() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=128, batch=4, micro_batches=1)

    def _wrong_batch(_step, _ctx, _state):
        return (Tensor(shape=(1, 128, _IN_FEATURES)),)

    with pytest.raises(ValueError, match="does not match"):
        simulate_horizon(spec, _linear_module, _wrong_batch, ctx)


def test_horizon_step_rejects_batch_below_one() -> None:
    with pytest.raises(ValueError, match="batch must be >= 1"):
        HorizonStep(kind=StepKind.GENERIC, seq_len=8, batch=0)


def test_training_horizon_batch_propagates_to_compose() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=128, batch=4, micro_batches=1)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)

    composed = [
        record for record in sim.timeline if record.step.kind != StepKind.OPTIMIZER
    ]
    assert composed
    for record in composed:
        root = record.graph.edge(record.graph.inputs[0]).tensor
        assert root.shape[:2] == (4, 128)


def test_prefill_decode_kv_bytes_scale_with_batch() -> None:
    ctx = reference_invocation()
    kv = KVConfig(
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        dtype=DType.FP16,
    )
    spec_1 = HorizonSpec.inference(prefill=64, decode_steps=2, batch=1, kv=kv)
    spec_b = HorizonSpec.inference(prefill=64, decode_steps=2, batch=4, kv=kv)

    report_1 = estimate_horizon(spec_1, _linear_module, _linear_inputs, ctx)
    report_b = estimate_horizon(spec_b, _linear_module, _linear_inputs, ctx)

    kv_1 = report_1.state_final.kv_caches[0]
    kv_b = report_b.state_final.kv_caches[0]
    assert kv_b.batch == 4
    assert kv_1.batch == 1
    assert kv_b.seq_len == kv_1.seq_len == 66
    assert kv_b.bytes == 4 * kv_1.bytes


def test_grad_accum_independent_of_parallel_batch() -> None:
    ctx = reference_invocation()
    spec_g = HorizonSpec().grad_accum(4, 128, batch=1)
    spec_b = HorizonSpec().grad_accum(4, 128, batch=4)

    sim_g = simulate_horizon(spec_g, _linear_module, _linear_inputs, ctx)
    sim_b = simulate_horizon(spec_b, _linear_module, _linear_inputs, ctx)

    assert sim_g.state_final.grad_accum is not None
    assert sim_b.state_final.grad_accum is not None
    assert sim_g.state_final.optimizer is None
    assert sim_b.state_final.optimizer is None
    assert (
        sim_g.state_final.grad_accum.bytes == sim_b.state_final.grad_accum.bytes
    )
    assert sim_g.state_final.grad_accum.parameter_bytes == (
        sim_b.state_final.grad_accum.parameter_bytes
    )

    mem_g = account_memory(sim_g)
    mem_b = account_memory(sim_b)
    assert mem_b.peak_live_bytes > mem_g.peak_live_bytes


def test_training_step_appends_optimizer_row() -> None:
    spec = HorizonSpec().training_step(8, optimizer=AdamW)
    assert len(spec.steps) == 2
    assert spec.steps[0].kind == StepKind.TRAIN
    assert spec.steps[0].phase == "full"
    assert spec.steps[0].name == "train"
    assert spec.steps[-1].kind == StepKind.OPTIMIZER
    assert spec.optimizer_policy is AdamW


def test_training_step_g2_appends_two_train_rows() -> None:
    spec = HorizonSpec().training_step(8, micro_batches=2, optimizer=AdamW)
    assert len(spec.steps) == 3
    assert spec.steps[0].kind == StepKind.TRAIN
    assert spec.steps[1].kind == StepKind.TRAIN
    assert spec.steps[0].name == "train_0"
    assert spec.steps[1].name == "train_1"
    assert spec.steps[0].phase == spec.steps[1].phase == "full"
    assert spec.steps[-1].kind == StepKind.OPTIMIZER
    assert spec.optimizer_policy is AdamW


def test_training_g1_has_no_grad_accum() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=128, batch=4, micro_batches=1, optimizer=AdamW)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)

    assert sim.state_final.grad_accum is None
    assert sim.timeline[0].state_after.grad_accum is None
    assert sim.state_initial.grad_accum is None
    for record in sim.timeline:
        assert record.state_after.grad_accum is None


def test_training_g1_peak_not_inflated_by_param_accum() -> None:
    from dataclasses import replace

    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=128, batch=4, micro_batches=1, optimizer=AdamW)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    honest = account_memory(sim)

    param_b = parameter_bytes(sim.timeline[0].lowered)
    fake_accum = GradAccumState(parameter_bytes=param_b, micro_batches_seen=1)

    def _with_accum(record):
        return replace(
            record,
            state_after=replace(record.state_after, grad_accum=fake_accum),
        )

    # Inject on every snapshot so carry_state lifts every step peak. Linear
    # activations can dominate, so a buffer that only appears after micro
    # would not move the global max.
    inflated_sim = replace(
        sim,
        state_initial=replace(sim.state_initial, grad_accum=fake_accum),
        timeline=tuple(_with_accum(record) for record in sim.timeline),
        state_final=replace(sim.state_final, grad_accum=fake_accum),
    )
    inflated = account_memory(inflated_sim)

    assert honest.peak_live_bytes < inflated.peak_live_bytes
    assert inflated.peak_live_bytes - honest.peak_live_bytes == param_b


def test_g2_accum_bytes_use_grad_dtype() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec().grad_accum(2, 128)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    lowered = sim.timeline[-1].lowered

    assert sim.state_final.grad_accum is not None
    assert sim.state_final.grad_accum.micro_batches_seen == 2
    assert sim.state_final.grad_accum.bytes == trainable_gradient_bytes(lowered)
    assert trainable_gradient_bytes(lowered) == parameter_bytes(lowered)


def test_trainable_gradient_bytes_uses_grad_dtype() -> None:
    ctx = reference_invocation(
        precision=PrecisionPolicy.from_byte_sizes(param_bytes=2, grad_bytes=4),
    )
    spec = HorizonSpec().grad_accum(2, 128)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    lowered = sim.timeline[-1].lowered
    grad_bytes = trainable_gradient_bytes(lowered)
    param_b = parameter_bytes(lowered)

    assert grad_bytes == 2 * param_b
    assert sim.state_final.grad_accum is not None
    assert sim.state_final.grad_accum.bytes == grad_bytes
    assert sim.state_final.grad_accum.micro_batches_seen == 2


def test_steady_state_seeds_optimizer_before_train_step() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)

    warmed = simulate_horizon(
        spec, _linear_module, _linear_inputs, ctx, steady_state=True
    )
    cold = simulate_horizon(
        spec, _linear_module, _linear_inputs, ctx, steady_state=False
    )

    assert warmed.state_initial.optimizer is not None
    assert warmed.state_initial.optimizer.bytes > 0
    assert warmed.timeline[0].step.kind == StepKind.TRAIN

    warm_mem = account_memory(warmed)
    params = parameter_bytes(warmed.timeline[0].lowered)
    opt = warmed.state_initial.optimizer.bytes
    assert warm_mem.peak_live_bytes >= params + opt

    assert cold.state_initial.optimizer is None
    assert cold.timeline[0].state_after.optimizer is not None
    assert cold.timeline[0].state_after.optimizer.bytes > 0
