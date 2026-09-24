"""Optimizer boundary policy tests (out-of-graph training boundary)."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis import AdamW, HorizonSpec, StepKind, estimate_horizon, reference_invocation
from zepto.analysis.horizon.state import trainable_parameter_elements
from zepto.compose import Tensor, compose_graph
from zepto.modules import ApertusForCausalLM
from zepto.modules.layers.linear import Linear
from zepto.semantic.metadata import DType

_IN_FEATURES = 64
_OUT_FEATURES = 32

GOLDEN = dict(
    hidden_size=32,
    intermediate_size=64,
    num_heads=8,
    num_kv_heads=2,
    num_layers=2,
    vocab_size=100,
    seq_len=8,
    head_dim=4,
)


def _linear_module(_compose):
    return Linear(_IN_FEATURES, _OUT_FEATURES)


def _linear_inputs(step, _ctx, _state):
    return (Tensor(shape=(step.batch, step.seq_len, _IN_FEATURES)),)


def _training_module(_ctx):
    return ApertusForCausalLM(**GOLDEN)


def _training_inputs(step, _ctx, _state):
    return (
        Tensor(shape=(step.seq_len,)),
        Tensor(shape=(step.seq_len,), semantic_type="labels", requires_grad=False),
    )


def _golden_ctx():
    return reference_invocation(
        phase="forward",
        default_dtype=DType.FP32,
        optim_prec=4,
        attention_backend="eager",
        hardware="cuda",
        compute_capability=(8, 0),
        requested_capabilities=frozenset({"fused", "sdpa", "gqa"}),
    )


def test_training_spec_includes_optimizer_step() -> None:
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)
    assert len(spec.steps) == 2
    assert spec.steps[0].kind == StepKind.TRAIN
    assert spec.steps[0].phase == "full"
    assert spec.steps[-1].kind == StepKind.OPTIMIZER


def test_adamw_state_bytes_uses_optim_prec() -> None:
    ctx = reference_invocation(optim_prec=4)
    assert AdamW.state_bytes(trainable_elements=1000, context=ctx) == 8000


def test_horizon_flops_excludes_phantom_forward() -> None:
    ctx = _golden_ctx()
    spec = HorizonSpec.training(seq_len=GOLDEN["seq_len"], micro_batches=1, optimizer=AdamW)
    report = estimate_horizon(spec, _training_module, _training_inputs, ctx)

    full_flops = report.per_step[0].flops.total_flops
    update_reported = report.per_step[1].flops.total_flops
    train_graph = compose_graph(
        _training_module,
        (
            Tensor(shape=(GOLDEN["seq_len"],)),
            Tensor(shape=(GOLDEN["seq_len"],), semantic_type="labels", requires_grad=False),
        ),
    )
    from zepto.analysis import lower

    train_lowered = lower(train_graph, replace(ctx, phase="forward"))
    trainable = trainable_parameter_elements(train_lowered)
    update_flops = AdamW.update_flops(trainable_elements=trainable, context=ctx)

    expected = full_flops + update_flops
    assert update_reported == update_flops
    assert report.total_flops == expected


def test_horizon_flops_no_duplicate_forward() -> None:
    ctx = _golden_ctx()
    spec = HorizonSpec.training(seq_len=GOLDEN["seq_len"], micro_batches=1, optimizer=AdamW)
    report = estimate_horizon(spec, _training_module, _training_inputs, ctx)

    full_flops = report.per_step[0].flops.total_flops
    update_flops = report.per_step[1].flops.total_flops
    assert report.total_flops == full_flops + update_flops
    from zepto.analysis import account_flops, lower

    train_graph = compose_graph(
        _training_module,
        (
            Tensor(shape=(GOLDEN["seq_len"],)),
            Tensor(shape=(GOLDEN["seq_len"],), semantic_type="labels", requires_grad=False),
        ),
    )
    fwd_only = account_flops(lower(train_graph, replace(ctx, phase="forward"))).total_flops
    # Reject a phantom extra forward: total is full (fwd+bwd) + Adam, not +fwd again.
    assert report.total_flops < full_flops + fwd_only


def test_optimizer_state_in_state_final() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=128, micro_batches=1, optimizer=AdamW)
    report = estimate_horizon(spec, _linear_module, _linear_inputs, ctx)
    assert report.state_final.optimizer is not None
    assert report.state_final.optimizer.bytes > 0


def test_peak_vram_includes_optimizer_state() -> None:
    ctx = _golden_ctx()
    spec = HorizonSpec.training(seq_len=GOLDEN["seq_len"], micro_batches=1, optimizer=AdamW)
    report = estimate_horizon(spec, _training_module, _training_inputs, ctx)

    from zepto.analysis import lower
    from zepto.analysis.horizon.state import parameter_bytes

    train_graph = compose_graph(
        _training_module,
        (
            Tensor(shape=(GOLDEN["seq_len"],)),
            Tensor(shape=(GOLDEN["seq_len"],), semantic_type="labels", requires_grad=False),
        ),
    )
    train_lowered = lower(train_graph, replace(ctx, phase="forward"))
    params = parameter_bytes(train_lowered)
    opt_bytes = report.state_final.optimizer.bytes if report.state_final.optimizer else 0
    weight_grads = report.per_step[0].memory.breakdown.weight_grads
    assert report.peak_vram >= params + opt_bytes
    assert weight_grads > 0
    assert report.peak_vram >= params + opt_bytes + weight_grads
