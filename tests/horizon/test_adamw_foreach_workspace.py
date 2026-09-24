"""AdamW foreach workspace is ephemeral and raises the 5P training floor."""

from __future__ import annotations

from zepto.analysis import (
    AdamW,
    AdamWPolicy,
    HorizonSpec,
    estimate_horizon,
    reference_invocation,
)
from zepto.analysis.horizon.state import trainable_parameter_elements
from zepto.compose import Tensor
from zepto.modules.layers.linear import Linear

_IN_FEATURES = 64
_OUT_FEATURES = 32
_WIDE = 4096


def _linear_module(_compose):
    return Linear(_IN_FEATURES, _OUT_FEATURES)


def _linear_inputs(step, _ctx, _state):
    return (Tensor(shape=(step.batch, step.seq_len, _IN_FEATURES)),)


def _wide_module(_compose):
    return Linear(_WIDE, _WIDE)


def _wide_inputs(step, _ctx, _state):
    return (Tensor(shape=(step.batch, step.seq_len, _WIDE)),)


def test_foreach_workspace_is_one_p() -> None:
    ctx = reference_invocation(optim_prec=4)
    assert AdamW.update_workspace_bytes(trainable_elements=1000, context=ctx) == 4000
    no_foreach = AdamWPolicy(foreach=False)
    assert no_foreach.update_workspace_bytes(trainable_elements=1000, context=ctx) == 0


def test_foreach_workspace_not_persisted_in_optimizer_state() -> None:
    ctx = reference_invocation(optim_prec=4)
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)
    report, sim = estimate_horizon(
        spec, _linear_module, _linear_inputs, ctx, return_simulation=True
    )
    trainable = trainable_parameter_elements(sim.timeline[0].lowered)
    moments = AdamW.state_bytes(trainable_elements=trainable, context=ctx)
    workspace = AdamW.update_workspace_bytes(
        trainable_elements=trainable, context=ctx
    )
    assert workspace == moments // 2
    assert sim.state_final.optimizer is not None
    assert sim.state_final.optimizer.bytes == moments
    assert sim.state_initial.optimizer is not None
    assert sim.state_initial.optimizer.bytes == moments


def test_foreach_raises_param_dominated_peak_to_five_p() -> None:
    ctx = reference_invocation(optim_prec=4)
    spec = HorizonSpec.training(seq_len=2, micro_batches=1, optimizer=AdamW)
    spec_off = HorizonSpec.training(
        seq_len=2, micro_batches=1, optimizer=AdamWPolicy(foreach=False)
    )
    report_on = estimate_horizon(spec, _wide_module, _wide_inputs, ctx)
    report_off, sim = estimate_horizon(
        spec_off, _wide_module, _wide_inputs, ctx, return_simulation=True
    )
    train = report_on.per_step[0].memory.breakdown
    p = train.parameters
    g = train.weight_grads
    runtime = train.runtime_workspace
    adam = sim.state_final.optimizer.bytes if sim.state_final.optimizer else 0
    workspace = AdamW.update_workspace_bytes(
        trainable_elements=p // 4, context=ctx
    )
    floor_off = p + g + adam + runtime
    floor_on = floor_off + workspace
    assert report_off.peak_vram >= floor_off
    assert report_on.peak_vram >= floor_on
    assert report_on.peak_vram == max(report_off.peak_vram, floor_on)
    assert report_on.peak_vram > report_off.peak_vram
