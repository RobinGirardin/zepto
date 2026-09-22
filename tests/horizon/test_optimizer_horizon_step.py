"""Optimizer horizon step: state-only simulation and FLOP accounting."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    StepKind,
    optimizer_update_flops,
    reference_invocation,
    simulate_horizon,
)
from zepto.analysis.horizon.account import account_horizon_flops
from zepto.analysis.horizon.optimizer_step import reference_lowered_for_optimizer
from zepto.analysis.horizon.spec import HorizonStep, optimizer_step
from zepto.analysis.horizon.state import parameter_bytes
from tests.horizon.test_horizon import _linear_inputs, _linear_module

_IN_FEATURES = 64
_OUT_FEATURES = 32


def test_optimizer_step_skips_compose() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)
    with patch(
        "zepto.analysis.horizon.simulate.compose_graph",
        wraps=__import__(
            "zepto.compose", fromlist=["compose_graph"]
        ).compose_graph,
    ) as compose_graph:
        simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    assert compose_graph.call_count == 2


def test_training_flops_not_triple_forward() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    hf = account_horizon_flops(sim)
    fwd = hf.per_step[0].total_flops
    bwd = hf.per_step[1].total_flops
    opt = hf.per_step[2].total_flops
    param_bytes = parameter_bytes(sim.timeline[2].lowered)
    bpe = param_bytes // (_IN_FEATURES * _OUT_FEATURES)
    expected_opt = optimizer_update_flops(AdamW, param_bytes, bytes_per_element=bpe)
    assert hf.per_step[2].forward_flops == 0
    assert opt == expected_opt
    assert hf.total_flops == fwd + bwd + opt
    assert hf.total_flops < 3 * fwd


def test_optimizer_row_flop_report() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    hf = account_horizon_flops(sim)
    last = hf.per_step[-1]
    record = sim.timeline[-1]
    assert record.step.kind == StepKind.OPTIMIZER
    param_bytes = parameter_bytes(record.lowered)
    bpe = param_bytes // (_IN_FEATURES * _OUT_FEATURES)
    assert last.forward_flops == 0
    assert last.total_flops == optimizer_update_flops(
        AdamW, param_bytes, bytes_per_element=bpe
    )


def test_optimizer_advances_state() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    assert sim.state_final.optimizer is not None
    assert sim.state_final.optimizer.bytes > 0
    assert sim.state_final.grad_accum is None


def test_reference_lowered_is_backward() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec.training(seq_len=8, micro_batches=1, optimizer=AdamW)
    sim = simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
    reference = reference_lowered_for_optimizer(sim.timeline)
    backward = sim.timeline[1].lowered
    assert reference is backward
    assert len(sim.timeline[2].lowered.nodes) == 0
    assert parameter_bytes(sim.timeline[2].lowered) == parameter_bytes(backward)


def test_optimizer_without_prior_step_raises() -> None:
    ctx = reference_invocation()
    spec = HorizonSpec(steps=[optimizer_step(seq_len=8)])
    with pytest.raises(ValueError, match="prior invocation"):
        simulate_horizon(spec, _linear_module, _linear_inputs, ctx)
