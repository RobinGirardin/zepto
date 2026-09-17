"""Tests for TopK semantic operation."""

from __future__ import annotations

import pytest

from zepto.analysis import lower, reference_invocation
from zepto.analysis.resolved import ResolvedValue
from zepto.compose import Module, Tensor, compose_graph
from zepto.semantic import EstimationContext, ResourceEventKind, TopK
from zepto.semantic.metadata import DType


def _resolved(tensor: Tensor) -> ResolvedValue:
    return ResolvedValue(tensor=tensor, role=None, dtype=tensor.dtype or DType.FP32)


def test_infer_outputs_basic() -> None:
    op = TopK(k=4, dim=-1)
    values, indices = op.infer_outputs((Tensor(shape=(8, 32)),))
    assert values.shape == (8, 4)
    assert indices.shape == (8, 4)
    assert indices.semantic_type == "routing_index"
    assert indices.requires_grad is False


def test_infer_outputs_grouped() -> None:
    op = TopK(k=1, dim=-1)
    values, indices = op.infer_outputs((Tensor(shape=(4, 2, 4)),))
    assert values.shape == (4, 2, 1)
    assert indices.shape == (4, 2, 1)


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="positive"):
        TopK(k=0).infer_outputs((Tensor(shape=(4, 8)),))
    with pytest.raises(ValueError, match="must not exceed"):
        TopK(k=9).infer_outputs((Tensor(shape=(4, 8)),))
    with pytest.raises(ValueError, match="ranked"):
        TopK(k=1).infer_outputs((Tensor(shape=()),))
    with pytest.raises(ValueError, match="out of range"):
        TopK(k=1, dim=3).infer_outputs((Tensor(shape=(4, 8)),))


def test_compose_and_allocate() -> None:
    class _TopKFixture(Module):
        module_kind = "TopKFixture"

        def forward(self, scores: Tensor) -> Tensor:
            values, _indices = TopK(k=2, dim=-1)(scores)
            return values  # type: ignore[return-value]

    graph = compose_graph(
        lambda _ctx: _TopKFixture(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    allocs = [
        ev
        for node in lowered.nodes
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    ]
    assert len(allocs) == 2


def test_forward_flops_zero() -> None:
    op = TopK(k=8, dim=-1)
    ctx = EstimationContext(
        port_values=(
            ("input", _resolved(Tensor(shape=(128, 256), requires_grad=True))),
            ("values", _resolved(Tensor(shape=(128, 8)))),
            ("indices", _resolved(Tensor(shape=(128, 8), semantic_type="routing_index"))),
        )
    )
    assert op.forward_flops(ctx) == 0
