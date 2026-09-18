"""Discovery, lowering, and variant tests for region/squared_relu."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.layers.relu import ReLU
from zepto.semantic import Multiply, ResourceEventKind


class _SquaredReLUFixture(Module):
    """Maximum(X, 0) → Multiply(relu_out, relu_out) test chain."""

    module_kind = "SquaredReLUFixture"

    def forward(self, x: Tensor) -> Tensor:
        relu_out = ReLU()(x)
        return Multiply()(relu_out, relu_out)


def _compose_squared_relu(*, requires_grad: bool = True):
    return compose_graph(
        lambda ctx: _SquaredReLUFixture(),
        (Tensor(shape=(4, 8), requires_grad=requires_grad),),
    )


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_compose_squared_relu_discovers_region() -> None:
    graph = _compose_squared_relu()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    squared = [r for r in regions if r.kind == "region/squared_relu"]
    assert len(squared) == 1
    assert len(squared[0].operation_ids) == 2


def test_squared_relu_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_squared_relu()
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/squared_relu"
    assert op.forward_flops == 2 * 4 * 8
    assert op.backward_flops == 3 * 4 * 8
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 1
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert saves[0].value == op.input_edges[0]


def test_squared_relu_fusion_map_absorbs_all_ops() -> None:
    graph = _compose_squared_relu()
    lowered = lower(graph, reference_invocation())
    assert len(lowered.fusion_map) == 2
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_squared_relu_unfused_identity_baseline() -> None:
    graph = _compose_squared_relu()
    registry = _identity_registry()
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    assert not regions
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 2

    unfused = lower(graph, ctx, registry=registry)
    fused = lower(graph, reference_invocation())
    unfused_allocs = sum(
        1
        for n in unfused.nodes
        for ev in n.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    fused_allocs = sum(
        1
        for n in fused.nodes
        for ev in n.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    assert unfused_allocs == 2
    assert fused_allocs == 1
    unfused_forward = sum(n.forward_flops for n in unfused.nodes)
    assert unfused_forward == 4 * 8


def test_squared_relu_no_backward_without_grad() -> None:
    graph = _compose_squared_relu(requires_grad=False)
    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]
    assert op.backward_flops == 0
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 0


def test_squared_relu_does_not_match_plain_relu() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    squared = [r for r in regions if r.kind == "region/squared_relu"]
    relu = [r for r in regions if r.kind == "region/relu"]
    assert not squared
    assert len(relu) == 1
