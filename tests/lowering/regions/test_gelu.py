"""Discovery, lowering, and variant tests for region/gelu and region/gelu_erf."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Tensor, compose_graph
from modules.layers.gelu import GELUErf, GELUTanh
from zepto.semantic import ResourceEventKind


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_compose_gelu_tanh_discovers_region() -> None:
    graph = compose_graph(
        lambda _ctx: GELUTanh(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    gelu = [r for r in regions if r.kind == "region/gelu"]
    assert len(gelu) == 1
    assert len(gelu[0].operation_ids) == 9
    assert gelu[0].anchor.component_type == "GELUTanh"


def test_compose_gelu_erf_discovers_region() -> None:
    graph = compose_graph(
        lambda _ctx: GELUErf(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    gelu = [r for r in regions if r.kind == "region/gelu_erf"]
    assert len(gelu) == 1
    assert len(gelu[0].operation_ids) == 5
    assert gelu[0].anchor.component_type == "GELUErf"


def test_gelu_tanh_fused_lowering_flops_and_allocs() -> None:
    graph = compose_graph(
        lambda _ctx: GELUTanh(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/gelu"
    assert op.forward_flops == 12 * 4 * 8
    assert op.backward_flops == 20 * 4 * 8
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 1
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert saves[0].value == op.input_edges[0]


def test_gelu_erf_fused_lowering_flops_and_allocs() -> None:
    graph = compose_graph(
        lambda _ctx: GELUErf(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/gelu_erf"
    assert op.forward_flops == 8 * 4 * 8
    assert op.backward_flops == 17 * 4 * 8
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 1
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert saves[0].value == op.input_edges[0]


def test_gelu_tanh_fusion_map_absorbs_all_ops() -> None:
    graph = compose_graph(
        lambda _ctx: GELUTanh(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.fusion_map) == 9
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_gelu_erf_fusion_map_absorbs_all_ops() -> None:
    graph = compose_graph(
        lambda _ctx: GELUErf(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.fusion_map) == 5
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_gelu_tanh_unfused_identity_baseline() -> None:
    graph = compose_graph(
        lambda _ctx: GELUTanh(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = _identity_registry()
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    assert not regions
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 9

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
    assert unfused_allocs == 9
    assert fused_allocs == 1


def test_gelu_erf_unfused_identity_baseline() -> None:
    graph = compose_graph(
        lambda _ctx: GELUErf(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = _identity_registry()
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    assert not regions
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 5

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
    assert unfused_allocs == 5
    assert fused_allocs == 1


def test_gelu_no_backward_without_grad() -> None:
    for module_factory in (GELUTanh, GELUErf):
        graph = compose_graph(
            lambda _ctx, factory=module_factory: factory(),
            (Tensor(shape=(4, 8), requires_grad=False),),
        )
        lowered = lower(graph, reference_invocation())
        op = lowered.nodes[0]
        assert op.backward_flops == 0
        saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
        assert len(saves) == 0


def test_gelu_tanh_and_erf_do_not_cross_match() -> None:
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()

    tanh_graph = compose_graph(
        lambda c: GELUTanh(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    tanh_regions = discover_regions(tanh_graph, ctx, registry)
    assert all(r.kind == "region/gelu" for r in tanh_regions)
    assert not any(r.kind == "region/gelu_erf" for r in tanh_regions)

    erf_graph = compose_graph(
        lambda c: GELUErf(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    erf_regions = discover_regions(erf_graph, ctx, registry)
    assert all(r.kind == "region/gelu_erf" for r in erf_regions)
    assert not any(r.kind == "region/gelu" for r in erf_regions)
