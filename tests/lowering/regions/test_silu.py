"""Discovery, lowering, and variant tests for region/silu."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.layers.silu import SiLU
from zepto.semantic import ResourceEventKind


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_compose_silu_discovers_region() -> None:
    graph = compose_graph(
        lambda ctx: SiLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    silu = [r for r in regions if r.kind == "region/silu"]
    assert len(silu) == 1
    assert len(silu[0].operation_ids) == 2
    assert silu[0].anchor.component_type == "SiLU"


def test_silu_fused_lowering_flops_and_allocs() -> None:
    graph = compose_graph(
        lambda ctx: SiLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/silu"
    assert op.forward_flops == 5 * 4 * 8
    assert op.backward_flops == 8 * 4 * 8
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 1  # output only; σ(X) elided
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert saves[0].value == op.input_edges[0]


def test_silu_fusion_map_absorbs_all_ops() -> None:
    graph = compose_graph(
        lambda ctx: SiLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.fusion_map) == 2
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_silu_unfused_identity_baseline() -> None:
    graph = compose_graph(
        lambda ctx: SiLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
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
    assert unfused_allocs == 2  # σ(X) temp + Y
    assert fused_allocs == 1
    unfused_bwd = sum(n.backward_flops for n in unfused.nodes)
    fused_bwd = fused.nodes[0].backward_flops
    # Identity chain undercounts sigmoid VJP (6n); fused leaf is authoritative (8n).
    assert unfused_bwd < fused_bwd


def test_silu_no_backward_without_grad() -> None:
    graph = compose_graph(
        lambda ctx: SiLU(),
        (Tensor(shape=(4, 8), requires_grad=False),),
    )
    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]
    assert op.backward_flops == 0
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 0


def test_silu_module_lowers_to_region() -> None:
    graph = compose_graph(
        lambda ctx: SiLU(),
        (Tensor(shape=(4,), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert lowered.nodes[0].implementation == "region/silu"
