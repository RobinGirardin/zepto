"""Discovery, lowering, and variant tests for region/xielu."""

from __future__ import annotations

from types import MappingProxyType

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.layers.xielu import XIELU
from zepto.semantic import ResourceEventKind


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_compose_xielu_discovers_region() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    xielu = [r for r in regions if r.kind == "region/xielu"]
    assert len(xielu) == 1
    assert len(xielu[0].operation_ids) == 13
    assert xielu[0].anchor.component_type == "XIELU"


def test_xielu_unfused_without_fused_capability() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 13


def test_xielu_fused_lowering_flops_and_allocs() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/xielu/reference"
    assert op.forward_flops == 8 * 4 * 8
    assert op.backward_flops == 10 * 4 * 8
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 2  # output + sign_mask
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("sign_mask" in aux for aux in op.auxiliary_edges)


def test_xielu_fusion_map_absorbs_all_ops() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == 13
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_xielu_pin_reference() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    ctx = _fused_context(
        region_implementation_pins=MappingProxyType(
            {"region/xielu": "region/xielu/reference"}
        ),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/xielu/reference"


def test_xielu_cuda_wins_on_cuda_hardware() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/xielu/cuda"


def test_xielu_reference_wins_on_non_cuda_hardware() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    ctx = _fused_context(hardware="mps")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/xielu/reference"


def test_xielu_fused_vs_unfused_vram_delta() -> None:
    graph = compose_graph(
        lambda ctx: XIELU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    unfused = lower(graph, reference_invocation())
    fused = lower(graph, _fused_context())
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
    assert unfused_allocs == 13
    assert fused_allocs == 2
    assert sum(n.forward_flops for n in unfused.nodes) > fused.nodes[0].forward_flops
