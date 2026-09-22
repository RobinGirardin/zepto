"""Discovery, lowering, and variant tests for region/softmax."""

from __future__ import annotations

import math

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.attention.softmax import Softmax
from zepto.semantic import ResourceEventKind

_TILE = (4, 8, 8)
_NUMEL = 4 * 8 * 8


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _compose_scaled_softmax():
    return compose_graph(
        lambda ctx: Softmax(scale=1.0 / math.sqrt(8.0)),
        (Tensor(shape=_TILE, requires_grad=True),),
    )


def _compose_unscaled_softmax():
    return compose_graph(
        lambda ctx: Softmax(),
        (Tensor(shape=_TILE, requires_grad=True),),
    )


def test_compose_softmax_discovers_region_scaled() -> None:
    graph = _compose_scaled_softmax()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    softmax = [r for r in regions if r.kind == "region/softmax"]
    assert len(softmax) == 1
    assert len(softmax[0].operation_ids) == 4
    assert softmax[0].anchor.component_type == "Softmax"


def test_compose_softmax_discovers_region_unscaled() -> None:
    graph = _compose_unscaled_softmax()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    softmax = [r for r in regions if r.kind == "region/softmax"]
    assert len(softmax) == 1
    assert len(softmax[0].operation_ids) == 3
    assert softmax[0].anchor.component_type == "Softmax"


def test_softmax_unfused_without_fused_capability() -> None:
    graph = _compose_scaled_softmax()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 4


def test_softmax_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_scaled_softmax()
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/softmax/reference"
    assert op.forward_flops == 5 * _NUMEL
    assert op.backward_flops == 4 * _NUMEL
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(allocs) == 1
    assert len(saves) == 1


def test_softmax_fusion_map_absorbs_all_ops() -> None:
    graph = _compose_scaled_softmax()
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == 4
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_softmax_cuda_variant_wins_on_cuda() -> None:
    graph = _compose_scaled_softmax()
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/softmax/cuda"


def test_softmax_reference_wins_on_non_cuda_hardware() -> None:
    graph = _compose_scaled_softmax()
    ctx = _fused_context(hardware="mps")
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/softmax/reference"


def test_softmax_fused_vs_unfused_vram_delta() -> None:
    graph = _compose_scaled_softmax()
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
    assert unfused_allocs == 4
    assert fused_allocs == 1
    assert fused.nodes[0].forward_flops == 5 * _NUMEL
    assert sum(n.forward_flops for n in unfused.nodes) < fused.nodes[0].forward_flops
