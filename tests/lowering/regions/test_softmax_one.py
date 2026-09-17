"""Discovery, lowering, and variant tests for region/softmax-one."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from modules.attention.attention_softmax_with_sink import AttentionSoftmaxWithSink
from zepto.semantic import ResourceEventKind

_HEADS = 4
_SEQ = 8
_NUMEL = _HEADS * _SEQ * _SEQ


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _compose_softmax_one():
    return compose_graph(
        lambda _ctx: AttentionSoftmaxWithSink(_HEADS),
        (Tensor(shape=(_HEADS, _SEQ, _SEQ), requires_grad=True),),
    )


def _softmax_one_regions(graph, ctx, registry):
    return [
        r for r in discover_regions(graph, ctx, registry) if r.kind == "region/softmax-one"
    ]


def _softmax_one_node(lowered):
    nodes = [
        n for n in lowered.nodes if n.implementation.startswith("region/softmax-one/")
    ]
    assert len(nodes) == 1
    return nodes[0]


def test_compose_softmax_one_discovers_region() -> None:
    graph = _compose_softmax_one()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = _softmax_one_regions(graph, reference_invocation(), registry)
    assert len(regions) == 1
    assert len(regions[0].operation_ids) == 1
    assert regions[0].anchor.component_type == "AttentionSoftmaxWithSink"


def test_softmax_one_unfused_without_fused_capability() -> None:
    graph = _compose_softmax_one()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 1


def test_softmax_one_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_softmax_one()
    ctx = _fused_context(
        module_implementation_pins={"AttentionSoftmaxWithSink": "region/softmax-one/reference"}
    )
    lowered = lower(graph, ctx)
    op = _softmax_one_node(lowered)
    assert op.implementation == "region/softmax-one/reference"
    assert op.forward_flops == 5 * _NUMEL + _HEADS * _SEQ
    assert op.backward_flops == 4 * _HEADS * _SEQ * (_SEQ + 1)
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(allocs) == 1
    assert len(saves) == 1


def test_softmax_one_megatron_variant_flops() -> None:
    graph = _compose_softmax_one()
    ctx = _fused_context(
        module_implementation_pins={"AttentionSoftmaxWithSink": "region/softmax-one/megatron"}
    )
    lowered = lower(graph, ctx)
    op = _softmax_one_node(lowered)
    assert op.implementation == "region/softmax-one/megatron"
    assert op.forward_flops == 5 * _HEADS * _SEQ * (_SEQ + 1)


def test_softmax_one_fusion_map_absorbs_op() -> None:
    graph = _compose_softmax_one()
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == 1
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_softmax_one_megatron_wins_over_reference_on_tie() -> None:
    graph = _compose_softmax_one()
    lowered = lower(graph, _fused_context())
    selection = lowered.region_selections[0]
    assert selection.chosen.id == "region/softmax-one/megatron"
    assert selection.reason in ("highest_priority", "only_candidate")


def test_softmax_one_fused_vs_unfused_vram_delta() -> None:
    graph = _compose_softmax_one()
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
    assert unfused_allocs == 1
    assert fused_allocs == 1
