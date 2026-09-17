"""Discovery, lowering, and variant tests for region/masked_softmax."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from modules.attention.gqa import GroupedQueryAttention
from zepto.semantic import ResourceEventKind

_SEQ = 8
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_HIDDEN = _HEADS * _HEAD_DIM
_NUMEL = _HEADS * _SEQ * _SEQ

_SCORE_PATH = ("add", "multiply", "exp", "reduce_sum", "divide")


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _compose_gqa():
    return compose_graph(
        lambda _ctx: GroupedQueryAttention(
            _HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM
        ),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )


def _masked_softmax_regions(graph, ctx, registry):
    return [r for r in discover_regions(graph, ctx, registry) if r.kind == "region/masked_softmax"]


def _masked_softmax_node(lowered):
    nodes = [
        n
        for n in lowered.nodes
        if n.implementation.startswith("region/masked_softmax")
    ]
    assert len(nodes) == 1
    return nodes[0]


def test_compose_gqa_discovers_masked_softmax_region() -> None:
    graph = _compose_gqa()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = _masked_softmax_regions(
        graph, reference_invocation(), registry
    )
    assert len(regions) == 1
    assert len(regions[0].operation_ids) == 5
    families = tuple(
        graph.node(op_id).operation_family
        for op_id in regions[0].operation_ids
    )
    assert families == _SCORE_PATH


def test_masked_softmax_unfused_without_fused_capability() -> None:
    graph = _compose_gqa()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    score_steps = [
        step
        for step in plan.steps
        if step.kind == "operation"
        and graph.node(step.operation_id).operation_family in _SCORE_PATH
    ]
    assert len(score_steps) == 5


def test_masked_softmax_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _fused_context())
    op = _masked_softmax_node(lowered)
    assert op.implementation == "region/masked_softmax/reference"
    assert op.forward_flops == 5 * _NUMEL
    assert op.backward_flops == 4 * _NUMEL
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(allocs) == 1
    assert len(saves) == 1


def test_masked_softmax_fusion_map_absorbs_all_ops() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _fused_context())
    region = _masked_softmax_node(lowered)
    fused_ops = [
        op_id
        for op_id, region_node_id in lowered.fusion_map.items()
        if region_node_id == region.id
    ]
    assert len(fused_ops) == 5


def test_masked_softmax_megatron_wins_on_cuda() -> None:
    graph = _compose_gqa()
    ctx = _fused_context(hardware="cuda")
    lowered = lower(graph, ctx)
    op = _masked_softmax_node(lowered)
    assert op.implementation == "region/masked_softmax/megatron"


def test_masked_softmax_reference_wins_on_non_cuda_hardware() -> None:
    graph = _compose_gqa()
    ctx = _fused_context(hardware="mps")
    lowered = lower(graph, ctx)
    op = _masked_softmax_node(lowered)
    assert op.implementation == "region/masked_softmax/reference"


def test_masked_softmax_fused_vs_unfused_vram_delta() -> None:
    graph = _compose_gqa()
    unfused = lower(graph, reference_invocation())
    fused = lower(graph, _fused_context())

    unfused_score_allocs = sum(
        1
        for node in unfused.nodes
        if node.node_id
        and graph.node(node.node_id).operation_family in _SCORE_PATH
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    fused_score_allocs = sum(
        1
        for node in fused.nodes
        if node.implementation.startswith("region/masked_softmax")
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    assert unfused_score_allocs >= 3
    assert fused_score_allocs == 1
    assert unfused_score_allocs - fused_score_allocs >= 2
