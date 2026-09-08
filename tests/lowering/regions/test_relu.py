"""Discovery, lowering, and variant tests for region/relu."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.relu import ReLU
from zepto.semantic import ResourceEventKind


def _saved_input_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"saved_input"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def test_compose_relu_discovers_region() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    relu = [r for r in regions if r.kind == "region/relu"]
    assert len(relu) == 1
    assert len(relu[0].operation_ids) == 1
    assert relu[0].anchor.component_type == "ReLU"


def test_relu_lowering_flops_and_mask() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/relu"
    assert op.forward_flops == 4 * 8
    assert op.backward_flops == 4 * 8
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 2  # output + relu_mask
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("mask" in aux for aux in op.auxiliary_edges)


def test_relu_saved_input_saves_input_not_mask() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, _saved_input_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/relu/saved_input"
    assert op.forward_flops == 4 * 8
    assert op.backward_flops == 4 * 8
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 1  # output only
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert saves[0].value == op.input_edges[0]
    assert op.auxiliary_edges == ()


def test_relu_default_wins_without_saved_input_capability() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert lowered.nodes[0].implementation == "region/relu"


def test_relu_fusion_map_single_op() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.fusion_map) == 1
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_relu_module_lowers_to_region() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (Tensor(shape=(4,), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation())
    assert lowered.nodes[0].implementation == "region/relu"
