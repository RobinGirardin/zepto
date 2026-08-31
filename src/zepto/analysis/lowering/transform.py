"""Graph-to-lowered graph transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from zepto.graph.graph import Graph
from zepto.graph.ids import NodeId
from ..lowered import LoweredGraph, LoweredNode
from .context import InvocationContext
from .defaults import DEFAULT_REGISTRY
from .discovery import discover_regions
from .helpers import (
    build_estimation_context,
    build_region_estimation_context,
    ensure_lowered_edge,
)
from .plan import build_lowering_plan
from .registry import (
    ImplementationSelection,
    LoweringRegistry,
    RegionImplementationSelection,
    select_implementation,
    select_region_implementation,
)
from .region import Region
from .validation import LoweredNodeValidator

_NODE_VALIDATOR = LoweredNodeValidator()


@dataclass
class LoweringState:
    """Mutable state accumulated during one lowering pass."""

    lowered_edges: dict[str, object] = field(default_factory=dict)
    edge_map: dict = field(default_factory=dict)
    nodes: list[LoweredNode] = field(default_factory=list)
    node_map: dict[NodeId, str] = field(default_factory=dict)
    fusion_map: dict[NodeId, str] = field(default_factory=dict)
    region_map: dict[str, tuple[NodeId, ...]] = field(default_factory=dict)
    selections: list[ImplementationSelection] = field(default_factory=list)
    region_selections: list[RegionImplementationSelection] = field(
        default_factory=list
    )


def lower(
    graph: Graph,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
) -> LoweredGraph:
    """Lower one graph for a concrete invocation context."""
    active_registry = registry if registry is not None else DEFAULT_REGISTRY
    state = LoweringState()

    for input_id in graph.inputs:
        ensure_lowered_edge(
            input_id,
            graph,
            context,
            state.lowered_edges,
            state.edge_map,
        )

    regions = discover_regions(graph, context, active_registry)
    plan = build_lowering_plan(graph, regions, context, active_registry)

    for step in plan.steps:
        if step.kind == "region":
            assert step.region is not None
            lower_region(step.region, graph, context, active_registry, state)
        else:
            assert step.operation_id is not None
            lower_operation(
                step.operation_id, graph, context, active_registry, state
            )

    return LoweredGraph(
        edges=MappingProxyType(dict(state.lowered_edges)),
        nodes=tuple(state.nodes),
        context=context,
        edge_map=MappingProxyType(dict(state.edge_map)),
        node_map=MappingProxyType(dict(state.node_map)),
        selections=tuple(state.selections),
        fusion_map=MappingProxyType(dict(state.fusion_map)),
        region_map=MappingProxyType(dict(state.region_map)),
        region_selections=tuple(state.region_selections),
    )


def lower_operation(
    operation_id: NodeId,
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
    state: LoweringState,
) -> None:
    """Lower one graph node via the per-op route."""
    node = graph.node(operation_id)
    for bound_edge_id in node.input_edges:
        ensure_lowered_edge(
            bound_edge_id,
            graph,
            context,
            state.lowered_edges,
            state.edge_map,
        )

    impl, selection = select_implementation(node, graph, context, registry)
    estimation = build_estimation_context(node, graph, context)
    lowered_node = impl.lower(
        node,
        graph,
        context,
        state.edge_map,
        estimation=estimation,
        lowered_edges=state.lowered_edges,
    )
    _NODE_VALIDATOR.validate(lowered_node)
    state.nodes.append(lowered_node)
    state.node_map[node.id] = lowered_node.id
    state.selections.append(selection)


def lower_region(
    region: Region,
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
    state: LoweringState,
) -> None:
    """Lower one region via the N-to-1 route."""
    for op_id in region.operation_ids:
        node = graph.node(op_id)
        for bound_edge_id in node.input_edges:
            ensure_lowered_edge(
                bound_edge_id,
                graph,
                context,
                state.lowered_edges,
                state.edge_map,
            )

    for edge_id in region.boundary_inputs:
        ensure_lowered_edge(
            edge_id,
            graph,
            context,
            state.lowered_edges,
            state.edge_map,
        )

    impl, selection = select_region_implementation(
        region, graph, context, registry
    )
    estimation = build_region_estimation_context(region, graph, context)
    lowered_node = impl.lower(
        region,
        graph,
        context,
        state.edge_map,
        estimation=estimation,
        lowered_edges=state.lowered_edges,
    )
    _NODE_VALIDATOR.validate(lowered_node)
    state.nodes.append(lowered_node)
    state.region_map[region.id] = region.operation_ids
    state.region_selections.append(selection)
    for op_id in region.operation_ids:
        state.node_map[op_id] = lowered_node.id
        state.fusion_map[op_id] = lowered_node.id
