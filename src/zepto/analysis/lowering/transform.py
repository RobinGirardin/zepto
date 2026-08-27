"""Graph-to-lowered graph transformation."""

from __future__ import annotations

from types import MappingProxyType

from zepto.graph.graph import Graph
from ..lowered import LoweredGraph, LoweredNode
from .context import InvocationContext
from .defaults import DEFAULT_REGISTRY
from .helpers import build_estimation_context, ensure_lowered_edge
from .registry import ImplementationSelection, LoweringRegistry, select_implementation
from .validation import LoweredNodeValidator

_NODE_VALIDATOR = LoweredNodeValidator()


def lower(
    graph: Graph,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
) -> LoweredGraph:
    """Lower one graph for a concrete invocation context."""
    active_registry = registry if registry is not None else DEFAULT_REGISTRY
    lowered_edges: dict[str, object] = {}
    edge_map: dict = {}
    nodes: list[LoweredNode] = []
    node_map: dict = {}
    selections: list[ImplementationSelection] = []

    for input_id in graph.inputs:
        ensure_lowered_edge(
            input_id,
            graph,
            context,
            lowered_edges,
            edge_map,
        )

    for graph_node_id in graph.node_order:
        node = graph.node(graph_node_id)
        for bound_edge_id in node.input_edges:
            ensure_lowered_edge(
                bound_edge_id,
                graph,
                context,
                lowered_edges,
                edge_map,
            )

        impl, selection = select_implementation(
            node, graph, context, active_registry
        )
        estimation = build_estimation_context(node, graph, context)
        lowered_node = impl.lower(
            node,
            graph,
            context,
            edge_map,
            estimation=estimation,
            lowered_edges=lowered_edges,
        )
        _NODE_VALIDATOR.validate(lowered_node)
        nodes.append(lowered_node)
        node_map[node.id] = lowered_node.id
        selections.append(selection)

    return LoweredGraph(
        edges=MappingProxyType(dict(lowered_edges)),
        nodes=tuple(nodes),
        context=context,
        edge_map=MappingProxyType(dict(edge_map)),
        node_map=MappingProxyType(dict(node_map)),
        selections=tuple(selections),
    )
