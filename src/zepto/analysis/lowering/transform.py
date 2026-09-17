"""Graph-to-lowered graph transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from zepto.graph.graph import Graph

if TYPE_CHECKING:
    from zepto.analysis.horizon.state import StatePortRegistry
from zepto.graph.ids import NodeId, ParameterId
from zepto.graph.parameter import parameter_as_tensor
from ..lowered import LoweredGraph, LoweredNode, LoweredParameter, StatePortEvent
from .context import InvocationContext
from .role import RoleContext, resolve_lowered_edge
from .defaults import DEFAULT_REGISTRY
from .discovery import discover_regions
from .helpers import (
    build_estimation_context,
    build_region_estimation_context,
    ensure_lowered_edge,
)
from .mask_elision import apply_mask_elision
from .plan import build_lowering_plan, resolve_overlaps
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
    lowered_parameters: dict[str, LoweredParameter] = field(default_factory=dict)
    parameter_map: dict[ParameterId, str] = field(default_factory=dict)
    edge_map: dict = field(default_factory=dict)
    nodes: list[LoweredNode] = field(default_factory=list)
    node_map: dict[NodeId, str] = field(default_factory=dict)
    fusion_map: dict[NodeId, str] = field(default_factory=dict)
    region_map: dict[str, tuple[NodeId, ...]] = field(default_factory=dict)
    selections: list[ImplementationSelection] = field(default_factory=list)
    region_selections: list[RegionImplementationSelection] = field(
        default_factory=list
    )
    state_port_events: list[StatePortEvent] = field(default_factory=list)


def lower(
    graph: Graph,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
    state_ports: StatePortRegistry | None = None,
) -> LoweredGraph:
    """Lower one graph for a concrete invocation context."""
    if state_ports is not None:
        context = state_ports.bind_to_context(context)

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

    _register_parameters(graph, context, state)

    regions = discover_regions(graph, context, active_registry)
    winning_regions = resolve_overlaps(graph, regions, context, active_registry)
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

    apply_mask_elision(
        graph,
        winning_regions,
        nodes=state.nodes,
        node_map=state.node_map,
    )

    output_edge_ids = tuple(
        state.edge_map[output_id] for output_id in graph.outputs
    )

    return LoweredGraph(
        edges=MappingProxyType(dict(state.lowered_edges)),
        parameters=MappingProxyType(dict(state.lowered_parameters)),
        parameter_map=MappingProxyType(dict(state.parameter_map)),
        nodes=tuple(state.nodes),
        context=context,
        edge_map=MappingProxyType(dict(state.edge_map)),
        node_map=MappingProxyType(dict(state.node_map)),
        output_edge_ids=output_edge_ids,
        selections=tuple(state.selections),
        fusion_map=MappingProxyType(dict(state.fusion_map)),
        region_map=MappingProxyType(dict(state.region_map)),
        region_selections=tuple(state.region_selections),
        state_port_events=tuple(state.state_port_events),
    )


def _register_parameters(
    graph: Graph,
    context: InvocationContext,
    state: LoweringState,
) -> None:
    """Register graph parameters on the lowered graph."""
    for param_id, param in graph.parameters.items():
        lowered_id = f"p{param_id.index}"
        tensor, role = resolve_lowered_edge(
            parameter_as_tensor(param),
            role_ctx=RoleContext(graph=graph, port_direction="parameter"),
            context=context,
        )
        storage_id = f"param:{param_id.index}"
        state.lowered_parameters[lowered_id] = LoweredParameter(
            id=lowered_id,
            parameter_id=param_id,
            tensor=tensor,
            role=role,
            storage_id=storage_id,
            trainable=param.trainable,
        )
        state.parameter_map[param_id] = lowered_id


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
    lower_kwargs: dict = {
        "estimation": estimation,
        "lowered_edges": state.lowered_edges,
    }
    if region.kind in (
        "region/gqa",
        "region/gqa-sink",
        "region/depthwise_causal_conv1d",
        "region/gated_delta_scan",
        "region/mamba2_scan",
    ):
        lower_kwargs["state_port_collector"] = state.state_port_events
    lowered_node = impl.lower(
        region,
        graph,
        context,
        state.edge_map,
        **lower_kwargs,
    )
    _NODE_VALIDATOR.validate(lowered_node)
    state.nodes.append(lowered_node)
    state.region_map[region.id] = region.operation_ids
    state.region_selections.append(selection)
    for op_id in region.operation_ids:
        state.node_map[op_id] = lowered_node.id
        state.fusion_map[op_id] = lowered_node.id
