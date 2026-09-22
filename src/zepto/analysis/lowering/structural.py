"""Structural vs execution phases of graph lowering."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from zepto.graph.graph import Graph

from ..lowered import LoweredGraph
from .context import InvocationContext
from .discovery import discover_regions
from .helpers import ensure_lowered_edge
from .mask_elision import apply_mask_elision
from .plan import (
    LoweringPlan,
    build_lowering_plan_from_winners,
    resolve_overlaps,
)
from .region import Region
from .transform import (
    LoweringState,
    _register_parameters,
    lower_operation,
    lower_region,
)

if TYPE_CHECKING:
    from .registry import LoweringRegistry


@dataclass(frozen=True, slots=True)
class StructuralLowering:
    """Cached discovery + schedule; reusable for the same graph and structural key."""

    discovered_regions: tuple[Region, ...]
    winning_regions: tuple[Region, ...]
    plan: LoweringPlan


def discover_and_plan(
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
) -> StructuralLowering:
    """Run region discovery, overlap resolution, and plan construction once."""
    regions = discover_regions(graph, context, registry)
    winning_regions = resolve_overlaps(graph, regions, context, registry)
    plan = build_lowering_plan_from_winners(graph, winning_regions)
    return StructuralLowering(
        discovered_regions=regions,
        winning_regions=winning_regions,
        plan=plan,
    )


def execute_lowering_plan(
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
    structural: StructuralLowering,
) -> LoweredGraph:
    """Execute a cached plan with a fresh invocation context."""
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

    for step in structural.plan.steps:
        if step.kind == "region":
            assert step.region is not None
            lower_region(step.region, graph, context, registry, state)
        else:
            assert step.operation_id is not None
            lower_operation(
                step.operation_id, graph, context, registry, state
            )

    apply_mask_elision(
        graph,
        structural.winning_regions,
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
        discovered_regions=structural.discovered_regions,
        winning_regions=structural.winning_regions,
        lowering_plan=structural.plan,
        source_graph=graph,
    )
