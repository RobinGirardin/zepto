"""Structural-to-lowered graph transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from ..graph import StructuralGraph
from ..ids import OperationId
from ..lowered import LoweredGraph, LoweredOperation
from .context import InvocationContext
from .defaults import DEFAULT_REGISTRY
from .discovery import discover_regions
from .helpers import (
    RegionEstimationContext,
    build_estimation_context,
    build_region_estimation_context,
    ensure_lowered_tensor,
)
from .plan import LoweringPlan, build_lowering_plan
from .registry import (
    ImplementationSelection,
    LoweringRegistry,
    RegionImplementationSelection,
    select_implementation,
    select_region_implementation,
)
from .region import StructuralRegion
from .validation import LoweredOperationValidator

_OPERATION_VALIDATOR = LoweredOperationValidator()


@dataclass
class LoweringState:
    """Mutable state accumulated during one lowering pass."""

    lowered_tensors: dict[str, object] = field(default_factory=dict)
    tensor_map: dict = field(default_factory=dict)
    operations: list[LoweredOperation] = field(default_factory=list)
    operation_map: dict[OperationId, str] = field(default_factory=dict)
    fusion_map: dict[OperationId, str] = field(default_factory=dict)
    region_map: dict[str, tuple[OperationId, ...]] = field(default_factory=dict)
    selections: list[ImplementationSelection] = field(default_factory=list)
    region_selections: list[RegionImplementationSelection] = field(
        default_factory=list
    )
    consumed: set[OperationId] = field(default_factory=set)


def lower(
    graph: StructuralGraph,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
) -> LoweredGraph:
    """Lower one structural graph for a concrete invocation context."""
    active_registry = registry if registry is not None else DEFAULT_REGISTRY
    state = LoweringState()

    for input_id in graph.inputs:
        ensure_lowered_tensor(
            input_id,
            graph,
            context,
            state.lowered_tensors,
            state.tensor_map,
        )

    regions = discover_regions(graph, context, active_registry)
    plan = build_lowering_plan(graph, regions, context, active_registry)

    for step in plan.steps:
        if step.kind == "region":
            assert step.region is not None
            lower_region(
                step.region, graph, context, active_registry, state
            )
        else:
            assert step.operation_id is not None
            lower_operation(
                step.operation_id, graph, context, active_registry, state
            )

    return assemble_lowered_graph(graph, context, state)


def lower_operation(
    operation_id: OperationId,
    graph: StructuralGraph,
    context: InvocationContext,
    registry: LoweringRegistry,
    state: LoweringState,
) -> None:
    """Lower one structural operation via the per-op route."""
    structural = graph.operation(operation_id)
    for tensor_id in structural.input_tensors:
        ensure_lowered_tensor(
            tensor_id,
            graph,
            context,
            state.lowered_tensors,
            state.tensor_map,
        )

    impl, selection = select_implementation(
        structural, graph, context, registry
    )
    estimation = build_estimation_context(structural, graph, context)
    lowered_op = impl.lower(
        structural,
        graph,
        context,
        state.tensor_map,
        estimation=estimation,
        lowered_tensors=state.lowered_tensors,
    )
    _OPERATION_VALIDATOR.validate(lowered_op)
    state.operations.append(lowered_op)
    state.operation_map[structural.id] = lowered_op.id
    state.selections.append(selection)


def lower_region(
    region: StructuralRegion,
    graph: StructuralGraph,
    context: InvocationContext,
    registry: LoweringRegistry,
    state: LoweringState,
) -> None:
    """Lower one structural region via the N-to-1 route."""
    for op_id in region.operation_ids:
        structural = graph.operation(op_id)
        for tensor_id in structural.input_tensors:
            ensure_lowered_tensor(
                tensor_id,
                graph,
                context,
                state.lowered_tensors,
                state.tensor_map,
            )

    for tensor_id in region.boundary_inputs:
        ensure_lowered_tensor(
            tensor_id,
            graph,
            context,
            state.lowered_tensors,
            state.tensor_map,
        )

    impl, selection = select_region_implementation(
        region, graph, context, registry
    )
    estimation = build_region_estimation_context(region, graph, context)
    lowered_op = impl.lower(
        region,
        graph,
        context,
        state.tensor_map,
        estimation=estimation,
        lowered_tensors=state.lowered_tensors,
    )
    _OPERATION_VALIDATOR.validate(lowered_op)
    state.operations.append(lowered_op)
    state.region_map[region.id] = region.operation_ids
    state.region_selections.append(selection)
    for op_id in region.operation_ids:
        state.operation_map[op_id] = lowered_op.id
        state.fusion_map[op_id] = lowered_op.id


def assemble_lowered_graph(
    graph: StructuralGraph,
    context: InvocationContext,
    state: LoweringState,
) -> LoweredGraph:
    """Finalize one lowered graph from accumulated lowering state."""
    del graph
    return LoweredGraph(
        tensors=MappingProxyType(dict(state.lowered_tensors)),
        operations=tuple(state.operations),
        context=context,
        tensor_map=MappingProxyType(dict(state.tensor_map)),
        operation_map=MappingProxyType(dict(state.operation_map)),
        selections=tuple(state.selections),
        fusion_map=MappingProxyType(dict(state.fusion_map)),
        region_map=MappingProxyType(dict(state.region_map)),
        region_selections=tuple(state.region_selections),
    )
