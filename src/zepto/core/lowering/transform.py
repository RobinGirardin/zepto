"""Structural-to-lowered graph transformation."""

from __future__ import annotations

from types import MappingProxyType

from ..graph import StructuralGraph
from ..lowered import LoweredGraph, LoweredOperation
from .context import InvocationContext
from .defaults import DEFAULT_REGISTRY
from .helpers import build_estimation_context, ensure_lowered_tensor
from .registry import ImplementationSelection, LoweringRegistry, select_implementation
from .validation import LoweredOperationValidator

_OPERATION_VALIDATOR = LoweredOperationValidator()


def lower(
    graph: StructuralGraph,
    context: InvocationContext,
    *,
    registry: LoweringRegistry | None = None,
) -> LoweredGraph:
    """Lower one structural graph for a concrete invocation context."""
    active_registry = registry if registry is not None else DEFAULT_REGISTRY
    lowered_tensors: dict[str, object] = {}
    tensor_map: dict = {}
    operations: list[LoweredOperation] = []
    operation_map: dict = {}
    selections: list[ImplementationSelection] = []

    for input_id in graph.inputs:
        ensure_lowered_tensor(
            input_id,
            graph,
            context,
            lowered_tensors,
            tensor_map,
        )

    for operation_id in graph.operations:
        structural = graph.operation(operation_id)
        for tensor_id in structural.input_tensors:
            ensure_lowered_tensor(
                tensor_id,
                graph,
                context,
                lowered_tensors,
                tensor_map,
            )

        impl, selection = select_implementation(
            structural, graph, context, active_registry
        )
        estimation = build_estimation_context(structural, graph, context)
        lowered_op = impl.lower(
            structural,
            graph,
            context,
            tensor_map,
            estimation=estimation,
            lowered_tensors=lowered_tensors,
        )
        _OPERATION_VALIDATOR.validate(lowered_op)
        operations.append(lowered_op)
        operation_map[structural.id] = lowered_op.id
        selections.append(selection)

    return LoweredGraph(
        tensors=MappingProxyType(dict(lowered_tensors)),
        operations=tuple(operations),
        context=context,
        tensor_map=MappingProxyType(dict(tensor_map)),
        operation_map=MappingProxyType(dict(operation_map)),
        selections=tuple(selections),
    )
