"""Lowering-time views over structural operation subgraphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..graph import StructuralGraph
from ..ids import OperationId, ParameterId, TensorId
from ..provenance import Provenance


@dataclass(frozen=True, slots=True)
class StructuralRegion:
    """One candidate fusible slice of a structural graph."""

    id: str
    kind: str
    anchor: Provenance
    operation_ids: tuple[OperationId, ...]
    boundary_inputs: tuple[TensorId, ...]
    boundary_outputs: tuple[TensorId, ...]
    parameter_ids: tuple[ParameterId, ...]
    matcher_id: str


@dataclass(frozen=True, slots=True)
class RegionMatchRule:
    """Base type for region discovery rules."""

    id: str
    kind: str
    priority: int = 0


@dataclass(frozen=True, slots=True)
class ProvenanceMatchRule(RegionMatchRule):
    """Match all ops sharing a module invocation envelope."""

    component_type: str | None = None
    module_path_suffix: tuple[str, ...] = ()
    require_contiguous_in_graph_order: bool = True


@dataclass(frozen=True, slots=True)
class PatternConstraint:
    """Structural predicate on one matched operation."""

    kind: Literal[
        "right_operand_is_zero",
        "parameter_count",
        "min_rank",
    ]
    max_parameters: int | None = None
    min_rank: int | None = None


@dataclass(frozen=True, slots=True)
class PatternMatchRule(RegionMatchRule):
    """Match a subgraph by operation pattern."""

    op_families: tuple[str, ...] = ()
    constraints: tuple[PatternConstraint, ...] = ()
    edge_constraints: tuple[tuple[int, int, str], ...] = ()


def compute_region_boundaries(
    graph: StructuralGraph,
    operation_ids: tuple[OperationId, ...],
) -> tuple[tuple[TensorId, ...], tuple[TensorId, ...], tuple[ParameterId, ...]]:
    """Compute boundary tensors and parameters for one operation set."""
    op_set = frozenset(operation_ids)
    boundary_inputs: list[TensorId] = []
    boundary_outputs: list[TensorId] = []
    parameters: list[ParameterId] = []
    graph_outputs = frozenset(graph.outputs)

    def _append_unique(target: list, item: object) -> None:
        if item not in target:
            target.append(item)

    for op_id in operation_ids:
        op = graph.operation(op_id)
        for tensor_id in op.input_tensors:
            tensor = graph.tensor(tensor_id)
            if tensor.producer is None or tensor.producer.operation_id not in op_set:
                _append_unique(boundary_inputs, tensor_id)
        for tensor_id in op.output_tensors:
            tensor = graph.tensor(tensor_id)
            has_external_consumer = any(
                consumer.operation_id not in op_set
                for consumer in tensor.consumers
            )
            if has_external_consumer or tensor_id in graph_outputs:
                _append_unique(boundary_outputs, tensor_id)
        for parameter_id in op.parameter_ids:
            _append_unique(parameters, parameter_id)

    return tuple(boundary_inputs), tuple(boundary_outputs), tuple(parameters)


def operations_contiguous_in_graph(
    graph: StructuralGraph,
    operation_ids: tuple[OperationId, ...],
) -> bool:
    """Return whether ops form one contiguous block in graph order."""
    if not operation_ids:
        return True
    indices = sorted(graph.operations.index(op_id) for op_id in operation_ids)
    return indices[-1] - indices[0] + 1 == len(indices)


def make_region_id(kind: str, anchor: Provenance, first_op: OperationId) -> str:
    """Build a stable region identity string."""
    path = ".".join(anchor.module_path) or "root"
    component = anchor.component_type or "unknown"
    return f"{kind}:{path}:{component}:{first_op.index}"
