"""Lowering-time views over fusible graph node subgraphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.graph.graph import Graph
from zepto.graph.ids import EdgeId, NodeId, ParameterId
from zepto.graph.provenance import Provenance


@dataclass(frozen=True, slots=True)
class Region:
    """One candidate fusible slice of a graph."""

    id: str
    kind: str
    anchor: Provenance
    operation_ids: tuple[NodeId, ...]
    boundary_inputs: tuple[EdgeId, ...]
    boundary_outputs: tuple[EdgeId, ...]
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
        "silu_mul_x_sigmoid_x",
        "gelu_tanh_activation",
        "gelu_erf_activation",
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
    graph: Graph,
    operation_ids: tuple[NodeId, ...],
) -> tuple[tuple[EdgeId, ...], tuple[EdgeId, ...], tuple[ParameterId, ...]]:
    """Compute boundary tensors and parameters for one operation set."""
    op_set = frozenset(operation_ids)
    boundary_inputs: list[EdgeId] = []
    boundary_outputs: list[EdgeId] = []
    parameters: list[ParameterId] = []
    graph_outputs = frozenset(graph.outputs)

    def _append_unique(target: list, item: object) -> None:
        if item not in target:
            target.append(item)

    for op_id in operation_ids:
        op = graph.node(op_id)
        for edge_id in op.input_edges:
            edge = graph.edge(edge_id)
            if edge.producer is None or edge.producer.node_id not in op_set:
                _append_unique(boundary_inputs, edge_id)
        for edge_id in op.output_edges:
            edge = graph.edge(edge_id)
            has_external_consumer = any(
                consumer.node_id not in op_set for consumer in edge.consumers
            )
            if has_external_consumer or edge_id in graph_outputs:
                _append_unique(boundary_outputs, edge_id)
        for parameter_id in op.parameter_ids:
            _append_unique(parameters, parameter_id)

    return tuple(boundary_inputs), tuple(boundary_outputs), tuple(parameters)


def operations_contiguous_in_graph(
    graph: Graph,
    operation_ids: tuple[NodeId, ...],
) -> bool:
    """Return whether ops form one contiguous block in graph order."""
    if not operation_ids:
        return True
    indices = sorted(graph.node_order.index(op_id) for op_id in operation_ids)
    return indices[-1] - indices[0] + 1 == len(indices)


def make_region_id(kind: str, anchor: Provenance, first_op: NodeId) -> str:
    """Build a stable region identity string."""
    path = ".".join(anchor.module_path) or "root"
    component = anchor.component_type or "unknown"
    return f"{kind}:{path}:{component}:{first_op.index}"
