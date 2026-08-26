"""Pattern-based region discovery."""

from __future__ import annotations

from ...graph import StructuralGraph
from ...ids import OperationId
from ..context import InvocationContext
from ..helpers import is_zero_operand
from ..region import (
    PatternConstraint,
    PatternMatchRule,
    StructuralRegion,
    compute_region_boundaries,
    make_region_id,
)


class PatternRegionMatcher:
    """Discover regions from operation subgraph patterns."""

    def __init__(self, rules: tuple[PatternMatchRule, ...]) -> None:
        self._rules = rules

    def find(
        self,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> tuple[StructuralRegion, ...]:
        del context
        regions: list[StructuralRegion] = []
        for rule in self._rules:
            regions.extend(self._match_sequence(graph, rule))
        return tuple(regions)

    def matches_subgraph(
        self,
        graph: StructuralGraph,
        operation_ids: tuple[OperationId, ...],
        rule: PatternMatchRule,
    ) -> bool:
        """Return whether an existing op set satisfies a pattern rule."""
        if len(operation_ids) != len(rule.op_families):
            return False
        for op_id, expected_family in zip(
            operation_ids, rule.op_families, strict=True
        ):
            if graph.operation(op_id).operation_family != expected_family:
                return False
        if not self._satisfies_edge_constraints(graph, operation_ids, rule):
            return False
        return self._satisfies_constraints(graph, operation_ids, rule)

    def _match_sequence(
        self,
        graph: StructuralGraph,
        rule: PatternMatchRule,
    ) -> list[StructuralRegion]:
        families = rule.op_families
        if not families:
            return []
        matches: list[StructuralRegion] = []
        seen: set[tuple] = set()

        for start_op_id in graph.operations:
            start = graph.operation(start_op_id)
            if start.operation_family != families[0]:
                continue
            chain = self._extend_chain(graph, start_op_id, families, rule)
            if chain is None:
                continue
            if chain in seen:
                continue
            seen.add(chain)
            if not self._satisfies_constraints(graph, chain, rule):
                continue
            anchor = graph.operation(chain[0]).provenance
            boundary_inputs, boundary_outputs, parameter_ids = (
                compute_region_boundaries(graph, chain)
            )
            matches.append(
                StructuralRegion(
                    id=make_region_id(rule.kind, anchor, chain[0]),
                    kind=rule.kind,
                    anchor=anchor,
                    operation_ids=chain,
                    boundary_inputs=boundary_inputs,
                    boundary_outputs=boundary_outputs,
                    parameter_ids=parameter_ids,
                    matcher_id=rule.id,
                )
            )
        return matches

    def _extend_chain(
        self,
        graph: StructuralGraph,
        start_id: OperationId,
        families: tuple[str, ...],
        rule: PatternMatchRule,
    ) -> tuple[OperationId, ...] | None:
        chain = [start_id]
        for step in range(1, len(families)):
            successor = self._find_dataflow_successor(
                graph, chain, step, families, rule.edge_constraints
            )
            if successor is None:
                return None
            chain.append(successor)
        if not self._satisfies_edge_constraints(graph, tuple(chain), rule):
            return None
        return tuple(chain)

    def _find_dataflow_successor(
        self,
        graph: StructuralGraph,
        chain: list[OperationId],
        step_index: int,
        families: tuple[str, ...],
        edges: tuple[tuple[int, int, str], ...],
    ) -> OperationId | None:
        expected_family = families[step_index]
        chain_set = set(chain)
        candidates: list[OperationId] = []
        for op_id in graph.operations:
            if op_id in chain_set:
                continue
            op = graph.operation(op_id)
            if op.operation_family != expected_family:
                continue
            if self._step_satisfies_edges(graph, chain, step_index, op_id, edges):
                candidates.append(op_id)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        last_index = graph.operations.index(chain[-1])
        candidates.sort(key=lambda item: graph.operations.index(item))
        for candidate in candidates:
            if graph.operations.index(candidate) > last_index:
                return candidate
        return candidates[0]

    @staticmethod
    def _step_satisfies_edges(
        graph: StructuralGraph,
        chain: list[OperationId],
        step_index: int,
        consumer_op_id: OperationId,
        edges: tuple[tuple[int, int, str], ...],
    ) -> bool:
        relevant = [edge for edge in edges if edge[1] == step_index]
        if not relevant:
            return True
        for from_index, _to_index, edge_kind in relevant:
            if from_index >= len(chain):
                return False
            if not _edge_satisfied(
                graph, chain[from_index], consumer_op_id, edge_kind
            ):
                return False
        return True

    @staticmethod
    def _satisfies_edge_constraints(
        graph: StructuralGraph,
        chain: tuple[OperationId, ...],
        rule: PatternMatchRule,
    ) -> bool:
        for from_index, to_index, edge_kind in rule.edge_constraints:
            if from_index >= len(chain) or to_index >= len(chain):
                return False
            if not _edge_satisfied(
                graph, chain[from_index], chain[to_index], edge_kind
            ):
                return False
        return True

    def _satisfies_constraints(
        self,
        graph: StructuralGraph,
        operation_ids: tuple[OperationId, ...],
        rule: PatternMatchRule,
    ) -> bool:
        for constraint in rule.constraints:
            if not self._check_constraint(graph, operation_ids, constraint):
                return False
        return True

    @staticmethod
    def _check_constraint(
        graph: StructuralGraph,
        operation_ids: tuple[OperationId, ...],
        constraint: PatternConstraint,
    ) -> bool:
        if constraint.kind == "right_operand_is_zero":
            op = graph.operation(operation_ids[0])
            if len(op.input_tensors) < 2:
                return False
            right = graph.tensor(op.input_tensors[1])
            return is_zero_operand(right, graph)
        if constraint.kind == "parameter_count":
            params: set = set()
            for op_id in operation_ids:
                params.update(graph.operation(op_id).parameter_ids)
            limit = constraint.max_parameters
            if limit is not None and len(params) > limit:
                return False
        if constraint.kind == "min_rank":
            op = graph.operation(operation_ids[0])
            if not op.input_tensors:
                return False
            rank = len(graph.tensor(op.input_tensors[0]).metadata.shape)
            minimum = constraint.min_rank or 0
            if rank < minimum:
                return False
        return True


def _edge_satisfied(
    graph: StructuralGraph,
    producer_op_id: OperationId,
    consumer_op_id: OperationId,
    edge_kind: str,
) -> bool:
    producer_op = graph.operation(producer_op_id)
    consumer_op = graph.operation(consumer_op_id)
    producer_outputs = set(producer_op.output_tensors)
    if edge_kind in ("output_to_input", "output_to_first_input"):
        if not consumer_op.input_tensors:
            return False
        return consumer_op.input_tensors[0] in producer_outputs
    if edge_kind == "output_to_second_input":
        if len(consumer_op.input_tensors) < 2:
            return False
        return consumer_op.input_tensors[1] in producer_outputs
    return False
