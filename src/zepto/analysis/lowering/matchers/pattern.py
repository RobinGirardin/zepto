"""Pattern-based region discovery."""

from __future__ import annotations

from zepto.graph.graph import Graph
from zepto.graph.ids import NodeId
from ..context import InvocationContext
from ..helpers import is_zero_operand
from ..region import (
    PatternConstraint,
    PatternMatchRule,
    Region,
    compute_region_boundaries,
    make_region_id,
)


class PatternRegionMatcher:
    """Discover regions from operation subgraph patterns."""

    def __init__(self, rules: tuple[PatternMatchRule, ...]) -> None:
        self._rules = rules

    def find(
        self,
        graph: Graph,
        context: InvocationContext,
    ) -> tuple[Region, ...]:
        del context
        regions: list[Region] = []
        for rule in self._rules:
            regions.extend(self._match_sequence(graph, rule))
        return tuple(regions)

    def matches_subgraph(
        self,
        graph: Graph,
        operation_ids: tuple[NodeId, ...],
        rule: PatternMatchRule,
    ) -> bool:
        """Return whether an existing op set satisfies a pattern rule."""
        if len(operation_ids) != len(rule.op_families):
            return False
        for op_id, expected_family in zip(
            operation_ids, rule.op_families, strict=True
        ):
            if graph.node(op_id).operation_family != expected_family:
                return False
        if not self._satisfies_edge_constraints(graph, operation_ids, rule):
            return False
        return self._satisfies_constraints(graph, operation_ids, rule)

    def _match_sequence(
        self,
        graph: Graph,
        rule: PatternMatchRule,
    ) -> list[Region]:
        families = rule.op_families
        if not families:
            return []
        matches: list[Region] = []
        seen: set[tuple] = set()

        for start_op_id in graph.node_order:
            start = graph.node(start_op_id)
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
            anchor = graph.node(chain[0]).provenance
            boundary_inputs, boundary_outputs, parameter_ids = (
                compute_region_boundaries(graph, chain)
            )
            matches.append(
                Region(
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
        graph: Graph,
        start_id: NodeId,
        families: tuple[str, ...],
        rule: PatternMatchRule,
    ) -> tuple[NodeId, ...] | None:
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
        graph: Graph,
        chain: list[NodeId],
        step_index: int,
        families: tuple[str, ...],
        edges: tuple[tuple[int, int, str], ...],
    ) -> NodeId | None:
        expected_family = families[step_index]
        chain_set = set(chain)
        candidates: list[NodeId] = []
        for op_id in graph.node_order:
            if op_id in chain_set:
                continue
            op = graph.node(op_id)
            if op.operation_family != expected_family:
                continue
            if self._step_satisfies_edges(graph, chain, step_index, op_id, edges):
                candidates.append(op_id)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        last_index = graph.node_order.index(chain[-1])
        candidates.sort(key=lambda item: graph.node_order.index(item))
        for candidate in candidates:
            if graph.node_order.index(candidate) > last_index:
                return candidate
        return candidates[0]

    @staticmethod
    def _step_satisfies_edges(
        graph: Graph,
        chain: list[NodeId],
        step_index: int,
        consumer_op_id: NodeId,
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
        graph: Graph,
        chain: tuple[NodeId, ...],
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
        graph: Graph,
        operation_ids: tuple[NodeId, ...],
        rule: PatternMatchRule,
    ) -> bool:
        for constraint in rule.constraints:
            if not self._check_constraint(graph, operation_ids, constraint):
                return False
        return True

    @staticmethod
    def _check_constraint(
        graph: Graph,
        operation_ids: tuple[NodeId, ...],
        constraint: PatternConstraint,
    ) -> bool:
        if constraint.kind == "right_operand_is_zero":
            op = graph.node(operation_ids[0])
            if len(op.input_edges) < 2:
                return False
            right = graph.edge(op.input_edges[1])
            return is_zero_operand(right, graph)
        if constraint.kind == "parameter_count":
            params: set = set()
            for op_id in operation_ids:
                params.update(graph.node(op_id).parameter_ids)
            limit = constraint.max_parameters
            if limit is not None and len(params) > limit:
                return False
        if constraint.kind == "min_rank":
            op = graph.node(operation_ids[0])
            if not op.input_edges:
                return False
            rank = len(graph.edge(op.input_edges[0]).tensor.shape)
            minimum = constraint.min_rank or 0
            if rank < minimum:
                return False
        if constraint.kind == "silu_mul_x_sigmoid_x":
            if len(operation_ids) < 2:
                return False
            sigmoid_op = graph.node(operation_ids[0])
            multiply_op = graph.node(operation_ids[1])
            if not sigmoid_op.input_edges or not multiply_op.input_edges:
                return False
            return multiply_op.input_edges[0] == sigmoid_op.input_edges[0]
        if constraint.kind == "gelu_tanh_activation":
            return _check_gelu_tanh_activation(graph, operation_ids)
        if constraint.kind == "gelu_erf_activation":
            return _check_gelu_erf_activation(graph, operation_ids)
        if constraint.kind == "softplus_decomposed_chain":
            return _check_softplus_decomposed_chain(graph, operation_ids)
        return True


def _edge_semantic_type(graph: Graph, edge_id: object) -> str:
    return graph.edge(edge_id).tensor.semantic_type


def _check_softplus_decomposed_chain(
    graph: Graph,
    operation_ids: tuple[NodeId, ...],
) -> bool:
    if len(operation_ids) < 3:
        return False
    exp_op = graph.node(operation_ids[0])
    add_op = graph.node(operation_ids[1])
    log_op = graph.node(operation_ids[2])
    if not exp_op.input_edges or len(add_op.input_edges) < 2:
        return False
    if not exp_op.output_edges or not add_op.output_edges:
        return False
    exp_out = exp_op.output_edges[0]
    if add_op.input_edges[1] != exp_out:
        return False
    add_types = {
        _edge_semantic_type(graph, edge_id) for edge_id in add_op.input_edges
    }
    if "one" not in add_types:
        return False
    if not log_op.input_edges:
        return False
    return log_op.input_edges[0] in add_op.output_edges


def _check_gelu_tanh_activation(
    graph: Graph,
    operation_ids: tuple[NodeId, ...],
) -> bool:
    if len(operation_ids) < 9:
        return False
    mul_x_squared = graph.node(operation_ids[0])
    mul_x_cubed = graph.node(operation_ids[1])
    mul_kappa = graph.node(operation_ids[2])
    mul_scale = graph.node(operation_ids[4])
    tanh_op = graph.node(operation_ids[5])
    if len(mul_x_squared.input_edges) < 2 or len(mul_x_cubed.input_edges) < 2:
        return False
    x_edge = mul_x_squared.input_edges[0]
    if mul_x_squared.input_edges[0] != mul_x_squared.input_edges[1]:
        return False
    if x_edge not in mul_x_cubed.input_edges:
        return False
    if mul_x_squared.output_edges[0] not in mul_x_cubed.input_edges:
        return False
    kappa_types = {
        _edge_semantic_type(graph, edge_id) for edge_id in mul_kappa.input_edges
    }
    if "gelu_kappa" not in kappa_types:
        return False
    sqrt_types = {
        _edge_semantic_type(graph, edge_id) for edge_id in mul_scale.input_edges
    }
    if "gelu_sqrt_2_over_pi" not in sqrt_types:
        return False
    if not mul_scale.output_edges or not tanh_op.input_edges:
        return False
    return mul_scale.output_edges[0] in tanh_op.input_edges


def _check_gelu_erf_activation(
    graph: Graph,
    operation_ids: tuple[NodeId, ...],
) -> bool:
    if len(operation_ids) < 5:
        return False
    mul_scale = graph.node(operation_ids[0])
    erf_op = graph.node(operation_ids[1])
    add_op = graph.node(operation_ids[2])
    mul4 = graph.node(operation_ids[4])
    scale_types = {
        _edge_semantic_type(graph, edge_id) for edge_id in mul_scale.input_edges
    }
    if "gelu_inv_sqrt2" not in scale_types:
        return False
    if not mul_scale.output_edges or not erf_op.input_edges:
        return False
    if mul_scale.output_edges[0] not in erf_op.input_edges:
        return False
    add_types = {_edge_semantic_type(graph, edge_id) for edge_id in add_op.input_edges}
    if "one" not in add_types:
        return False
    half_types = {_edge_semantic_type(graph, edge_id) for edge_id in mul4.input_edges}
    return "gelu_half" in half_types


def _edge_satisfied(
    graph: Graph,
    producer_op_id: NodeId,
    consumer_op_id: NodeId,
    edge_kind: str,
) -> bool:
    producer_op = graph.node(producer_op_id)
    consumer_op = graph.node(consumer_op_id)
    producer_outputs = set(producer_op.output_edges)
    if edge_kind in ("output_to_input", "output_to_first_input"):
        if not consumer_op.input_edges:
            return False
        return consumer_op.input_edges[0] in producer_outputs
    if edge_kind == "output_to_second_input":
        if len(consumer_op.input_edges) < 2:
            return False
        return consumer_op.input_edges[1] in producer_outputs
    return False
