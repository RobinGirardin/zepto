"""Provenance-based region discovery."""

from __future__ import annotations

from collections import defaultdict

from ...graph import StructuralGraph
from ..context import InvocationContext
from ..region import (
    ProvenanceMatchRule,
    StructuralRegion,
    compute_region_boundaries,
    make_region_id,
    operations_contiguous_in_graph,
)


class ProvenanceRegionMatcher:
    """Discover module-scoped regions from provenance metadata."""

    def __init__(self, rules: tuple[ProvenanceMatchRule, ...]) -> None:
        self._rules = rules

    def find(
        self,
        graph: StructuralGraph,
        context: InvocationContext,
    ) -> tuple[StructuralRegion, ...]:
        del context
        groups: dict[tuple[tuple[str, ...], str | None], list] = defaultdict(list)
        for operation_id in graph.operations:
            op = graph.operation(operation_id)
            key = (op.provenance.module_path, op.provenance.component_type)
            groups[key].append(operation_id)

        regions: list[StructuralRegion] = []
        for rule in self._rules:
            for (module_path, component_type), operation_ids in groups.items():
                if not self._matches_rule(rule, module_path, component_type):
                    continue
                op_tuple = tuple(operation_ids)
                if rule.require_contiguous_in_graph_order and not (
                    operations_contiguous_in_graph(graph, op_tuple)
                ):
                    continue
                anchor = graph.operation(op_tuple[0]).provenance
                boundary_inputs, boundary_outputs, parameter_ids = (
                    compute_region_boundaries(graph, op_tuple)
                )
                regions.append(
                    StructuralRegion(
                        id=make_region_id(rule.kind, anchor, op_tuple[0]),
                        kind=rule.kind,
                        anchor=anchor,
                        operation_ids=op_tuple,
                        boundary_inputs=boundary_inputs,
                        boundary_outputs=boundary_outputs,
                        parameter_ids=parameter_ids,
                        matcher_id=rule.id,
                    )
                )
        return tuple(regions)

    @staticmethod
    def _matches_rule(
        rule: ProvenanceMatchRule,
        module_path: tuple[str, ...],
        component_type: str | None,
    ) -> bool:
        if rule.component_type is not None and rule.component_type != component_type:
            return False
        if rule.module_path_suffix:
            suffix = rule.module_path_suffix
            if len(module_path) < len(suffix):
                return False
            if module_path[-len(suffix) :] != suffix:
                return False
        return True
