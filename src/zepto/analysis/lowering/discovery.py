"""Region discovery orchestration."""

from __future__ import annotations

from dataclasses import replace

from zepto.graph.graph import Graph
from zepto.graph.ids import NodeId
from .context import InvocationContext
from .matchers.pattern import PatternRegionMatcher, pattern_rule_variants
from .matchers.provenance import ProvenanceRegionMatcher
from .region import (
    PatternMatchRule,
    ProvenanceMatchRule,
    Region,
    compute_region_boundaries,
)
from .registry import LoweringRegistry


def discover_regions(
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
) -> tuple[Region, ...]:
    """Return region candidates according to each impl's registered rules."""
    candidates: list[Region] = []

    for kind in registry.registered_region_kinds():
        seen_rule_pairs: set[tuple[str, str]] = set()
        for impl in registry.region_candidates(kind):
            descriptor = impl.descriptor
            prov_rule = descriptor.provenance_rule
            pat_rule = descriptor.pattern_rule
            pair_key = (
                prov_rule.id if prov_rule is not None else "",
                pat_rule.id if pat_rule is not None else "",
            )
            if pair_key in seen_rule_pairs:
                continue
            seen_rule_pairs.add(pair_key)

            if prov_rule is not None and pat_rule is not None:
                hybrid = _discover_hybrid_regions(
                    graph, context, prov_rule, pat_rule
                )
                if hybrid:
                    candidates.extend(hybrid)
                elif prov_rule.module_path_envelope:
                    candidates.extend(
                        PatternRegionMatcher((pat_rule,)).find(graph, context)
                    )
            elif pat_rule is not None:
                candidates.extend(
                    PatternRegionMatcher((pat_rule,)).find(graph, context)
                )
            elif prov_rule is not None:
                candidates.extend(
                    ProvenanceRegionMatcher((prov_rule,)).find(graph, context)
                )

    return tuple(candidates)


def _discover_hybrid_regions(
    graph: Graph,
    context: InvocationContext,
    provenance_rule: ProvenanceMatchRule,
    pattern_rule: PatternMatchRule,
) -> tuple[Region, ...]:
    """Provenance envelope gated by internal pattern match."""
    provenance_matcher = ProvenanceRegionMatcher((provenance_rule,))
    pattern_matcher = PatternRegionMatcher((pattern_rule,))

    refined: list[Region] = []
    for envelope in provenance_matcher.find(graph, context):
        chain: tuple[NodeId, ...] | None = None
        for variant in pattern_rule_variants(pattern_rule):
            if len(envelope.operation_ids) == len(variant.op_families):
                if pattern_matcher.matches_subgraph(
                    graph, envelope.operation_ids, variant
                ):
                    chain = envelope.operation_ids
                    break
            found = pattern_matcher.find_chain_within(
                graph, envelope.operation_ids, variant
            )
            if found is not None:
                chain = found
                break
        if chain is None:
            continue
        boundary_inputs, boundary_outputs, parameter_ids = compute_region_boundaries(
            graph, chain
        )
        refined.append(
            replace(
                envelope,
                kind=pattern_rule.kind,
                operation_ids=chain,
                boundary_inputs=boundary_inputs,
                boundary_outputs=boundary_outputs,
                parameter_ids=parameter_ids,
                matcher_id=f"hybrid:{provenance_rule.id}+{pattern_rule.id}",
            )
        )
    return tuple(refined)
