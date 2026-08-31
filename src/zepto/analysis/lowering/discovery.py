"""Region discovery orchestration."""

from __future__ import annotations

from dataclasses import replace

from zepto.graph.graph import Graph
from .context import InvocationContext
from .matchers.pattern import PatternRegionMatcher
from .matchers.provenance import ProvenanceRegionMatcher
from .region import PatternMatchRule, ProvenanceMatchRule, Region
from .registry import LoweringRegistry


def discover_regions(
    graph: Graph,
    context: InvocationContext,
    registry: LoweringRegistry,
) -> tuple[Region, ...]:
    """Return region candidates according to each impl's registered rules."""
    candidates: list[Region] = []

    for kind in registry.registered_region_kinds():
        descriptor = registry.region_descriptor(kind)
        prov_rule = descriptor.provenance_rule
        pat_rule = descriptor.pattern_rule

        if prov_rule is not None and pat_rule is not None:
            candidates.extend(
                _discover_hybrid_regions(graph, context, prov_rule, pat_rule)
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
        if not pattern_matcher.matches_subgraph(
            graph, envelope.operation_ids, pattern_rule
        ):
            continue
        refined.append(
            replace(
                envelope,
                kind=pattern_rule.kind,
                matcher_id=f"hybrid:{provenance_rule.id}+{pattern_rule.id}",
            )
        )
    return tuple(refined)
