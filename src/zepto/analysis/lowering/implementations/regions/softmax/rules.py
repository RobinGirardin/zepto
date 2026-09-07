"""Discovery rules for decomposed Softmax regions."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

# Identity chains from ``src/zepto/modules/softmax.py`` (with / without scale).
SOFTMAX_PATTERN_SCALED = PatternMatchRule(
    id="pat-softmax-scaled",
    kind="region/softmax",
    priority=5,
    op_families=("multiply", "exp", "reduce_sum", "divide"),
)

SOFTMAX_PATTERN = PatternMatchRule(
    id="pat-softmax-decomposed",
    kind="region/softmax",
    priority=5,
    op_families=("exp", "reduce_sum", "divide"),
)

SOFTMAX_OP_SEQUENCES: tuple[tuple[str, ...], ...] = (
    SOFTMAX_PATTERN_SCALED.op_families,
    SOFTMAX_PATTERN.op_families,
)

SOFTMAX_PROVENANCE = ProvenanceMatchRule(
    id="prov-softmax",
    kind="region/softmax",
    priority=5,
    component_type="Softmax",
    require_contiguous_in_graph_order=True,
)
