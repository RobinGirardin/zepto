"""Discovery rules for fused masked-softmax regions (boundary B)."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

# GQA score path: Add(causal_mask) → Softmax(scale) decomposed chain.
MASKED_SOFTMAX_PATTERN = PatternMatchRule(
    id="pat-masked-softmax-gqa-score",
    kind="region/masked_softmax",
    priority=6,
    op_families=("add", "multiply", "exp", "reduce_sum", "divide"),
)

MASKED_SOFTMAX_OP_SEQUENCES: tuple[tuple[str, ...], ...] = (
    MASKED_SOFTMAX_PATTERN.op_families,
)

# Reserved for future hybrid discovery; pattern-only matching is used today because
# the full GQA module envelope exceeds the five-op score subgraph.
MASKED_SOFTMAX_PROVENANCE = ProvenanceMatchRule(
    id="prov-masked-softmax-gqa",
    kind="region/masked_softmax",
    priority=6,
    component_type="GroupedQueryAttention",
    require_contiguous_in_graph_order=True,
)

MASKED_SOFTMAX_SINK_PATTERN = PatternMatchRule(
    id="pat-masked-softmax-sink-gqa-score",
    kind="region/masked_softmax",
    priority=7,
    op_families=("add", "multiply", "attention_softmax_with_sink"),
)

MASKED_SOFTMAX_SINK_OP_SEQUENCES: tuple[tuple[str, ...], ...] = (
    MASKED_SOFTMAX_SINK_PATTERN.op_families,
    ("add", "attention_softmax_with_sink"),
)
