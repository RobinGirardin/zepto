"""Discovery rules for fused attention-sink softmax regions (boundary A)."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

SOFTMAX_ONE_PATTERN = PatternMatchRule(
    id="pat-softmax-one-fused-op",
    kind="region/softmax-one",
    priority=5,
    op_families=("attention_softmax_with_sink",),
)

SOFTMAX_ONE_OP_SEQUENCES: tuple[tuple[str, ...], ...] = (
    SOFTMAX_ONE_PATTERN.op_families,
)

SOFTMAX_ONE_PROVENANCE = ProvenanceMatchRule(
    id="prov-softmax-one",
    kind="region/softmax-one",
    priority=5,
    component_type="AttentionSoftmaxWithSink",
    require_contiguous_in_graph_order=True,
)
