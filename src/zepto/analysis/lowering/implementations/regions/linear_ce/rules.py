"""Discovery rules for decomposed FusedLinearCrossEntropy regions."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

# Matches the 8-op chain from ``src/zepto/modules/output/fused_linear_cross_entropy.py``.
# Rank-3 hidden inserts two Reshape ops (logits + labels) after the GEMM.
LINEAR_CE_FAMILIES = (
    "linear_matmul",
    "exp",
    "reduce_sum",
    "divide",
    "gather",
    "log",
    "multiply",
    "reduce_sum",
)
LINEAR_CE_BATCHED_FAMILIES = (
    "linear_matmul",
    "reshape",
    "reshape",
    "exp",
    "reduce_sum",
    "divide",
    "gather",
    "log",
    "multiply",
    "reduce_sum",
)

LINEAR_CE_PATTERN = PatternMatchRule(
    id="pat-linear-ce-decomposed",
    kind="region/linear_ce",
    priority=8,
    op_families=LINEAR_CE_FAMILIES,
    alternate_op_families=(LINEAR_CE_BATCHED_FAMILIES,),
)

LINEAR_CE_OP_SEQUENCES: tuple[tuple[str, ...], ...] = (
    LINEAR_CE_FAMILIES,
    LINEAR_CE_BATCHED_FAMILIES,
)

LINEAR_CE_PROVENANCE = ProvenanceMatchRule(
    id="prov-linear-ce",
    kind="region/linear_ce",
    priority=8,
    component_type="FusedLinearCrossEntropy",
    require_contiguous_in_graph_order=True,
)
