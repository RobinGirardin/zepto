"""Discovery rules for decomposed CappedFusedLinearCrossEntropy regions."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

LINEAR_CE_SOFTCAP_PATTERN = PatternMatchRule(
    id="pat-linear-ce-softcap-decomposed",
    kind="region/linear_ce_softcap",
    priority=8,
    op_families=(
        "linear_matmul",
        "divide",
        "tanh",
        "multiply",
        "multiply",
        "exp",
        "reduce_sum",
        "divide",
        "gather",
        "log",
        "multiply",
        "reduce_sum",
    ),
    edge_constraints=(
        (0, 1, "output_to_input"),
        (1, 2, "output_to_input"),
        (2, 3, "output_to_input"),
        (3, 4, "output_to_input"),
        (4, 5, "output_to_input"),
    ),
)

LINEAR_CE_SOFTCAP_PROVENANCE = ProvenanceMatchRule(
    id="prov-linear-ce-softcap",
    kind="region/linear_ce_softcap",
    priority=8,
    component_type="CappedFusedLinearCrossEntropy",
    module_path_envelope=True,
    require_contiguous_in_graph_order=True,
)
