"""Discovery rules for decomposed logit soft-cap regions."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

LOGIT_SOFT_CAP_PATTERN = PatternMatchRule(
    id="pat-logit-soft-cap-decomposed",
    kind="region/logit_softcap",
    priority=6,
    op_families=(
        "divide",
        "tanh",
        "multiply",
        "multiply",
    ),
    edge_constraints=(
        (0, 1, "output_to_input"),
        (1, 2, "output_to_input"),
        (2, 3, "output_to_input"),
    ),
)

LOGIT_SOFT_CAP_PROVENANCE = ProvenanceMatchRule(
    id="prov-logit-soft-cap",
    kind="region/logit_softcap",
    priority=6,
    component_type="LogitSoftCap",
    require_contiguous_in_graph_order=True,
)
