"""Discovery rules for decomposed GELU regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

GELU_TANH_PATTERN = PatternMatchRule(
    id="pat-gelu-tanh-decomposed",
    kind="region/gelu",
    priority=10,
    op_families=(
        "multiply",
        "multiply",
        "multiply",
        "add",
        "multiply",
        "tanh",
        "add",
        "multiply",
        "multiply",
    ),
    constraints=(PatternConstraint(kind="gelu_tanh_activation"),),
)

GELU_ERF_PATTERN = PatternMatchRule(
    id="pat-gelu-erf-decomposed",
    kind="region/gelu_erf",
    priority=10,
    op_families=("multiply", "erf", "add", "multiply", "multiply"),
    constraints=(PatternConstraint(kind="gelu_erf_activation"),),
)

GELU_TANH_PROVENANCE = ProvenanceMatchRule(
    id="prov-gelu-tanh",
    kind="region/gelu",
    priority=10,
    component_type="GELUTanh",
    require_contiguous_in_graph_order=True,
)

GELU_ERF_PROVENANCE = ProvenanceMatchRule(
    id="prov-gelu-erf",
    kind="region/gelu_erf",
    priority=10,
    component_type="GELUErf",
    require_contiguous_in_graph_order=True,
)
