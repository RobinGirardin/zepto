"""Discovery rules for decomposed SiLU regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

SILU_PATTERN = PatternMatchRule(
    id="pat-silu-decomposed",
    kind="region/silu",
    priority=10,
    op_families=("sigmoid", "multiply"),
    edge_constraints=(
        (0, 1, "output_to_second_input"),  # σ(X) → multiply right input
    ),
    constraints=(
        PatternConstraint(kind="silu_mul_x_sigmoid_x"),
    ),
)

SILU_PROVENANCE = ProvenanceMatchRule(
    id="prov-silu",
    kind="region/silu",
    priority=10,
    component_type="SiLU",
    require_contiguous_in_graph_order=True,
)
