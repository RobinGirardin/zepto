"""Discovery rules for decomposed Softplus regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

SOFTPLUS_PATTERN = PatternMatchRule(
    id="pat-softplus-decomposed",
    kind="region/softplus",
    priority=10,
    op_families=("exp", "add", "log"),
    edge_constraints=(
        (0, 1, "output_to_second_input"),  # Exp → Add right input
        (1, 2, "output_to_input"),  # Add → Log input (log_add_exp_chain)
    ),
    constraints=(
        PatternConstraint(kind="softplus_decomposed_chain"),
    ),
)

SOFTPLUS_PROVENANCE = ProvenanceMatchRule(
    id="prov-softplus",
    kind="region/softplus",
    priority=10,
    component_type="Softplus",
    require_contiguous_in_graph_order=True,
)
