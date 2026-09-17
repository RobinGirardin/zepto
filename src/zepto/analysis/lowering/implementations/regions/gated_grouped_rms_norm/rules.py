"""Discovery rules for decomposed GatedGroupedRMSNorm regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

GATED_GROUPED_RMS_NORM_PATTERN = PatternMatchRule(
    id="pat-gated-grouped-rms-norm-decomposed",
    kind="region/gated_grouped_rms_norm",
    priority=5,
    op_families=(
        "cast",
        "cast",
        "sigmoid",
        "multiply",
        "multiply",
        "reshape",
        "multiply",
        "reduce_sum",
        "divide",
        "add",
        "square_root",
        "divide",
        "reshape",
        "cast",
        "parameter_scale",
    ),
    edge_constraints=(
        (0, 4, "output_to_first_input"),
        (1, 2, "output_to_input"),
        (1, 3, "output_to_first_input"),
        (2, 3, "output_to_second_input"),
        (3, 4, "output_to_second_input"),
        (4, 5, "output_to_input"),
        (5, 6, "output_to_first_input"),
        (5, 6, "output_to_second_input"),
        (6, 7, "output_to_input"),
        (7, 8, "output_to_first_input"),
        (8, 9, "output_to_first_input"),
        (9, 10, "output_to_input"),
        (10, 11, "output_to_second_input"),
        (5, 11, "output_to_first_input"),
        (11, 12, "output_to_input"),
        (12, 13, "output_to_input"),
        (13, 14, "output_to_input"),
    ),
    constraints=(
        PatternConstraint(kind="parameter_count", max_parameters=1),
    ),
)

GATED_GROUPED_RMS_NORM_PROVENANCE = ProvenanceMatchRule(
    id="prov-gated-grouped-rms-norm",
    kind="region/gated_grouped_rms_norm",
    priority=5,
    component_type="GatedGroupedRMSNorm",
    require_contiguous_in_graph_order=True,
)
