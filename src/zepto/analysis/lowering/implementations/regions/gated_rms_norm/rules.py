"""Discovery rules for decomposed GatedRMSNorm regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

GATED_RMS_NORM_PATTERN = PatternMatchRule(
    id="pat-gated-rms-norm-decomposed",
    kind="region/gated_rms_norm",
    priority=5,
    op_families=(
        "cast",  # value → fp32
        "cast",  # gate → fp32
        "multiply",  # square value
        "reduce_sum",
        "divide",  # mean of squares
        "add",  # + eps
        "square_root",
        "divide",  # normalize value
        "cast",  # downcast to activation dtype
        "parameter_scale",  # × γ
        "sigmoid",  # on gate
        "multiply",  # gate × σ  (SiLU)
        "multiply",  # scaled × silu_gate
    ),
    edge_constraints=(
        (0, 2, "output_to_input"),
        (2, 3, "output_to_input"),
        (3, 4, "output_to_first_input"),
        (4, 5, "output_to_first_input"),
        (5, 6, "output_to_input"),
        (6, 7, "output_to_second_input"),
        (0, 7, "output_to_first_input"),
        (7, 8, "output_to_input"),
        (8, 9, "output_to_input"),
        (1, 10, "output_to_input"),
        (10, 11, "output_to_second_input"),
        (1, 11, "output_to_first_input"),
        (9, 12, "output_to_first_input"),
        (11, 12, "output_to_second_input"),
    ),
    constraints=(
        PatternConstraint(kind="parameter_count", max_parameters=1),
    ),
)

GATED_RMS_NORM_PROVENANCE = ProvenanceMatchRule(
    id="prov-gated-rms-norm",
    kind="region/gated_rms_norm",
    priority=5,
    component_type="GatedRMSNorm",
    require_contiguous_in_graph_order=True,
)
