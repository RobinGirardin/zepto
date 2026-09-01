"""Discovery rules for decomposed RMSNorm regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

RMSNORM_PATTERN = PatternMatchRule(
    id="pat-rmsnorm-decomposed",
    kind="region/rmsnorm",
    priority=5,
    op_families=(
        "cast",            # 0: upcast to fp32
        "multiply",        # 1: square
        "reduce_sum",      # 2
        "divide",          # 3: mean of squares
        "add",             # 4: + eps
        "square_root",     # 5
        "divide",          # 6: x_fp32 / rms
        "cast",            # 7: downcast to activation dtype
        "parameter_scale", # 8: × γ
    ),
    edge_constraints=(
        (0, 1, "output_to_input"),           # x_fp32 → multiply (left)
        (1, 2, "output_to_input"),           # squared → reduce_sum
        (2, 3, "output_to_first_input"),     # sum → divide (left)
        (3, 4, "output_to_first_input"),     # variance → add (left)
        (4, 5, "output_to_input"),           # radicand → sqrt
        (5, 6, "output_to_second_input"),    # denom → divide (right)
        (0, 6, "output_to_first_input"),     # x_fp32 → divide (left)
        (6, 7, "output_to_input"),           # normalized → downcast
        (7, 8, "output_to_input"),           # normalized_act → parameter_scale
    ),
    constraints=(
        PatternConstraint(kind="parameter_count", max_parameters=1),
    ),
)

RMSNORM_PROVENANCE = ProvenanceMatchRule(
    id="prov-rmsnorm",
    kind="region/rmsnorm",
    priority=5,
    component_type="RMSNorm",
    require_contiguous_in_graph_order=True,
)
