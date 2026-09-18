"""Discovery rules for decomposed L2-normalize regions."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

L2_NORMALIZE_PATTERN = PatternMatchRule(
    id="pat-l2-normalize-decomposed",
    kind="region/l2_normalize",
    priority=6,
    op_families=(
        "cast",  # 0: fp32 reduction dtype
        "multiply",  # 1: square
        "reduce_sum",  # 2: sum of squares (keepdim)
        "add",  # 3: + eps
        "square_root",  # 4: sqrt(sum_sq + eps)
        "divide",  # 5: x / denom
        "cast",  # 6: restore activation dtype
    ),
    edge_constraints=(
        (0, 1, "output_to_input"),  # x_fp32 → square
        (1, 2, "output_to_input"),  # squared → reduce_sum
        (2, 3, "output_to_first_input"),  # norm_sq → add (left)
        (3, 4, "output_to_input"),  # radicand → sqrt
        (4, 5, "output_to_second_input"),  # denom → divide (right)
        (0, 5, "output_to_first_input"),  # x_fp32 → divide (left)
        (5, 6, "output_to_input"),  # normalized → downcast
    ),
)

L2_NORMALIZE_PROVENANCE = ProvenanceMatchRule(
    id="prov-l2-normalize",
    kind="region/l2_normalize",
    priority=6,
    component_type="L2Normalize",
    require_contiguous_in_graph_order=True,
)
