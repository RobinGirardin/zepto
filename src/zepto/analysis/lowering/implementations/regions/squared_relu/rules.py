"""Discovery rules for decomposed ReLU² regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule

SQUARED_RELU_PATTERN = PatternMatchRule(
    id="pat-squared-relu-decomposed",
    kind="region/squared_relu",
    priority=10,
    op_families=("maximum", "multiply"),
    edge_constraints=(
        (0, 1, "output_to_first_input"),
    ),
    constraints=(
        PatternConstraint(kind="right_operand_is_zero"),
        PatternConstraint(kind="squared_relu_mul_relu_relu"),
    ),
)
