"""Discovery rules for ReLU (maximum with zero right operand)."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule

RELU_PATTERN = PatternMatchRule(
    id="pat-relu",
    kind="region/relu",
    priority=10,
    op_families=("maximum",),
    constraints=(PatternConstraint(kind="right_operand_is_zero"),),
)
