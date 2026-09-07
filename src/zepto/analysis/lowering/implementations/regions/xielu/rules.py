"""Discovery rules for decomposed xIELU regions."""

from __future__ import annotations

from ....region import PatternMatchRule, ProvenanceMatchRule

# Matches the 13-op chain emitted by ``src/zepto/modules/xielu.py``:
#   greater_than → positive branch (×3, +) → negative branch
#   (minimum, exp, ×2 subtract, ×2 multiply, +) → where
XIELU_PATTERN = PatternMatchRule(
    id="pat-xielu-decomposed",
    kind="region/xielu",
    priority=5,
    op_families=(
        "greater_than",
        "multiply",
        "multiply",
        "multiply",
        "add",
        "minimum",
        "exp",
        "subtract",
        "subtract",
        "multiply",
        "multiply",
        "add",
        "where",
    ),
)

XIELU_PROVENANCE = ProvenanceMatchRule(
    id="prov-xielu",
    kind="region/xielu",
    priority=5,
    component_type="XIELU",
    require_contiguous_in_graph_order=True,
)
