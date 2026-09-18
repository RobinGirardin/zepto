"""Discovery rules for GeGLU MLP stack regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

_GEGLU_EDGE_CONSTRAINTS = (
    (0, 2, "output_to_input"),
    (2, 3, "output_to_first_input"),
    (1, 3, "output_to_second_input"),
    (3, 4, "output_to_input"),
)

_GEGLU_CONSTRAINTS = (
    PatternConstraint(kind="geglu_shared_gate_up_input"),
    PatternConstraint(kind="geglu_gelu_on_gate_branch"),
    PatternConstraint(kind="geglu_gate_act_mul_up"),
)

GEGLU_PATTERN = PatternMatchRule(
    id="pat-geglu-decomposed",
    kind="region/geglu",
    priority=8,
    op_families=(
        "linear_matmul",  # 0: gate_proj
        "linear_matmul",  # 1: up_proj
        "gelu",           # 2: GeluTanhGate (atomic; not nine-op GELUTanh chain)
        "multiply",       # 3: gelu(g) ⊙ u
        "linear_matmul",  # 4: down_proj
    ),
    edge_constraints=_GEGLU_EDGE_CONSTRAINTS,
    constraints=_GEGLU_CONSTRAINTS,
)

GEGLU_ERF_PATTERN = PatternMatchRule(
    id="pat-geglu-decomposed-erf",
    kind="region/geglu",
    priority=8,
    op_families=(
        "linear_matmul",
        "linear_matmul",
        "gelu_erf",       # GeluErfGate
        "multiply",
        "linear_matmul",
    ),
    edge_constraints=_GEGLU_EDGE_CONSTRAINTS,
    constraints=_GEGLU_CONSTRAINTS,
)

GEGLU_PROVENANCE = ProvenanceMatchRule(
    id="prov-geglu",
    kind="region/geglu",
    priority=8,
    component_type="GeGLU",
    require_contiguous_in_graph_order=True,
)
