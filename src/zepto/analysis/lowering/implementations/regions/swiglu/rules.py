"""Discovery rules for SwiGLU MLP stack regions."""

from __future__ import annotations

from ....region import PatternConstraint, PatternMatchRule, ProvenanceMatchRule

SWIGLU_PATTERN = PatternMatchRule(
    id="pat-swiglu-decomposed",
    kind="region/swiglu",
    priority=8,
    op_families=(
        "linear_matmul",  # 0: gate_proj
        "linear_matmul",  # 1: up_proj
        "sigmoid",        # 2: SiLU part 1 (gate branch)
        "multiply",       # 3: SiLU part 2 (x * σ(x))
        "multiply",       # 4: SiLU(g) ⊙ u
        "linear_matmul",  # 5: down_proj
    ),
    edge_constraints=(
        (0, 2, "output_to_input"),           # g → sigmoid
        (2, 3, "output_to_second_input"),    # σ(g) → SiLU multiply right
        (0, 3, "output_to_first_input"),     # g → SiLU multiply left
        (3, 4, "output_to_first_input"),     # silu(g) → gate×up left
        (1, 4, "output_to_second_input"),    # u → gate×up right
        (4, 5, "output_to_input"),           # h → down_proj
    ),
    constraints=(
        PatternConstraint(kind="swiglu_silu_branch"),
        PatternConstraint(kind="swiglu_shared_gate_up_input"),
        PatternConstraint(kind="swiglu_gate_act_mul_up"),
    ),
)

SWIGLU_PROVENANCE = ProvenanceMatchRule(
    id="prov-swiglu",
    kind="region/swiglu",
    priority=8,
    component_type="SwiGLU",
    require_contiguous_in_graph_order=True,
)
