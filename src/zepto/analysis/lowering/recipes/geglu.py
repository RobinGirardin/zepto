"""Closed-form cost leaf for fused GeGLU MLP stacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeGLURecipe:
    """Fused GeGLU MLP stack costing for ``region/geglu/*``.

    ``seq_len`` is the token count (``S`` or ``B·S``), not the sequence axis alone.
    Forward: 6*S*d*d_ff + forward_flops_per_element * S * d_ff
    Backward: 12*S*d*d_ff + backward_flops_per_element * S * d_ff if requires_grad else 0
    Paper-comparable: 6*S*d*d_ff / 12*S*d*d_ff (GEMM-only; Appendix E).
    See docs/kernel/geglu.md §4 / §5.
    """

    gelu_approx: str  # "tanh" | "none"
    forward_flops_per_element: int  # gelu_fwd + 1 mul: 13 tanh, 9 erf
    backward_flops_per_element: int  # gelu_bwd + 2 mul: 22 tanh, 19 erf
    save_gate: bool = True
    save_up: bool = True
    save_down_input: bool = True
    elide_gelu_temp: bool = False  # True for liger (elides gelu_gate_output)

    def forward_flops(
        self, *, seq_len: int, hidden_size: int, intermediate_size: int
    ) -> int:
        s, d, d_ff = seq_len, hidden_size, intermediate_size
        return 6 * s * d * d_ff + self.forward_flops_per_element * s * d_ff

    def backward_flops(
        self,
        *,
        seq_len: int,
        hidden_size: int,
        intermediate_size: int,
        requires_grad: bool,
    ) -> int:
        if not requires_grad:
            return 0
        s, d, d_ff = seq_len, hidden_size, intermediate_size
        return 12 * s * d * d_ff + self.backward_flops_per_element * s * d_ff

    def paper_forward_flops(
        self, *, seq_len: int, hidden_size: int, intermediate_size: int
    ) -> int:
        s, d, d_ff = seq_len, hidden_size, intermediate_size
        return 6 * s * d * d_ff

    def paper_backward_flops(
        self, *, seq_len: int, hidden_size: int, intermediate_size: int
    ) -> int:
        s, d, d_ff = seq_len, hidden_size, intermediate_size
        return 12 * s * d * d_ff


DEFAULT_GEGLU_RECIPE = GeGLURecipe(
    gelu_approx="tanh",
    forward_flops_per_element=13,
    backward_flops_per_element=22,
)

GEGLU_ERF_RECIPE = GeGLURecipe(
    gelu_approx="none",
    forward_flops_per_element=9,
    backward_flops_per_element=19,
)

LIGER_GEGLU_RECIPE = GeGLURecipe(
    gelu_approx="tanh",
    forward_flops_per_element=13,
    backward_flops_per_element=22,
    elide_gelu_temp=True,
)
