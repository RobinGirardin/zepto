"""Closed-form cost leaf for fused SwiGLU MLP stacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SwiGLURecipe:
    """Fused SwiGLU MLP stack costing for ``region/swiglu/*``.

    ``seq_len`` is the token count (``S`` or ``B·S``), not the sequence axis alone.
    Forward: 6*S*d*d_ff + 6*S*d_ff (three GEMMs + SiLU+gate×up elementwise).
    Backward: 12*S*d*d_ff + 10*S*d_ff when requires_grad else 0.
    Paper-comparable (Appendix E): 6*S*d*d_ff / 12*S*d*d_ff (GEMM-only).
    See docs/kernel/swiglu.md §4 / §5.
    """

    save_gate: bool = True
    save_up: bool = True
    save_fused_gate_up: bool = False
    save_down_input: bool = True

    def intermediate_numel(self, *, seq_len: int, intermediate_size: int) -> int:
        return seq_len * intermediate_size

    def forward_flops(
        self, *, seq_len: int, hidden_size: int, intermediate_size: int
    ) -> int:
        s, d, d_ff = seq_len, hidden_size, intermediate_size
        return 6 * s * d * d_ff + 6 * s * d_ff

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
        return 12 * s * d * d_ff + 10 * s * d_ff

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


DEFAULT_SWIGLU_RECIPE = SwiGLURecipe()

LIGER_SWIGLU_RECIPE = SwiGLURecipe(
    save_gate=True,
    save_up=True,
    save_fused_gate_up=False,
    save_down_input=True,
)

LIGER_FUSED_GATE_UP_SWIGLU_RECIPE = SwiGLURecipe(
    save_gate=False,
    save_up=False,
    save_fused_gate_up=True,
    save_down_input=True,
)
