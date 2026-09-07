"""Closed-form cost leaf for fused xIELU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class XIELURecipe:
    """Fused xIELU costing shared by reference / cuda variants.

    Forward: 8 FLOPs/element (neg-branch upper bound).
    Backward: 10 FLOPs/element (8 Jacobian + 2 parameter MAC).
    See docs/kernel-implementation.md §13 and Atto ``55 - xielu.md``.
    """

    forward_flops_per_element: int = 8
    backward_flops_per_element: int = 10
    save_sign_mask: bool = True

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_XIELU_RECIPE = XIELURecipe()
