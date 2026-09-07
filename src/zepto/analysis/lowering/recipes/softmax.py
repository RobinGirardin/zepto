"""Closed-form cost leaf for fused stable softmax (aten::softmax)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SoftmaxRecipe:
    """Fused stable softmax costing shared by reference / cuda variants.

    Forward: 5 FLOPs/element (max + sub + exp + sum + div).
    Backward: 4 FLOPs/element (VJP with saved P).
    See docs/kernel-implementation.md §10 and _workspace/research.md §4.
    """

    forward_flops_per_element: int = 5
    backward_flops_per_element: int = 4
    save_P: bool = True

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_SOFTMAX_RECIPE = SoftmaxRecipe()
