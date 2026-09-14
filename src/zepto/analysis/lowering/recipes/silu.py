"""Closed-form cost leaf for fused SiLU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SiLURecipe:
    """Fused SiLU costing for ``region/silu``.

    Forward: 5 FLOPs/element (4 sigmoid special-function + 1 mul).
    Backward: 8 FLOPs/element (4 sigmoid recompute + 3 fused VJP + 1 grad mul).
    See research §4.2 / §5.2 and docs/kernel-implementation.md §13 cross-ref.
    """

    forward_flops_per_element: int = 5
    backward_flops_per_element: int = 8
    save_input: bool = True

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_SILU_RECIPE = SiLURecipe()
