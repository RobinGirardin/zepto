"""Closed-form cost leaf for fused ReLU²."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SquaredReLURecipe:
    """Fused ReLU² costing for ``region/squared_relu``.

    Forward: 2 FLOPs/element (max + mul; comparison not billed).
    Backward: 3 FLOPs/element (recompute ReLU from saved input + 2 muls).
    See _workspace/research.md §4.2 / §5.2.
    """

    forward_flops_per_element: int = 2
    backward_flops_per_element: int = 3
    save_input: bool = True
    save_relu_intermediate: bool = False

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_SQUARED_RELU_RECIPE = SquaredReLURecipe()
