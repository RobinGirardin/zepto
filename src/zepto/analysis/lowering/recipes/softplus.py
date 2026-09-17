"""Closed-form cost leaf for fused Softplus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SoftplusRecipe:
    """Fused Softplus costing for ``region/softplus`` variants.

    Forward: 9 FLOPs/element (4 exp + 1 add + 4 log).
    Backward: 7 FLOPs/element (4 exp + 1 add + 1 div + 1 mul).
    See _workspace/research.md §4.2 / §5.2 and docs/kernel-implementation.md §13.1.
    """

    forward_flops_per_element: int = 9
    backward_flops_per_element: int = 7
    save_input: bool = True
    beta: float = 1.0
    threshold: float = 20.0

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_SOFTPLUS_RECIPE = SoftplusRecipe()
XIELU_SCALAR_SOFTPLUS_RECIPE = SoftplusRecipe(save_input=False)
