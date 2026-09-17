"""Closed-form cost leaf for fused GELU variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeluRecipe:
    """Fused GELU costing shared by tanh / erf / quick variants.

    See docs/kernel/gelu.md §4/§5 and docs/kernel-implementation.md §GELU.
    """

    forward_flops_per_element: int
    backward_flops_per_element: int
    save_input: bool = True
    approximate: str = "tanh"  # "tanh" | "none" | "quick"

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


GELU_TANH_RECIPE = GeluRecipe(
    forward_flops_per_element=12,
    backward_flops_per_element=20,
    approximate="tanh",
)
GELU_ERF_RECIPE = GeluRecipe(
    forward_flops_per_element=8,
    backward_flops_per_element=17,
    approximate="none",
)
GELU_QUICK_RECIPE = GeluRecipe(
    forward_flops_per_element=6,
    backward_flops_per_element=9,
    approximate="quick",
)
