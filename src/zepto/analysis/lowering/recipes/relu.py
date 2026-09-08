"""Closed-form cost leaf for fused ReLU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReLURecipe:
    """Fused ReLU costing for ``region/relu``.

    Forward: 1 FLOP/element (``Maximum`` primitive; comparison not billed).
    Backward: 1 FLOP/element (grad × bool mask).
    See ``docs/kernel/relu.md`` and research §4/§5.
    """

    forward_flops_per_element: int = 1
    backward_flops_per_element: int = 1
    save_relu_mask: bool = True
    save_input: bool = False

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_RELU_RECIPE = ReLURecipe()
SAVED_INPUT_RELU_RECIPE = ReLURecipe(save_relu_mask=False, save_input=True)
