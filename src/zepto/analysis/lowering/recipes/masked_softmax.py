"""Closed-form cost leaf for fused scale+mask+softmax (Megatron/TE boundary B)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MaskedSoftmaxRecipe:
    """Fused scale+mask+softmax costing shared by reference / megatron / te variants.

    Forward: 5 FLOPs/element (scale + mask + exp + sum + div billing leaf).
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


DEFAULT_MASKED_SOFTMAX_RECIPE = MaskedSoftmaxRecipe()
