"""Closed-form cost leaf for fused scale+mask+softmax (Megatron/TE boundary B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .softmax_one import MEGATRON_SOFTMAX_ONE_RECIPE, SoftmaxOneRecipe


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


@dataclass(frozen=True, slots=True)
class MaskedSoftmaxSinkRecipe:
    """Scale + mask + sink softmax (Megatron FusedScaleMaskSoftmax torch fallback)."""

    flop_profile: Literal["zepto", "megatron"] = "megatron"
    save_P: bool = True

    def forward_flops(self, *, num_heads: int, seq_len: int) -> int:
        n = num_heads * seq_len * seq_len
        scale_mask = 2 * n
        sink_softmax = SoftmaxOneRecipe(
            flop_profile=self.flop_profile
        ).forward_flops(num_heads=num_heads, seq_len=seq_len)
        return scale_mask + sink_softmax

    def backward_flops(
        self, *, num_heads: int, seq_len: int, requires_grad: bool
    ) -> int:
        if not requires_grad:
            return 0
        return MEGATRON_SOFTMAX_ONE_RECIPE.backward_flops(
            num_heads=num_heads, seq_len=seq_len, requires_grad=True
        )


DEFAULT_MASKED_SOFTMAX_SINK_RECIPE = MaskedSoftmaxSinkRecipe()
