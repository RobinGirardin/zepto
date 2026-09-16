"""Closed-form cost leaf for fused attention-sink softmax (boundary A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SoftmaxOneRecipe:
    """Fused sink-softmax costing for boundary A variants.

    See _workspace/research.md §4.2 (forward) and §5 (backward).
    """

    flop_profile: Literal["zepto", "megatron"] = "zepto"
    forward_flops_per_element: int = 5
    backward_flops_per_element: int = 4
    save_P: bool = True
    materialize_extended_logits: bool = False

    def forward_flops(self, *, num_heads: int, seq_len: int) -> int:
        n = num_heads * seq_len * seq_len
        base = self.forward_flops_per_element * n
        if self.flop_profile == "zepto":
            return base + num_heads * seq_len
        return base + self.forward_flops_per_element * num_heads * seq_len

    def backward_flops(
        self, *, num_heads: int, seq_len: int, requires_grad: bool
    ) -> int:
        if not requires_grad:
            return 0
        n_ext = num_heads * seq_len * (seq_len + 1)
        return self.backward_flops_per_element * n_ext


DEFAULT_SOFTMAX_ONE_RECIPE = SoftmaxOneRecipe(flop_profile="zepto")
MEGATRON_SOFTMAX_ONE_RECIPE = SoftmaxOneRecipe(flop_profile="megatron")
