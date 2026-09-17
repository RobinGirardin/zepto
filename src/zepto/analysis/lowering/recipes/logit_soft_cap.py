"""Closed-form cost leaf for fused logit soft-cap."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogitSoftCapRecipe:
    """Fused ``scale * cap * tanh(logits / cap)`` leaf (identity decomposition)."""

    forward_flops_per_element: int = 7
    backward_flops_per_element: int = 8

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel


DEFAULT_LOGIT_SOFT_CAP_RECIPE = LogitSoftCapRecipe()
