"""Closed-form cost leaf for fused GatedRMSNorm."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatedRMSNormRecipe:
    """Fused GatedRMSNorm leaf (boundary A).

    Forward/backward: research §4.2 / §5.2 (10n / 14n); do not re-derive in code
    comments beyond cite.
    On-chip elisions: research §8 elided_temps (squared, normalized, silu intermediates).
    """

    forward_flops_per_element: int = 10
    backward_flops_per_element: int = 14
    materialize_rstd: bool = True
    save_rstd: bool = True

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel

    def rstd_shape(self, value_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Saved rstd: value with last dim 1 (fp32), same as RMSNorm."""
        if not value_shape:
            return (1,)
        return (*value_shape[:-1], 1)


DEFAULT_GATED_RMS_NORM_RECIPE = GatedRMSNormRecipe()
