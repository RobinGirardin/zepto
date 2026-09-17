"""Closed-form cost leaf for fused GatedGroupedRMSNorm."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatedGroupedRMSNormRecipe:
    """Fused GatedGroupedRMSNorm leaf (boundary A).

    Forward/backward: research §4.2 / §5.2 (10n / 14n); do not re-derive in code
    comments beyond cite.
    """

    forward_flops_per_element: int = 10
    backward_flops_per_element: int = 14
    materialize_group_rstd: bool = True
    save_group_rstd: bool = True
    include_group_scalar_forward_flops: bool = False

    def forward_flops(self, numel: int, *, num_row_groups: int = 0) -> int:
        base = self.forward_flops_per_element * numel
        if self.include_group_scalar_forward_flops:
            return base + 2 * num_row_groups
        return base

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel

    def group_rstd_shape(
        self, value_shape: tuple[int, ...], num_groups: int
    ) -> tuple[int, ...]:
        """Saved inverse-RMS: (*prefix, num_groups, 1) fp32 — not full-axis (..., 1)."""
        if not value_shape:
            return (num_groups, 1)
        return (*value_shape[:-1], num_groups, 1)


DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE = GatedGroupedRMSNormRecipe()
EAGER_HF_GATED_GROUPED_RMS_NORM_RECIPE = GatedGroupedRMSNormRecipe(
    include_group_scalar_forward_flops=True,
)
