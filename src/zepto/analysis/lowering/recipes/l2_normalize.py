"""Closed-form cost leaf for fused L2-normalize (last dim, no γ)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class L2NormalizeRecipe:
    """Fused L2-normalize leaf (un-averaged sum-of-squares, no γ).

    On-chip temps (squared, norm_sq, denom, x_fp32 cast scratch) never
    emit ALLOCATE/SAVE — only boundary ``y`` and saved ``rstd``.

    Elided temps (documentation only): ``squared``, ``normalized``, ``norm_sq``,
    ``denom``, ``x_fp32_cast_temp``.
    """

    forward_flops_per_element: int = 3
    backward_flops_per_element: int = 4
    materialize_rstd: bool = True
    save_rstd: bool = True

    def forward_flops(self, numel: int) -> int:
        return self.forward_flops_per_element * numel

    def backward_flops(self, numel: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return self.backward_flops_per_element * numel

    def rstd_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if not input_shape:
            return (1,)
        return (*input_shape[:-1], 1)


DEFAULT_L2_NORMALIZE_RECIPE = L2NormalizeRecipe()
