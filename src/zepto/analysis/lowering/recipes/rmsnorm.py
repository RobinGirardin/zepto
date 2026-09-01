"""Closed-form cost leaf for fused RMSNorm."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RMSNormRecipe:
    """Fused RMSNorm costing shared by reference / liger / hub variants.

    Accounting policy: on-chip SRAM intermediates (squared, normalized, cast
    temps inside a fused launch) never produce ALLOCATE/SAVE events — only
    HBM-resident boundary outputs and explicit saved-backward tensors.

    ``materialize_rstd`` / ``save_rstd`` model hub routing differences:
    - Liger / XPU hub: write ``rstd`` to HBM, SAVE for backward.
    - MPS mlx-rmsnorm (inference): ``inv_mean`` stays in threadgroup SLM;
      Zepto omits ``rstd`` ALLOCATE/SAVE on the hub-mps variant.
    """

    forward_flops_per_element: int = 4
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
        """Saved rstd: input with last dim replaced by 1."""
        if not input_shape:
            return (1,)
        return (*input_shape[:-1], 1)


DEFAULT_RMSNORM_RECIPE = RMSNormRecipe()

# HF hub MPS: forward inference does not materialize rstd in HBM (kernel-implementation §5.2).
HUB_MPS_INFERENCE_RECIPE = RMSNormRecipe(
    materialize_rstd=False,
    save_rstd=False,
)
