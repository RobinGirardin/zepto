"""Closed-form cost leaf for fused GQA prefill attention (FA2/FA3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GQARecipe:
    """Fused GQA prefill attention (boundary C).

    Forward: 4 h S² d_h + 5 h S² (kernel-accurate online softmax).
    Backward: recompute path ≈ 2× forward when row stats are saved.
    See docs/kernel-implementation.md §11 and docs/kernel/flash-attention3.md.
    """

    save_row_stats: bool = True
    backward_recompute_factor: int = 2

    def forward_flops(self, *, num_heads: int, seq_len: int, head_dim: int) -> int:
        h, s, dh = num_heads, seq_len, head_dim
        return 4 * h * s * s * dh + 5 * h * s * s

    def backward_flops(
        self,
        *,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        requires_grad: bool,
    ) -> int:
        if not requires_grad:
            return 0
        return self.backward_recompute_factor * self.forward_flops(
            num_heads=num_heads, seq_len=seq_len, head_dim=head_dim
        )


DEFAULT_GQA_RECIPE = GQARecipe()


@dataclass(frozen=True, slots=True)
class GQASDPAMathRecipe:
    """SDPA math sub-backend: materialized P, eager FLOP class.

    Forward: 4 h S² d_h + 6 h S² (includes mask add).
    Backward: standard GEMM + softmax VJP on saved P.
    See docs/kernel/scaled-dot-product-attention.md.
    """

    save_P: bool = True

    def forward_flops(self, *, num_heads: int, seq_len: int, head_dim: int) -> int:
        h, s, dh = num_heads, seq_len, head_dim
        return 4 * h * s * s * dh + 6 * h * s * s

    def backward_flops(
        self,
        *,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        requires_grad: bool,
    ) -> int:
        if not requires_grad:
            return 0
        h, s, dh = num_heads, seq_len, head_dim
        return 8 * h * s * s * dh + 10 * h * s * s


DEFAULT_SDPA_MATH_RECIPE = GQASDPAMathRecipe()
