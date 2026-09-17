"""Closed-form cost leaf for fused GQA prefill attention (FA2/FA3)."""

from __future__ import annotations

from dataclasses import dataclass


def effective_attention_pairs(seq_len: int, window_size: int | None) -> int:
    """Count of (query, key) pairs after causal + sliding band."""
    s = seq_len
    if window_size is None or window_size >= s:
        return s * s
    w = window_size
    if s <= w:
        return s * (s + 1) // 2
    return s * w - w * (w - 1) // 2


def gqa_forward_flops(
    *,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    window_size: int | None = None,
    softmax_extra_per_pair: float = 0.0,
) -> int:
    """Forward FLOPs for fused GQA: QK + PV GEMMs plus online softmax."""
    pairs = effective_attention_pairs(seq_len, window_size)
    h, dh = num_heads, head_dim
    gemm = 4 * h * pairs * dh
    softmax = 5 * h * pairs + int(softmax_extra_per_pair)
    return gemm + softmax


@dataclass(frozen=True, slots=True)
class GQARecipe:
    """Fused GQA prefill attention (boundary C).

    Forward: 4 h N d_h + 5 h N (kernel-accurate online softmax), N = effective pairs.
    Backward: recompute path ≈ 2× forward when row stats are saved.
    See docs/kernel-implementation.md §11 and docs/kernel/flash-attention3.md.
    """

    save_row_stats: bool = True
    backward_recompute_factor: int = 2
    window_size: int | None = None

    def forward_flops(self, *, num_heads: int, seq_len: int, head_dim: int) -> int:
        return gqa_forward_flops(
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            window_size=self.window_size,
        )

    def paged_forward_flops(
        self, *, num_heads: int, cache_len: int, head_dim: int
    ) -> int:
        """Decode-step FLOPs with S_q=1 attending over cache_len."""
        h, s, dh = num_heads, cache_len, head_dim
        return 4 * h * s * dh + 5 * h * s

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

    Forward: 4 h N d_h + 5 h N + h S² (includes full-buffer mask add when sliding).
    Backward: standard GEMM + softmax VJP on saved P.
    See docs/kernel/scaled-dot-product-attention.md.
    """

    save_P: bool = True
    window_size: int | None = None

    def forward_flops(self, *, num_heads: int, seq_len: int, head_dim: int) -> int:
        h, s, dh = num_heads, seq_len, head_dim
        pairs = effective_attention_pairs(s, self.window_size)
        gemm = 4 * h * pairs * dh
        softmax = 5 * h * pairs
        mask_add = h * s * s if self.window_size is not None else h * pairs
        return gemm + softmax + mask_add

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
        pairs = effective_attention_pairs(s, self.window_size)
        return 8 * h * pairs * dh + 10 * h * pairs


DEFAULT_SDPA_MATH_RECIPE = GQASDPAMathRecipe()
