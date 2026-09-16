"""Closed-form cost leaf for fused GQA with sink softmax (GPT-OSS)."""

from __future__ import annotations

from dataclasses import dataclass

from .gqa import effective_attention_pairs


@dataclass(frozen=True, slots=True)
class GQASinkRecipe:
    """Fused GQA prefill with attention-softmax-with-sink (boundary C variant).

    Forward: 4 h N d_h + 5 h N + h S (sink denominator term).
    """

    save_row_stats: bool = True
    backward_recompute_factor: int = 2
    window_size: int | None = None

    def forward_flops(self, *, num_heads: int, seq_len: int, head_dim: int) -> int:
        pairs = effective_attention_pairs(seq_len, self.window_size)
        h, s, dh = num_heads, seq_len, head_dim
        gemm = 4 * h * pairs * dh
        standard = 5 * h * pairs
        sink = h * s
        return gemm + standard + sink

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


DEFAULT_GQA_SINK_RECIPE = GQASinkRecipe()
