"""Closed-form cost leaf for fused Mamba-2 selective SSM scan (boundary C)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Mamba2ScanRecipe:
    """Fused selective SSM scan leaf (boundary C).

    Forward/backward closed forms: _workspace/research.md §4.1 / §5.
    """

    materialize_per_step_temps: bool = False
    save_state_all_timesteps: bool = True
    save_scan_state_out: bool = True
    state_dtype_bytes: int = 4  # fp32 H accumulation (research §0)

    def forward_flops(
        self,
        *,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
    ) -> int:
        s, h, p, n = seq_len, num_heads, head_dim, state_size
        return s * h * (5 * p * n + n + 2 * p + 7)

    def backward_flops(
        self,
        *,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        requires_grad: bool,
    ) -> int:
        if not requires_grad:
            return 0
        s, h, p, n = seq_len, num_heads, head_dim, state_size
        return s * h * (10 * p * n + 2 * p + n + 4)

    def saved_state_bytes(
        self,
        *,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
    ) -> int:
        if not self.save_state_all_timesteps:
            return 0
        return (
            self.state_dtype_bytes * seq_len * num_heads * head_dim * state_size
        )


DEFAULT_MAMBA2_SCAN_RECIPE = Mamba2ScanRecipe()

DECODE_MAMBA2_SCAN_RECIPE = DEFAULT_MAMBA2_SCAN_RECIPE
