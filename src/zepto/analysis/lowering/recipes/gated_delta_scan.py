"""Closed-form cost leaf for fused gated delta-rule scan (boundary C)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GatedDeltaScanRecipe:
    """Fused gated delta scan leaf (boundary C).

    Forward/backward closed forms: research §4.1 / §5 (Zepto identity grouping).
    Atto paper-comparable 9/10 grouping is documentation-only in derivations/.
    """

    materialize_per_step_temps: bool = False
    save_state_all_timesteps: bool = True
    save_scan_state_out: bool = True
    state_dtype_bytes: int = 4  # fp32 H accumulation

    def forward_flops(
        self, *, seq_len: int, num_heads: int, key_dim: int, value_dim: int
    ) -> int:
        s, h, dk, dv = seq_len, num_heads, key_dim, value_dim
        return s * h * (8 * dk * dv + 2 * dv + 1)

    def backward_flops(
        self,
        *,
        seq_len: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        requires_grad: bool,
    ) -> int:
        if not requires_grad:
            return 0
        s, h, dk, dv = seq_len, num_heads, key_dim, value_dim
        return s * h * (16 * dk * dv + 4 * dv + 4)

    def saved_state_bytes(
        self, *, seq_len: int, num_heads: int, key_dim: int, value_dim: int
    ) -> int:
        if not self.save_state_all_timesteps:
            return 0
        return (
            self.state_dtype_bytes * seq_len * num_heads * key_dim * value_dim
        )


DEFAULT_GATED_DELTA_SCAN_RECIPE = GatedDeltaScanRecipe()

DECODE_GATED_DELTA_SCAN_RECIPE = DEFAULT_GATED_DELTA_SCAN_RECIPE
