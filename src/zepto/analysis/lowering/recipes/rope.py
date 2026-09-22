"""Closed-form FLOP recipes for RoPE materialize and apply (docs/kernel-implementation.md §8)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoPEMaterializeRecipe:
    """Build shared ``cos``/``sin`` caches once per forward (model scope).

    Forward: ``S * (d/2)`` for ``position_ids @ inv_freq`` (outer product).
    ``concat``, ``cos``, and ``sin`` bill **0** FLOPs (Zepto / ``FlopCounterMode``).
    Backward: 0 (cache treated as forward-only for costing; llama3 ``inv_freq`` init
    is outside the scored graph).
    """

    def forward_flops(self, *, seq_len: int, head_dim: int) -> int:
        return seq_len * (head_dim // 2)

    def backward_flops(self, *, requires_grad: bool) -> int:
        del requires_grad
        return 0


@dataclass(frozen=True, slots=True)
class RoPEApplyRecipe:
    """Apply RoPE to one headed tensor (Q or K).

    Forward and backward: ``6 * num_heads * seq_len * head_dim`` when training
    (inverse rotation is the same 6 FLOPs/pair as forward; see Atto rope §2).
    """

    def forward_flops(
        self, *, num_heads: int, seq_len: int, head_dim: int
    ) -> int:
        return 6 * num_heads * seq_len * head_dim

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
        return self.forward_flops(
            num_heads=num_heads, seq_len=seq_len, head_dim=head_dim
        )


DEFAULT_ROPE_MATERIALIZE_RECIPE = RoPEMaterializeRecipe()
DEFAULT_ROPE_APPLY_RECIPE = RoPEApplyRecipe()
