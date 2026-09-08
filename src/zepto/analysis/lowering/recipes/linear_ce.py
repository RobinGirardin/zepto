"""Closed-form cost leaf for fused linear + cross-entropy (Liger FLCE)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinearCERecipe:
    """Fused LM-head linear + vocab softmax + CE costing.

    Forward FLOPs match ``docs/kernel-implementation.md`` §14:
    ``2*S*d*V + 3*S*V + 2*S``. Peak logits VRAM uses Liger chunk formula with
    ``chunk_mem_const`` (default 16). On-chip online softmax temps are elided.
    """

    chunk_mem_const: int = 16
    forward_softmax_flops_per_element: int = 3
    backward_softmax_flops_per_element: int = 4
    materialize_full_logits: bool = False
    save_logits: bool = False
    save_hidden: bool = True

    def forward_flops(self, s: int, d: int, v: int) -> int:
        return (
            2 * s * d * v
            + self.forward_softmax_flops_per_element * s * v
            + 2 * s
        )

    def backward_flops(self, s: int, d: int, v: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return 6 * s * d * v + 7 * s * v

    def chunk_size(self, s: int, d: int, v: int) -> int:
        """Liger token-chunk sizing (``CHUNK_MEM_CONST`` tiling)."""
        c = self.chunk_mem_const
        inc_factor = (v + c * d - 1) // (c * d)
        target = (s + inc_factor - 1) // inc_factor
        chunk = 1
        while chunk < target:
            chunk *= 2
        return min(chunk, s)

    def peak_logits_bytes(self, s: int, d: int, v: int, elem_bytes: int) -> int:
        full = s * v * elem_bytes
        chunked = self.chunk_size(s, d, v) * v * elem_bytes
        budget = self.chunk_mem_const * s * d * elem_bytes
        return min(full, chunked, budget)


DEFAULT_LINEAR_CE_RECIPE = LinearCERecipe()
