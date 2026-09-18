"""Closed-form cost leaf for fused linear + soft-cap + cross-entropy (Liger)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinearCESoftcapRecipe:
    """Fused LM-head linear + tanh soft-cap + vocab softmax + CE (Liger softcap path)."""

    chunk_mem_const: int = 16
    forward_softmax_flops_per_element: int = 3
    forward_cap_flops_per_element: int = 7
    backward_vocab_flops_per_element: int = 19
    materialize_full_logits: bool = False
    save_logits: bool = False
    save_hidden: bool = True

    def forward_flops(self, s: int, d: int, v: int) -> int:
        return (
            2 * s * d * v
            + (self.forward_softmax_flops_per_element + self.forward_cap_flops_per_element)
            * s
            * v
            + 2 * s
        )

    def backward_flops(self, s: int, d: int, v: int, *, requires_grad: bool) -> int:
        if not requires_grad:
            return 0
        return 6 * s * d * v + self.backward_vocab_flops_per_element * s * v

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


DEFAULT_LINEAR_CE_SOFTCAP_RECIPE = LinearCESoftcapRecipe()
