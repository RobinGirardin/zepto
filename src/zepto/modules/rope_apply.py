"""Rotary positional-encoding application module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.functional import add, concat, multiply, split, transpose
from ..core.functional import scalar_input


class RoPEApply(Module):
    """Apply RoPE to headed activations ``(h, S, d_h)`` using ``cos``/``sin`` caches.

    Matches HuggingFace eager: ``out = x * cos + rotate_half(x) * sin``.
    """

    module_kind = "RoPEApply"

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        self.head_dim = head_dim
        self._neg_one = scalar_input(semantic_type="neg_one")

    def _rotate_half(self, value: GraphTensor) -> GraphTensor:
        """Apply HuggingFace ``rotate_half``: ``cat(-x2, x1)`` on the head dimension."""
        half = self.head_dim // 2
        permuted = transpose(value, (2, 0, 1))
        first, second = split(permuted, (half, half))
        neg_second = multiply(second, self._neg_one)
        first_back = transpose(first, (1, 2, 0))
        neg_second_back = transpose(neg_second, (1, 2, 0))
        return concat(neg_second_back, first_back, axis=2)

    def forward(
        self,
        value: GraphTensor,
        cos: GraphTensor,
        sin: GraphTensor,
    ) -> GraphTensor:
        if len(value.metadata.shape) != 3:
            raise ValueError(
                f"RoPEApply expects headed input (h, S, d_h), got {value.metadata.shape}"
            )
        _h, seq_len, d_h = value.metadata.shape
        if d_h != self.head_dim:
            raise ValueError(
                f"RoPEApply head_dim mismatch: configured {self.head_dim}, "
                f"input last dim {d_h}"
            )
        if cos.metadata.shape != (seq_len, d_h) or sin.metadata.shape != (seq_len, d_h):
            raise ValueError(
                f"RoPEApply expects cos/sin shape {(seq_len, d_h)}, "
                f"got cos={cos.metadata.shape} sin={sin.metadata.shape}"
            )
        rotated = self._rotate_half(value)
        return add(multiply(value, cos), multiply(rotated, sin))


class RoPE(RoPEApply):
    """Backward-compatible alias for :class:`RoPEApply`."""

    module_kind = "RoPEApply"
