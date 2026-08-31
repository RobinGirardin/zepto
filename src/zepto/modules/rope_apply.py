"""Rotary positional-encoding application module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Concat, Multiply, Split, Transpose


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
        self._neg_one = Tensor(shape=(1,), semantic_type="neg_one", requires_grad=False)

    def _rotate_half(self, value: Tensor) -> Tensor:
        """Apply HuggingFace ``rotate_half``: ``cat(-x2, x1)`` on the head dimension."""
        half = self.head_dim // 2
        permuted = Transpose(permutation=(2, 0, 1))(value)
        first, second = Split(sizes=(half, half))(permuted)  # type: ignore[misc]
        neg_second = Multiply()(second, self._neg_one)
        first_back = Transpose(permutation=(1, 2, 0))(first)
        neg_second_back = Transpose(permutation=(1, 2, 0))(neg_second)
        return Concat(axis=2, input_count=2)(neg_second_back, first_back)  # type: ignore[return-value]

    def forward(
        self,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        if len(value.shape) != 3:
            raise ValueError(
                f"RoPEApply expects headed input (h, S, d_h), got {value.shape}"
            )
        _h, seq_len, d_h = value.shape
        if d_h != self.head_dim:
            raise ValueError(
                f"RoPEApply head_dim mismatch: configured {self.head_dim}, "
                f"input last dim {d_h}"
            )
        if cos.shape != (seq_len, d_h) or sin.shape != (seq_len, d_h):
            raise ValueError(
                f"RoPEApply expects cos/sin shape {(seq_len, d_h)}, "
                f"got cos={cos.shape} sin={sin.shape}"
            )
        rotated = self._rotate_half(value)
        return Add()(Multiply()(value, cos), Multiply()(rotated, sin))  # type: ignore[return-value]


class RoPE(RoPEApply):
    """Backward-compatible alias for :class:`RoPEApply`."""

    module_kind = "RoPEApply"
