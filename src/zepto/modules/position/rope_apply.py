"""Rotary positional-encoding application module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Concat, Multiply, Split, Transpose


class RoPEApply(Module):
    """Apply RoPE to headed activations using ``cos``/``sin`` caches.

    Supports rank-3 ``(h, S, d_h)`` and rank-4 ``(B, h, S, d_h)`` inputs.
    When ``rotary_dim < head_dim``, only the prefix channels are rotated.
    """

    module_kind = "RoPEApply"

    def __init__(
        self,
        head_dim: int,
        *,
        rotary_dim: int | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        resolved = rotary_dim if rotary_dim is not None else head_dim
        if resolved <= 0 or resolved % 2 != 0:
            raise ValueError(f"rotary_dim must be a positive even integer, got {resolved}")
        if resolved > head_dim:
            raise ValueError(
                f"rotary_dim ({resolved}) must not exceed head_dim ({head_dim})"
            )
        self.head_dim = head_dim
        self.rotary_dim = resolved
        self._neg_one = Tensor(shape=(1,), semantic_type="neg_one", requires_grad=False)

    def _rotate_half(self, value: Tensor, *, rotary_dim: int) -> Tensor:
        """Apply HuggingFace ``rotate_half``: ``cat(-x2, x1)`` on the head dimension."""
        half = rotary_dim // 2
        rank = len(value.shape)
        if rank == 3:
            permuted = Transpose(permutation=(2, 0, 1))(value)
            first, second = Split(sizes=(half, half))(permuted)  # type: ignore[misc]
            neg_second = Multiply()(second, self._neg_one)
            first_back = Transpose(permutation=(1, 2, 0))(first)
            neg_second_back = Transpose(permutation=(1, 2, 0))(neg_second)
            return Concat(axis=2, input_count=2)(neg_second_back, first_back)  # type: ignore[return-value]
        if rank == 4:
            permuted = Transpose(permutation=(3, 0, 1, 2))(value)
            first, second = Split(sizes=(half, half))(permuted)  # type: ignore[misc]
            neg_second = Multiply()(second, self._neg_one)
            first_back = Transpose(permutation=(1, 2, 3, 0))(first)
            neg_second_back = Transpose(permutation=(1, 2, 3, 0))(neg_second)
            return Concat(axis=3, input_count=2)(neg_second_back, first_back)  # type: ignore[return-value]
        raise ValueError(f"_rotate_half expects rank 3 or 4, got rank {rank}")

    def _apply_full(
        self,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        rotated = self._rotate_half(value, rotary_dim=self.rotary_dim)
        return Add()(Multiply()(value, cos), Multiply()(rotated, sin))  # type: ignore[return-value]

    def _apply_partial(
        self,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        rank = len(value.shape)
        if rank == 3:
            permuted = Transpose(permutation=(2, 0, 1))(value)
            prefix, suffix = Split(
                sizes=(self.rotary_dim, self.head_dim - self.rotary_dim)
            )(permuted)  # type: ignore[misc]
            prefix_heads = Transpose(permutation=(1, 2, 0))(prefix)
            suffix_heads = Transpose(permutation=(1, 2, 0))(suffix)
            rotated_prefix = self._rotate_half(prefix_heads, rotary_dim=self.rotary_dim)
            prefix_out = Add()(
                Multiply()(prefix_heads, cos),
                Multiply()(rotated_prefix, sin),
            )
            return Concat(axis=2, input_count=2)(prefix_out, suffix_heads)  # type: ignore[return-value]
        permuted = Transpose(permutation=(3, 0, 1, 2))(value)
        prefix, suffix = Split(
            sizes=(self.rotary_dim, self.head_dim - self.rotary_dim)
        )(permuted)  # type: ignore[misc]
        prefix_heads = Transpose(permutation=(1, 2, 3, 0))(prefix)
        suffix_heads = Transpose(permutation=(1, 2, 3, 0))(suffix)
        rotated_prefix = self._rotate_half(prefix_heads, rotary_dim=self.rotary_dim)
        prefix_out = Add()(
            Multiply()(prefix_heads, cos),
            Multiply()(rotated_prefix, sin),
        )
        return Concat(axis=3, input_count=2)(prefix_out, suffix_heads)  # type: ignore[return-value]

    def forward(
        self,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        rank = len(value.shape)
        if rank not in (3, 4):
            raise ValueError(
                f"RoPEApply expects headed input (h, S, d_h) or (B, h, S, d_h), "
                f"got {value.shape}"
            )
        d_h = value.shape[-1]
        seq_len = value.shape[-2]
        if d_h != self.head_dim:
            raise ValueError(
                f"RoPEApply head_dim mismatch: configured {self.head_dim}, "
                f"input last dim {d_h}"
            )
        expected_cos_shape = (seq_len, self.rotary_dim)
        if cos.shape != expected_cos_shape or sin.shape != expected_cos_shape:
            raise ValueError(
                f"RoPEApply expects cos/sin shape {expected_cos_shape}, "
                f"got cos={cos.shape} sin={sin.shape}"
            )
        if self.rotary_dim == self.head_dim:
            return self._apply_full(value, cos, sin)
        return self._apply_partial(value, cos, sin)


class RoPE(RoPEApply):
    """Backward-compatible alias for :class:`RoPEApply`."""

    module_kind = "RoPEApply"
