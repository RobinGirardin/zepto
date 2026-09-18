"""Concat helpers for compose-time sequence assembly."""

from __future__ import annotations

from zepto.compose import Tensor
from zepto.semantic import Concat


def split_leading(tensor: Tensor, parts: int) -> tuple[Tensor, ...]:
    """Split on leading dimension; always return a tuple (Split may return one tensor)."""
    from zepto.semantic import Split

    if parts <= 0:
        raise ValueError("parts must be positive")
    if parts == 1:
        return (tensor,)
    result = Split(sizes=(1,) * parts)(tensor)
    if isinstance(result, Tensor):
        return (result,)
    return result


def concat_leading(tensors: tuple[Tensor, ...]) -> Tensor:
    """Concatenate rank-2+ tensors along axis 0 (or return the sole tensor)."""
    if not tensors:
        raise ValueError("concat_leading requires at least one tensor")
    if len(tensors) == 1:
        return tensors[0]
    return Concat(axis=0, input_count=len(tensors))(*tensors)  # type: ignore[return-value]
