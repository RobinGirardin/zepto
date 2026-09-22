"""Concat helpers for compose-time sequence assembly."""

from __future__ import annotations

from zepto.compose import Tensor
from zepto.semantic import Concat, Split, Transpose


def _sequence_perm(rank: int) -> tuple[int, ...]:
    if rank == 2:
        return (1, 0)
    if rank == 3:
        return (1, 0, 2)
    if rank == 4:
        return (1, 0, 2, 3)
    raise ValueError(f"unsupported rank for sequence split: {rank}")


def split_on_sequence(tensor: Tensor, parts: int) -> tuple[Tensor, ...]:
    """Split on sequence axis: 0 for rank-2, 1 for rank-3+."""
    rank = len(tensor.shape)
    if rank == 2:
        return split_leading(tensor, parts)
    perm = _sequence_perm(rank)
    moved = Transpose(permutation=perm)(tensor)
    split_parts = split_leading(moved, parts)
    return tuple(Transpose(permutation=perm)(part) for part in split_parts)


def concat_on_sequence(tensors: tuple[Tensor, ...]) -> Tensor:
    """Concat along sequence axis (0 for rank-2, 1 for rank-3+)."""
    if not tensors:
        raise ValueError("concat_on_sequence requires at least one tensor")
    rank = len(tensors[0].shape)
    if rank == 2:
        return concat_leading(tensors)
    perm = _sequence_perm(rank)
    moved = tuple(Transpose(permutation=perm)(t) for t in tensors)
    merged = concat_leading(moved)
    return Transpose(permutation=perm)(merged)  # type: ignore[return-value]


def split_last_channels(
    value: Tensor, sizes: tuple[int, ...]
) -> tuple[Tensor, ...]:
    """Split the last dimension via leading Split (rank-2 or rank-3 ``(B,S,C)``)."""
    rank = len(value.shape)
    if rank == 2:
        transposed = Transpose(permutation=(1, 0))(value)
        parts = Split(sizes=sizes)(transposed)
        if isinstance(parts, Tensor):
            parts = (parts,)
        return tuple(Transpose(permutation=(1, 0))(part) for part in parts)
    if rank == 3:
        transposed = Transpose(permutation=(2, 0, 1))(value)
        parts = Split(sizes=sizes)(transposed)
        if isinstance(parts, Tensor):
            parts = (parts,)
        return tuple(Transpose(permutation=(1, 2, 0))(part) for part in parts)
    raise ValueError(
        f"split_last_channels expects rank-2 or rank-3, got {value.shape}"
    )


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
