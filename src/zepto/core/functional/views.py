"""Functional wrappers for identity and view operations."""

from ..composition import GraphTensor
from ..operation import Identity, Reshape, Split, Transpose
from ._common import context


def identity(value: GraphTensor) -> GraphTensor:
    """Record an identity operation for a graph tensor."""
    return context().apply(Identity(), value)  # type: ignore[return-value]


def reshape(value: GraphTensor, shape: tuple[int, ...]) -> GraphTensor:
    """Record a reshape view with the requested shape."""
    return context().apply(Reshape(shape), value)  # type: ignore[return-value]


def transpose(value: GraphTensor, permutation: tuple[int, ...]) -> GraphTensor:
    """Record a transpose view using a dimension permutation."""
    return context().apply(Transpose(permutation), value)  # type: ignore[return-value]


def split(value: GraphTensor, sizes: tuple[int, ...]) -> tuple[GraphTensor, ...]:
    """Record a split and return its outputs in declaration order."""
    result = context().apply(Split(sizes), value)
    return result if isinstance(result, tuple) else (result,)
