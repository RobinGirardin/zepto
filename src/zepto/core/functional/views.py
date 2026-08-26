"""Functional wrappers for identity and view operations."""

from ..composition import GraphTensor
from ..operation import Concat, Gather, Identity, RepeatKV, Reshape, Split, Transpose
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


def concat(*values: GraphTensor, axis: int = 0) -> GraphTensor:
    """Record concatenation of two or more tensors along one axis."""
    if len(values) < 2:
        raise ValueError("concat requires at least two input tensors")
    return context().apply(Concat(axis, len(values)), *values)  # type: ignore[return-value]


def gather(input: GraphTensor, index: GraphTensor, *, axis: int = 0) -> GraphTensor:
    """Record index selection along one axis."""
    return context().apply(Gather(axis), input, index)  # type: ignore[return-value]


def repeat_kv(
    value: GraphTensor,
    n_rep: int,
    *,
    axis: int = 0,
) -> GraphTensor:
    """Record KV-head replication along one axis."""
    return context().apply(RepeatKV(n_rep, axis=axis), value)  # type: ignore[return-value]
