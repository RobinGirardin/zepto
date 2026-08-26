"""Functional wrappers for reduction operations."""

from ..composition import GraphTensor
from ..operation import ReduceSum
from ._common import context


def reduce_sum(
    value: GraphTensor,
    axis: int | tuple[int, ...],
    *,
    keepdim: bool = False,
) -> GraphTensor:
    """Record a sum reduction along one or more axes."""
    return context().apply(ReduceSum(axis, keepdim=keepdim), value)  # type: ignore[return-value]
