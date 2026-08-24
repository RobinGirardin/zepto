"""Functional wrappers for matrix operations."""

from ..composition import GraphParameter, GraphTensor
from ..operation import LinearMatMul, MatMul
from ._common import context


def matmul(
    left: GraphTensor,
    right: GraphTensor | GraphParameter,
) -> GraphTensor:
    """Record matrix multiplication of two tensors or activation × weight."""
    if isinstance(right, GraphParameter):
        return context().apply(LinearMatMul(), left, parameters=(right,))  # type: ignore[return-value]
    return context().apply(MatMul(), left, right)  # type: ignore[return-value]


def linear_matmul(input: GraphTensor, weight: GraphParameter) -> GraphTensor:
    """Record activation × weight parameter multiplication."""
    return context().apply(LinearMatMul(), input, parameters=(weight,))  # type: ignore[return-value]
