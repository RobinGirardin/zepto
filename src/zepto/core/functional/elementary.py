"""Functional wrappers for elementary tensor operations."""

from ..composition import GraphTensor
from ..operation import Add, Divide, Maximum, Minimum, Multiply, SquareRoot, Subtract
from ._common import context


def add(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record elementwise addition of two graph tensors."""
    return context().apply(Add(), left, right)  # type: ignore[return-value]


def subtract(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record elementwise subtraction of two graph tensors."""
    return context().apply(Subtract(), left, right)  # type: ignore[return-value]


def multiply(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record elementwise multiplication of two graph tensors."""
    return context().apply(Multiply(), left, right)  # type: ignore[return-value]


def divide(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record elementwise division of two graph tensors."""
    return context().apply(Divide(), left, right)  # type: ignore[return-value]


def maximum(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record an elementwise maximum of two graph tensors."""
    return context().apply(Maximum(), left, right)  # type: ignore[return-value]


def minimum(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record an elementwise minimum of two graph tensors."""
    return context().apply(Minimum(), left, right)  # type: ignore[return-value]


def sqrt(value: GraphTensor) -> GraphTensor:
    """Record an elementwise square-root operation."""
    return context().apply(SquareRoot(), value)  # type: ignore[return-value]
