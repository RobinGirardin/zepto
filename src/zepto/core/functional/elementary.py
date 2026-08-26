"""Functional wrappers for elementary tensor operations."""

from ..composition import GraphTensor
from ..operation import (
    Add,
    Cos,
    Divide,
    Exp,
    Log,
    Maximum,
    Minimum,
    Multiply,
    Pow,
    Sin,
    SquareRoot,
    Subtract,
    Where,
)
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


def pow(base: GraphTensor, exponent: GraphTensor) -> GraphTensor:
    """Record an elementwise power operation."""
    return context().apply(Pow(), base, exponent)  # type: ignore[return-value]


def exp(value: GraphTensor) -> GraphTensor:
    """Record an elementwise natural exponential."""
    return context().apply(Exp(), value)  # type: ignore[return-value]


def log(value: GraphTensor) -> GraphTensor:
    """Record an elementwise natural logarithm."""
    return context().apply(Log(), value)  # type: ignore[return-value]


def sin(value: GraphTensor) -> GraphTensor:
    """Record an elementwise sine."""
    return context().apply(Sin(), value)  # type: ignore[return-value]


def cos(value: GraphTensor) -> GraphTensor:
    """Record an elementwise cosine."""
    return context().apply(Cos(), value)  # type: ignore[return-value]


def where(
    condition: GraphTensor,
    on_true: GraphTensor,
    on_false: GraphTensor,
) -> GraphTensor:
    """Record elementwise conditional selection."""
    return context().apply(Where(), condition, on_true, on_false)  # type: ignore[return-value]
