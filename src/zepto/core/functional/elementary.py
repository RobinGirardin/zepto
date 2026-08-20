"""Functional wrappers for elementary tensor operations."""

from ..composition import GraphTensor
from ..operation import Add, Multiply, ReLU
from ._common import context


def add(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record elementwise addition of two graph tensors."""
    return context().apply(Add(), left, right)  # type: ignore[return-value]


def multiply(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record elementwise multiplication of two graph tensors."""
    return context().apply(Multiply(), left, right)  # type: ignore[return-value]


def relu(value: GraphTensor) -> GraphTensor:
    """Record a rectified-linear-unit operation."""
    return context().apply(ReLU(), value)  # type: ignore[return-value]
