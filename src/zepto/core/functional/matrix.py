"""Functional wrappers for matrix operations."""

from ..composition import GraphTensor
from ..operation import MatMul
from ._common import context


def matmul(left: GraphTensor, right: GraphTensor) -> GraphTensor:
    """Record rank-two matrix multiplication of two graph tensors."""
    return context().apply(MatMul(), left, right)  # type: ignore[return-value]
