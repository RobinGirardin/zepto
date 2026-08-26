"""Functional helpers for declaring new graph tensors in the active graph."""

from __future__ import annotations

from ..composition import GraphTensor
from ..metadata import ValueMetadata
from ._common import context


def scalar_input(*, semantic_type: str = "constant") -> GraphTensor:
    """Declare a broadcastable rank-1 scalar graph input."""
    return context().input(
        ValueMetadata((1,), semantic_type=semantic_type, requires_grad=False)
    )


def graph_input(metadata: ValueMetadata) -> GraphTensor:
    """Declare a graph input with arbitrary metadata."""
    return context().input(metadata)


def constant_zero() -> GraphTensor:
    """Declare a scalar zero input (ReLU / region matching)."""
    return scalar_input(semantic_type="constant_zero")
