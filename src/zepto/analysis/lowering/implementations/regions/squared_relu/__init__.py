"""Fused ReLU² region implementation."""

from .rules import SQUARED_RELU_PATTERN
from .variants import FUSED_SQUARED_RELU_REGION, SQUARED_RELU_REGIONS

__all__ = [
    "FUSED_SQUARED_RELU_REGION",
    "SQUARED_RELU_PATTERN",
    "SQUARED_RELU_REGIONS",
]
