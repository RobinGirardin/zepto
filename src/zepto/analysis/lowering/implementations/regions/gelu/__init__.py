"""Fused GELU region implementations."""

from .rules import (
    GELU_ERF_PATTERN,
    GELU_ERF_PROVENANCE,
    GELU_TANH_PATTERN,
    GELU_TANH_PROVENANCE,
)
from .variants import FUSED_GELU_ERF, FUSED_GELU_TANH, GELU_REGIONS

__all__ = [
    "FUSED_GELU_ERF",
    "FUSED_GELU_TANH",
    "GELU_ERF_PATTERN",
    "GELU_ERF_PROVENANCE",
    "GELU_REGIONS",
    "GELU_TANH_PATTERN",
    "GELU_TANH_PROVENANCE",
]
