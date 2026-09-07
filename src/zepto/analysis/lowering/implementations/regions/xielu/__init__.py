"""Fused xIELU region implementations."""

from .rules import XIELU_PATTERN, XIELU_PROVENANCE
from .variants import (
    FUSED_XIELU_CUDA,
    FUSED_XIELU_REFERENCE,
    XIELU_REGIONS,
)

__all__ = [
    "FUSED_XIELU_CUDA",
    "FUSED_XIELU_REFERENCE",
    "XIELU_PATTERN",
    "XIELU_PROVENANCE",
    "XIELU_REGIONS",
]
