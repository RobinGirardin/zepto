"""Fused Softplus region implementation."""

from .rules import SOFTPLUS_PATTERN, SOFTPLUS_PROVENANCE
from .variants import (
    FUSED_SOFTPLUS,
    FUSED_SOFTPLUS_SCALAR,
    SOFTPLUS_REGIONS,
)

__all__ = [
    "FUSED_SOFTPLUS",
    "FUSED_SOFTPLUS_SCALAR",
    "SOFTPLUS_PATTERN",
    "SOFTPLUS_PROVENANCE",
    "SOFTPLUS_REGIONS",
]
