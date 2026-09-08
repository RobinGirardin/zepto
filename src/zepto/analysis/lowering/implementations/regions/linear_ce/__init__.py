"""Fused linear + cross-entropy region implementations."""

from .rules import LINEAR_CE_PATTERN, LINEAR_CE_PROVENANCE
from .variants import (
    FUSED_LINEAR_CE_HUB_TRL,
    FUSED_LINEAR_CE_LIGER,
    FUSED_LINEAR_CE_REFERENCE,
    LINEAR_CE_REGIONS,
)

__all__ = [
    "FUSED_LINEAR_CE_HUB_TRL",
    "FUSED_LINEAR_CE_LIGER",
    "FUSED_LINEAR_CE_REFERENCE",
    "LINEAR_CE_PATTERN",
    "LINEAR_CE_PROVENANCE",
    "LINEAR_CE_REGIONS",
]
