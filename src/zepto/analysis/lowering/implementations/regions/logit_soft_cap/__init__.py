"""Fused logit soft-cap region implementations."""

from .rules import LOGIT_SOFT_CAP_PATTERN, LOGIT_SOFT_CAP_PROVENANCE
from .variants import (
    FUSED_LOGIT_SOFT_CAP_REFERENCE,
    LOGIT_SOFT_CAP_REGIONS,
)

__all__ = [
    "FUSED_LOGIT_SOFT_CAP_REFERENCE",
    "LOGIT_SOFT_CAP_PATTERN",
    "LOGIT_SOFT_CAP_PROVENANCE",
    "LOGIT_SOFT_CAP_REGIONS",
]
