"""Fused GQA (FlashAttention) region implementations."""

from .rules import GQA_OP_SEQUENCES, GQA_PATTERN
from .variants import (
    FUSED_GQA_FLASH2,
    FUSED_GQA_FLASH3,
    GQA_REGIONS,
)

__all__ = [
    "FUSED_GQA_FLASH2",
    "FUSED_GQA_FLASH3",
    "GQA_OP_SEQUENCES",
    "GQA_PATTERN",
    "GQA_REGIONS",
]
