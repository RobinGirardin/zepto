"""Fused GQA (FlashAttention + SDPA) region implementations."""

from .rules import GQA_OP_SEQUENCES, GQA_PATTERN
from .variants import (
    FUSED_GQA_FLASH2,
    FUSED_GQA_FLASH3,
    FUSED_GQA_SDPA_FLASH,
    FUSED_GQA_SDPA_MATH,
    FUSED_GQA_SDPA_MEM_EFFICIENT,
    GQA_REGIONS,
)

__all__ = [
    "FUSED_GQA_FLASH2",
    "FUSED_GQA_FLASH3",
    "FUSED_GQA_SDPA_FLASH",
    "FUSED_GQA_SDPA_MATH",
    "FUSED_GQA_SDPA_MEM_EFFICIENT",
    "GQA_OP_SEQUENCES",
    "GQA_PATTERN",
    "GQA_REGIONS",
]
