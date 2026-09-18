"""Fused GQA-sink (GPT-OSS sink softmax) region implementations."""

from .rules import GQA_SINK_OP_SEQUENCES, GQA_SINK_PATTERN
from .variants import (
    FUSED_GQA_SINK_FLASH2,
    FUSED_GQA_SINK_FLASH3,
    GQA_SINK_REGIONS,
)

__all__ = [
    "FUSED_GQA_SINK_FLASH2",
    "FUSED_GQA_SINK_FLASH3",
    "GQA_SINK_OP_SEQUENCES",
    "GQA_SINK_PATTERN",
    "GQA_SINK_REGIONS",
]
