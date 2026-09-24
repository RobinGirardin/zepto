"""Locked protocol constants shared by every validity family.

Family packages keep MODEL_ID, TWIN_MODE, and OPTIONAL_LARGE_VOCAB.
"""

from __future__ import annotations

TRAINING_STEPS = 1  # K = 1; extra steps are not extra n
MIN_SEQ_LEN = 32

PHASES: tuple[str, ...] = ("inference", "training")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")

EQUIVALENCE_MARGIN = 0.10
ALPHA = 0.05
# 0.95 ** (1/8) so the conjunction of eight TOSTs targets joint power 0.95.
PER_COMBINATION_POWER = 0.9936
ASSUMED_TRUE_MEAN = 0.0
