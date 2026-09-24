"""Locked study constants. Framework §§1–2, 4, 6, 7.5."""

from __future__ import annotations

from validity_common.protocol_defaults import (  # re-export
    ALPHA,
    ASSUMED_TRUE_MEAN,
    EQUIVALENCE_MARGIN,
    MIN_SEQ_LEN,
    PER_COMBINATION_POWER,
    PHASES,
    PRECISIONS,
    TRAINING_STEPS,
)

MODEL_ID = "granite"
TWIN_MODE = "scored_window"
OPTIONAL_LARGE_VOCAB = 100352  # Granite 4.2 30B vocab; off until a pilot locks it
