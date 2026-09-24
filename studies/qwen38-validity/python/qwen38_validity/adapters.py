"""Qwen3.8-bound adapters over the shared measurement kernel."""

from __future__ import annotations

from qwen38_validity.protocol import MODEL_ID, TRAINING_STEPS, TWIN_MODE
from validity_common.adapters import measure_subject_precision as _measure
from validity_common.adapters import warmup_process

__all__ = ["measure_subject_precision", "warmup_process"]


def measure_subject_precision(subject, precision, **kwargs):
    return _measure(
        subject,
        precision,
        model_id=MODEL_ID,
        twin_mode=TWIN_MODE,
        training_steps=TRAINING_STEPS,
        **kwargs,
    )
