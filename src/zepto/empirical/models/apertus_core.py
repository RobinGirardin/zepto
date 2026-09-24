"""Apertus architecture options (no torch / heavy Zepto imports)."""

from __future__ import annotations

from zepto.empirical.models.gqa_dense_core import (
    OPTION_KEYS,
    GqaDenseFamilyCore,
    round_to_multiple,
)


class ApertusFamilyCore(GqaDenseFamilyCore):
    model_id = "apertus"


__all__ = [
    "OPTION_KEYS",
    "ApertusFamilyCore",
    "round_to_multiple",
]
