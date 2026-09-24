"""Granite architecture options (no torch / heavy Zepto imports)."""

from __future__ import annotations

from zepto.empirical.models.gqa_dense_core import (
    OPTION_KEYS,
    GqaDenseFamilyCore,
    round_to_multiple,
)


class GraniteFamilyCore(GqaDenseFamilyCore):
    model_id = "granite"


__all__ = [
    "OPTION_KEYS",
    "GraniteFamilyCore",
    "round_to_multiple",
]
