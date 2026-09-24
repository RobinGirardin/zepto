"""Registered model families for empirical studies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zepto.empirical.models.base import ModelFamily


def get_model_family(model_id: str) -> ModelFamily:
    if model_id == "apertus":
        from zepto.empirical.models.apertus import ApertusFamily

        return ApertusFamily()
    raise KeyError(f"unknown model_id: {model_id!r}")
