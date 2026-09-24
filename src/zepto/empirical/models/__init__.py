"""Registered model families for empirical studies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zepto.empirical.models.base import ModelFamily


def get_model_family(model_id: str) -> ModelFamily:
    if model_id == "apertus":
        from zepto.empirical.models.apertus import ApertusFamily

        return ApertusFamily()
    if model_id == "apertus15":
        from zepto.empirical.models.apertus15 import Apertus15Family

        return Apertus15Family()
    if model_id == "granite":
        from zepto.empirical.models.granite import GraniteFamily

        return GraniteFamily()
    if model_id == "qwen38":
        from zepto.empirical.models.qwen38 import Qwen38Family

        return Qwen38Family()
    raise KeyError(f"unknown model_id: {model_id!r}")
