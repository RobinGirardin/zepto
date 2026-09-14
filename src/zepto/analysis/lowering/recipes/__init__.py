"""Closed-form cost recipes for fused region implementations."""

from .gqa import (
    DEFAULT_GQA_RECIPE,
    DEFAULT_SDPA_MATH_RECIPE,
    GQARecipe,
    GQASDPAMathRecipe,
)
from .linear_ce import DEFAULT_LINEAR_CE_RECIPE, LinearCERecipe
from .masked_softmax import DEFAULT_MASKED_SOFTMAX_RECIPE, MaskedSoftmaxRecipe
from .rmsnorm import RMSNormRecipe
from .softmax import DEFAULT_SOFTMAX_RECIPE, SoftmaxRecipe
from .relu import DEFAULT_RELU_RECIPE, SAVED_INPUT_RELU_RECIPE, ReLURecipe
from .silu import DEFAULT_SILU_RECIPE, SiLURecipe
from .xielu import XIELURecipe

__all__ = [
    "DEFAULT_RELU_RECIPE",
    "SAVED_INPUT_RELU_RECIPE",
    "DEFAULT_GQA_RECIPE",
    "DEFAULT_LINEAR_CE_RECIPE",
    "LinearCERecipe",
    "DEFAULT_MASKED_SOFTMAX_RECIPE",
    "DEFAULT_SDPA_MATH_RECIPE",
    "DEFAULT_SOFTMAX_RECIPE",
    "GQARecipe",
    "GQASDPAMathRecipe",
    "MaskedSoftmaxRecipe",
    "DEFAULT_SILU_RECIPE",
    "ReLURecipe",
    "RMSNormRecipe",
    "SiLURecipe",
    "SoftmaxRecipe",
    "XIELURecipe",
]
