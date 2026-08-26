"""Softmax module."""

from __future__ import annotations

import math

from ..core.composition import GraphTensor, Module
from ..core.functional import divide, exp, multiply, reduce_sum, scalar_input
from ._helpers import norm_axis


class Softmax(Module):
    """Softmax along the last dimension, optionally scaled (e.g. ``1/√d_h``).

    Decomposed eager path: ``scale → exp → reduce_sum → divide``, matching the
    Atto/HuggingFace stable-softmax structural estimate without fusion.
    """

    module_kind = "Softmax"

    def __init__(self, *, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale
        self._scale = scalar_input(semantic_type="softmax_scale") if scale is not None else None

    def forward(self, value: GraphTensor) -> GraphTensor:
        axis = norm_axis(value)
        scores = (
            multiply(value, self._scale)
            if self._scale is not None
            else value
        )
        exp_scores = exp(scores)
        return divide(
            exp_scores,
            reduce_sum(exp_scores, axis=axis, keepdim=True),
        )


def attention_softmax_scale(head_dim: int) -> float:
    """Return the standard attention score scale ``1/√d_h``."""
    return 1.0 / math.sqrt(head_dim)
