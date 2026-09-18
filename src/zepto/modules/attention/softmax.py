"""Softmax module."""

from __future__ import annotations

import math

from zepto.compose import Module, Tensor
from zepto.semantic import Divide, Exp, Multiply, ReduceSum
from zepto.modules._internal._helpers import norm_axis


class Softmax(Module):
    """Softmax along the last dimension, optionally scaled (e.g. ``1/√d_h``).

    Decomposed eager path: ``scale → exp → reduce_sum → divide``, matching the
    Atto/HuggingFace stable-softmax structural estimate without fusion.
    """

    module_kind = "Softmax"

    def __init__(self, *, scale: float | None = None) -> None:
        super().__init__()
        self.scale = scale
        self._scale = (
            Tensor(shape=(1,), semantic_type="softmax_scale", requires_grad=False)
            if scale is not None
            else None
        )

    def forward(self, value: Tensor) -> Tensor:
        axis = norm_axis(value)
        scores = Multiply()(value, self._scale) if self._scale is not None else value
        exp_scores = Exp()(scores)
        return Divide()(
            exp_scores,
            ReduceSum(axis=axis, keepdim=True)(exp_scores),
        )  # type: ignore[return-value]


def attention_softmax_scale(head_dim: int) -> float:
    """Return the standard attention score scale ``1/√d_h``."""
    return 1.0 / math.sqrt(head_dim)
