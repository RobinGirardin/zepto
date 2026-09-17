"""SiLU (Swish) activation module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Multiply, Sigmoid


class SiLU(Module):
    """Elementwise SiLU: ``x * sigmoid(x)``.

    Lowers to ``region/silu`` when the decomposed ``Sigmoid → Multiply`` chain
    matches the fused region pattern.
    """

    module_kind = "SiLU"

    def forward(self, x: Tensor) -> Tensor:
        sig = Sigmoid()(x)
        return Multiply()(x, sig)  # type: ignore[return-value]
