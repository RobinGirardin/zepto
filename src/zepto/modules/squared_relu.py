"""Squared ReLU (ReLU²) activation module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Maximum, Multiply


class SquaredReLU(Module):
    """Elementwise ReLU²: ``Multiply(Maximum(x, 0), Maximum(x, 0))``.

    Unfused activation path bills **1 FLOP/element** (multiply only); the
    ``Maximum`` comparison is not billed as arithmetic.
    """

    module_kind = "SquaredReLU"

    def __init__(self) -> None:
        super().__init__()
        self._zero = Tensor(
            shape=(1,), semantic_type="constant_zero", requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        relu_out = Maximum()(x, self._zero)  # type: ignore[call-arg]
        return Multiply()(relu_out, relu_out)  # type: ignore[return-value]


__all__ = ["SquaredReLU"]
