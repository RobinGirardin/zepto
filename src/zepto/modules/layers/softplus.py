"""Softplus activation module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Exp, Log


class Softplus(Module):
    """Elementwise Softplus: ``log(1 + exp(x))``.

    Lowers to ``region/softplus`` when the decomposed ``Exp → Add → Log`` chain
    matches the fused region pattern.
    """

    module_kind = "Softplus"

    def __init__(self) -> None:
        super().__init__()
        self._one = Tensor(shape=(1,), semantic_type="one", requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        exp_x = Exp()(x)
        one_plus = Add()(self._one, exp_x)  # type: ignore[return-value]
        return Log()(one_plus)  # type: ignore[return-value]
