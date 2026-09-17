"""ReLU activation module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Maximum


class ReLU(Module):
    """Elementwise ReLU: ``max(x, 0)``.

    Lowers to ``maximum/relu-mask`` when the zero operand is tagged ``constant_zero``.
    """

    module_kind = "ReLU"

    def __init__(self) -> None:
        super().__init__()
        self._zero = Tensor(
            shape=(1,), semantic_type="constant_zero", requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return Maximum()(x, self._zero)  # type: ignore[return-value]
