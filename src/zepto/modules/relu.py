"""ReLU activation module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.functional.elementary import maximum
from ..core.functional.graph_tensors import constant_zero


class ReLU(Module):
    """Elementwise ReLU: ``max(x, 0)``.

    Lowers to ``region/relu`` when the zero operand is tagged ``constant_zero``.
    """

    module_kind = "ReLU"

    def __init__(self) -> None:
        super().__init__()
        self._zero: GraphTensor | None = None

    def forward(self, x: GraphTensor) -> GraphTensor:
        if self._zero is None:
            self._zero = constant_zero()
        return maximum(x, self._zero)
