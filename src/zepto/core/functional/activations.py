"""Functional wrappers for activation modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..composition import GraphTensor

if TYPE_CHECKING:
    from ...modules.relu import ReLU

_RELU: ReLU | None = None


def relu(x: GraphTensor) -> GraphTensor:
    """Apply ReLU via the :class:`~zepto.modules.relu.ReLU` module."""
    global _RELU
    if _RELU is None:
        from ...modules.relu import ReLU

        _RELU = ReLU()
    return _RELU(x)
