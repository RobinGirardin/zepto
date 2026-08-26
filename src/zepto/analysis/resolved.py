"""Analysis-time pairing of a structural tensor with resolved accounting."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose.values import Tensor
from zepto.semantic.metadata import DType, TensorRole


@dataclass(frozen=True, slots=True)
class ResolvedValue:
    """Ephemeral accounting view of a structural tensor."""

    tensor: Tensor
    role: TensorRole
    dtype: DType
