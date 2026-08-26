"""Backend-neutral dtypes, shapes, roles, and port contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from zepto.compose.values import Tensor

Dim: TypeAlias = int
Shape: TypeAlias = tuple[Dim, ...]


class DType(StrEnum):
    """Backend-neutral element type with an unambiguous byte width."""

    UNKNOWN = "unknown"
    BOOL = "bool"
    INT32 = "int32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"

    @property
    def itemsize(self) -> int | None:
        """Return the element byte width, or None for UNKNOWN."""
        return _DTYPE_ITEMSIZE[self]


_DTYPE_ITEMSIZE: dict[DType, int | None] = {
    DType.UNKNOWN: None,
    DType.BOOL: 1,
    DType.INT32: 4,
    DType.FP16: 2,
    DType.BF16: 2,
    DType.FP32: 4,
    DType.FP64: 8,
}


class TensorRole(StrEnum):
    """Storage/accounting role. Distinct from ValueKind and semantic_type."""

    INPUT = "input"
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    AUXILIARY = "auxiliary"
    STATE = "state"
    WORKSPACE = "workspace"
    GRADIENT = "gradient"


@dataclass(frozen=True, slots=True)
class PortContract:
    """Partial port specification. Unset fields and empty shape are wildcards."""

    shape: Shape | None = None
    dtype: DType | None = None
    semantic_type: str | None = None
    role: TensorRole | None = None

    def __post_init__(self) -> None:
        if self.shape is not None and any(
            not isinstance(dim, int) or dim < 0 for dim in self.shape
        ):
            raise ValueError("Tensor dimensions must be non-negative integers")
        if self.semantic_type is not None and not self.semantic_type:
            raise ValueError("Semantic type cannot be empty")
        if self.dtype is not None and not isinstance(self.dtype, DType):
            raise ValueError("dtype must be a DType when supplied")
        if self.role is not None and not isinstance(self.role, TensorRole):
            raise ValueError("role must be a TensorRole when supplied")


def contract_satisfied(contract: PortContract, tensor: Tensor) -> bool:
    """Return whether a tensor satisfies a port contract.

    Unset contract fields are wildcards. An empty shape ``()`` is also a
    wildcard. ``role`` is reserved for analysis-time validation; compose
    tensors do not carry a role, so a set role does not reject them.

    Args:
        contract: Partial specification declared on a port.
        tensor: Structural tensor offered at the port.

    Returns:
        ``True`` when ``tensor`` satisfies ``contract``.
    """
    if contract.semantic_type is not None and contract.semantic_type != tensor.semantic_type:
        return False
    if contract.shape and contract.shape != tensor.shape:
        return False
    if contract.dtype is not None and contract.dtype != tensor.dtype:
        return False
    return True
