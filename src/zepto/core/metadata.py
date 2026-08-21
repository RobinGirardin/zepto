"""Backend-neutral tensor shape and semantic metadata."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


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
class TensorMetadata:
    """Backend-neutral shape and semantic metadata for a graph value."""

    shape: Shape
    semantic_type: str = "tensor"
    requires_grad: bool = False
    dtype: DType | None = None
    role: TensorRole | None = None
    persistent: bool = False

    def __post_init__(self) -> None:
        """Reject non-concrete or negative dimensions and empty semantic types."""
        if any(not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ValueError("Tensor dimensions must be non-negative integers")
        if not self.semantic_type:
            raise ValueError("Semantic type cannot be empty")
        if self.dtype is not None and not isinstance(self.dtype, DType):
            raise ValueError("dtype must be a DType when supplied")
        if self.role is not None and not isinstance(self.role, TensorRole):
            raise ValueError("role must be a TensorRole when supplied")
        if not isinstance(self.persistent, bool):
            raise ValueError("persistent must be an explicit boolean")
        if not isinstance(self.requires_grad, bool):
            raise ValueError("requires_grad must be an explicit boolean")


def metadata_compatible(
    expected: TensorMetadata,
    actual: TensorMetadata,
) -> bool:
    """Return whether actual metadata satisfies an expected contract.

    Concrete dimensions and semantic types must match exactly. Unspecified
    ``dtype`` and ``role`` on the expected side are wildcards.

    Args:
        expected: Metadata declared by a port.
        actual: Metadata inferred or supplied by a graph value.

    Returns:
        ``True`` when ``actual`` satisfies ``expected``.
    """
    if expected.semantic_type != actual.semantic_type:
        return False
    if expected.shape != actual.shape:
        return False
    if expected.dtype is not None and expected.dtype != actual.dtype:
        return False
    if expected.role is not None and expected.role != actual.role:
        return False
    return True
