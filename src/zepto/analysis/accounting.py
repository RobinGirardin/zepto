"""Precision and byte accounting policies for resource events."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose.values import Tensor
from zepto.semantic.metadata import DType, Shape, TensorRole

_BYTES_TO_DTYPE: dict[int, DType] = {
    2: DType.FP16,
    4: DType.FP32,
    8: DType.FP64,
}

from .resolved import ResolvedValue


def numel(shape: Shape) -> int:
    """Return the element count for a concrete shape."""
    result = 1
    for dim in shape:
        result *= dim
    return result


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Default dtypes with role- and semantic-type-specific overrides."""

    default_dtype: DType
    by_role: tuple[tuple[TensorRole, DType], ...] = ()
    by_semantic_type: tuple[tuple[str, DType], ...] = ()

    def __post_init__(self) -> None:
        if self.default_dtype is DType.UNKNOWN:
            raise ValueError("Precision policy default cannot be UNKNOWN")
        role_names = [role for role, _ in self.by_role]
        if len(role_names) != len(set(role_names)):
            raise ValueError("Precision policy by_role keys must be unique")
        semantic_types = [name for name, _ in self.by_semantic_type]
        if len(semantic_types) != len(set(semantic_types)):
            raise ValueError(
                "Precision policy by_semantic_type keys must be unique"
            )

    def resolve(self, tensor: Tensor, *, role: TensorRole) -> DType:
        """Resolve precedence: explicit dtype > semantic-type > role > default."""
        if tensor.dtype is not None:
            return tensor.dtype
        for semantic_type, dtype in self.by_semantic_type:
            if semantic_type == tensor.semantic_type:
                return dtype
        for policy_role, dtype in self.by_role:
            if policy_role == role:
                return dtype
        return self.default_dtype

    @classmethod
    def from_byte_sizes(cls, *, param_bytes: int, grad_bytes: int) -> PrecisionPolicy:
        """Build a mixed-precision policy from parameter and gradient widths."""
        if param_bytes not in _BYTES_TO_DTYPE:
            raise ValueError(f"unsupported param_bytes: {param_bytes}")
        if grad_bytes not in _BYTES_TO_DTYPE:
            raise ValueError(f"unsupported grad_bytes: {grad_bytes}")
        param_dtype = _BYTES_TO_DTYPE[param_bytes]
        grad_dtype = _BYTES_TO_DTYPE[grad_bytes]
        return cls(
            default_dtype=param_dtype,
            by_semantic_type=(("weight", param_dtype),),
            by_role=(
                (TensorRole.GRADIENT, grad_dtype),
                (TensorRole.WORKSPACE, grad_dtype),
            ),
        )


@dataclass(frozen=True, slots=True)
class AccountingPolicy:
    """Resolve dtype and byte counts for structural tensors."""

    precision: PrecisionPolicy

    def resolve_dtype(self, tensor: Tensor, *, role: TensorRole) -> DType:
        return self.precision.resolve(tensor, role=role)

    def bytes_for(self, resolved: ResolvedValue) -> int:
        itemsize = resolved.dtype.itemsize
        if itemsize is None:
            raise ValueError(
                f"Cannot account bytes for unresolved dtype {resolved.dtype!r}"
            )
        return numel(resolved.tensor.shape) * itemsize
