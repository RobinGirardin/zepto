"""Precision and byte accounting policies for resource events."""

from dataclasses import dataclass

from .metadata import DType, Shape, ValueMetadata, TensorRole


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

    def resolve(self, metadata: ValueMetadata) -> DType:
        """Resolve precedence: explicit dtype > semantic-type > role > default."""
        if metadata.dtype is not None:
            return metadata.dtype
        for semantic_type, dtype in self.by_semantic_type:
            if semantic_type == metadata.semantic_type:
                return dtype
        if metadata.role is not None:
            for role, dtype in self.by_role:
                if role == metadata.role:
                    return dtype
        return self.default_dtype


@dataclass(frozen=True, slots=True)
class AccountingPolicy:
    """Resolve dtype and byte counts for tensor metadata."""

    precision: PrecisionPolicy

    def resolve_dtype(self, metadata: ValueMetadata) -> DType:
        return self.precision.resolve(metadata)

    def bytes_for(self, metadata: ValueMetadata) -> int:
        dtype = self.resolve_dtype(metadata)
        itemsize = dtype.itemsize
        if itemsize is None:
            raise ValueError(
                f"Cannot account bytes for unresolved dtype {dtype!r}"
            )
        return numel(metadata.shape) * itemsize
