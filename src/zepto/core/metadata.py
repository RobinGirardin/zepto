from dataclasses import dataclass
from typing import TypeAlias

from .ids import DimensionScope


@dataclass(frozen=True, slots=True)
class SymbolicDim:
    """Named dimension whose concrete size is resolved later by a context."""

    name: str
    scope: DimensionScope

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Symbolic dimension name cannot be empty")


Dim: TypeAlias = int | SymbolicDim
Shape: TypeAlias = tuple[Dim, ...]


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Backend-neutral shape and semantic metadata for a graph value."""

    shape: Shape
    semantic_type: str = "tensor"

    def __post_init__(self) -> None:
        if any(isinstance(dim, int) and dim < 0 for dim in self.shape):
            raise ValueError("Tensor dimensions cannot be negative")
        if not self.semantic_type:
            raise ValueError("Semantic type cannot be empty")


def metadata_compatible(
    expected: TensorMetadata,
    actual: TensorMetadata,
) -> bool:
    """Return whether actual metadata satisfies an expected contract.

    Concrete dimensions must match exactly. A symbolic expected dimension may
    match either the same symbolic identity or a concrete resolved dimension.
    This permits a later estimation context to resolve a symbolic shape while
    preserving strictness for declared concrete dimensions and semantic types.

    Args:
        expected: Metadata declared by a port.
        actual: Metadata inferred or supplied by a graph value.

    Returns:
        ``True`` when ``actual`` satisfies ``expected``.
    """
    if expected.semantic_type != actual.semantic_type:
        return False
    if len(expected.shape) != len(actual.shape):
        return False

    for expected_dim, actual_dim in zip(expected.shape, actual.shape, strict=True):
        if isinstance(expected_dim, int):
            if expected_dim != actual_dim:
                return False
        elif isinstance(actual_dim, SymbolicDim) and expected_dim != actual_dim:
            return False

    return True
