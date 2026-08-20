"""Backend-neutral tensor shape and semantic metadata."""

from dataclasses import dataclass
from typing import TypeAlias


Dim: TypeAlias = int
Shape: TypeAlias = tuple[Dim, ...]


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Backend-neutral shape and semantic metadata for a graph value."""

    shape: Shape
    semantic_type: str = "tensor"
    require_grad: bool = False

    def __post_init__(self) -> None:
        """Reject non-concrete or negative dimensions and empty semantic types."""
        if any(not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise ValueError("Tensor dimensions must be non-negative integers")
        if not self.semantic_type:
            raise ValueError("Semantic type cannot be empty")


def metadata_compatible(
    expected: TensorMetadata,
    actual: TensorMetadata,
) -> bool:
    """Return whether actual metadata satisfies an expected contract.

    Concrete dimensions and semantic types must match exactly.

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
        if expected_dim != actual_dim:
            return False

    return True
