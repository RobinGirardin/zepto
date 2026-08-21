"""Shared helpers for operation estimation and resource descriptions."""

from ..metadata import TensorMetadata
from .records import OperationResult, ResourceEvent, ResourceEventKind


def numel(metadata: TensorMetadata) -> int:
    """Return the concrete element count.

    Args:
        metadata: Tensor metadata whose shape should be counted.

    Returns:
        The product of the tensor dimensions.
    """
    result = 1
    for dim in metadata.shape:
        result *= dim
    return result


def broadcast_metadata(
    inputs: tuple[TensorMetadata, TensorMetadata],
    *,
    family: str,
    semantic_type: str = "tensor",
) -> TensorMetadata:
    """Infer metadata for a binary elementwise broadcast."""
    left, right = inputs
    rank = max(len(left.shape), len(right.shape))
    left_shape = (1,) * (rank - len(left.shape)) + left.shape
    right_shape = (1,) * (rank - len(right.shape)) + right.shape

    shape: list[int] = []
    for left_dim, right_dim in zip(left_shape, right_shape, strict=True):
        if left_dim == right_dim:
            shape.append(left_dim)
        elif left_dim == 1:
            shape.append(right_dim)
        elif right_dim == 1:
            shape.append(left_dim)
        else:
            raise ValueError(
                f"{family} shapes are not broadcast-compatible: "
                f"{left.shape} and {right.shape}"
            )

    return TensorMetadata(
        shape=tuple(shape),
        semantic_type=semantic_type,
        requires_grad=any(value.requires_grad for value in inputs),
    )


def allocate(result: OperationResult) -> tuple[ResourceEvent, ...]:
    """Describe allocation of each materialized operation output.

    Args:
        result: Resolved operation result containing the output count.

    Returns:
        One allocation event per output, in output-port order.
    """
    return tuple(
        ResourceEvent(ResourceEventKind.ALLOCATE, f"output:{index}")
        for index in range(len(result.outputs))
    )


def alias(target: str, *, phase: str = "forward") -> ResourceEvent:
    """Describe a view alias of existing storage."""
    return ResourceEvent(ResourceEventKind.ALIAS, target, phase=phase)


def save(target: str, *, phase: str = "forward") -> ResourceEvent:
    """Describe retention of a tensor for a later backward pass."""
    return ResourceEvent(ResourceEventKind.SAVE, target, phase=phase)


def release(target: str, *, phase: str = "backward") -> ResourceEvent:
    """Describe release of a previously retained tensor."""
    return ResourceEvent(ResourceEventKind.RELEASE, target, phase=phase)
