"""Shared helpers for operation estimation and resource descriptions."""

from __future__ import annotations

from collections.abc import Callable

from zepto.compose.values import Tensor
from .records import OperationResult, ResourceEvent, ResourceEventKind

GRAD_LEFT = "grad_left"
GRAD_RIGHT = "grad_right"
GRAD_LEFT_UNREDUCED = "grad_left_unreduced"
GRAD_RIGHT_UNREDUCED = "grad_right_unreduced"


def numel(tensor: Tensor) -> int:
    """Return the concrete element count."""
    result = 1
    for dim in tensor.shape:
        result *= dim
    return result


def broadcast_tensor(
    inputs: tuple[Tensor, Tensor],
    *,
    family: str,
    semantic_type: str = "tensor",
) -> Tensor:
    """Infer a tensor for a binary elementwise broadcast."""
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

    return Tensor(
        shape=tuple(shape),
        semantic_type=semantic_type,
        requires_grad=any(value.requires_grad for value in inputs),
    )


def broadcast_tensor_n(
    inputs: tuple[Tensor, ...],
    *,
    family: str,
    semantic_type: str = "tensor",
) -> Tensor:
    """Infer a tensor for an n-ary elementwise broadcast."""
    if not inputs:
        raise ValueError(f"{family} requires at least one tensor to broadcast")
    result = inputs[0]
    for other in inputs[1:]:
        result = broadcast_tensor(
            (result, other),
            family=family,
            semantic_type=semantic_type,
        )
    return result


def broadcast_reduction_flops(operand: Tensor, output: Tensor) -> int:
    """Return FLOPs to sum-reduce a broadcast operand gradient."""
    return numel(output) - numel(operand)


def reduced_gradient_tensor(operand: Tensor) -> Tensor:
    """Return a structural copy for a reduced operand gradient."""
    return operand.as_unregistered()


def unreduced_gradient_tensor(output: Tensor) -> Tensor:
    """Return a structural copy for an output-shaped unreduced VJP temporary."""
    return output.as_unregistered()


def unreduced_gradient_tensor_matmul(output: Tensor, operand: Tensor) -> Tensor:
    """Return a tensor for a batched unreduced MatMul VJP temporary."""
    shape = (*output.shape[:-2], *operand.shape[-2:])
    return Tensor(
        shape=shape,
        dtype=operand.dtype,
        semantic_type=operand.semantic_type,
        requires_grad=True,
    )


def _operand_was_broadcast(operand: Tensor, output: Tensor) -> bool:
    return operand.shape != output.shape


def _needs_reduced_grad_port(
    operand: Tensor,
    other: Tensor,
    output: Tensor,
) -> bool:
    """Return whether a reduced gradient port is live for one operand."""
    if not operand.requires_grad:
        return False
    if _operand_was_broadcast(operand, output):
        return True
    return other.shape == output.shape


def active_binary_auxiliary_ports(
    left: Tensor,
    right: Tensor,
    output: Tensor,
    *,
    materializes_vjp: bool,
) -> tuple[str, ...]:
    """Select live binary auxiliary ports for one invocation."""
    active: list[str] = []
    if materializes_vjp:
        if _needs_reduced_grad_port(left, right, output):
            active.append(GRAD_LEFT)
        if _needs_reduced_grad_port(right, left, output):
            active.append(GRAD_RIGHT)
        if left.requires_grad and _operand_was_broadcast(left, output):
            active.append(GRAD_LEFT_UNREDUCED)
        if right.requires_grad and _operand_was_broadcast(right, output):
            active.append(GRAD_RIGHT_UNREDUCED)
    else:
        if left.requires_grad and _operand_was_broadcast(left, output):
            active.append(GRAD_LEFT)
        if right.requires_grad and _operand_was_broadcast(right, output):
            active.append(GRAD_RIGHT)
    return tuple(active)


def active_matmul_auxiliary_ports(
    left: Tensor,
    right: Tensor,
    output: Tensor,
) -> tuple[str, ...]:
    """Select live MatMul auxiliary ports for one invocation."""
    active: list[str] = []
    if left.requires_grad:
        active.append(GRAD_LEFT)
    if right.requires_grad:
        active.append(GRAD_RIGHT)
    if left.requires_grad and left.shape[:-2] != output.shape[:-2]:
        active.append(GRAD_LEFT_UNREDUCED)
    if right.requires_grad and right.shape[:-2] != output.shape[:-2]:
        active.append(GRAD_RIGHT_UNREDUCED)
    return tuple(active)


def allocate(output_count: int) -> tuple[ResourceEvent, ...]:
    """Describe allocation of each materialized operation output."""
    return tuple(
        ResourceEvent(ResourceEventKind.ALLOCATE, f"output:{index}")
        for index in range(output_count)
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


def activation_grad_events(port_name: str) -> tuple[ResourceEvent, ...]:
    """Backward input activation grad: allocate only; simulator schedules release."""
    return (
        ResourceEvent(ResourceEventKind.ALLOCATE, port_name, phase="backward"),
    )


def unreduced_grad_events(unreduced_port: str) -> tuple[ResourceEvent, ...]:
    """Transient workspace: allocate and release within the same backward node."""
    return (
        ResourceEvent(ResourceEventKind.ALLOCATE, unreduced_port, phase="backward"),
        ResourceEvent(ResourceEventKind.RELEASE, unreduced_port, phase="backward"),
    )


def weight_grad_accum_events(port_name: str) -> tuple[ResourceEvent, ...]:
    """Parameter gradient accumulator: survives across backward nodes / horizon steps."""
    return (
        ResourceEvent(ResourceEventKind.ALLOCATE, port_name, phase="backward"),
        ResourceEvent(ResourceEventKind.PERSIST, port_name, phase="backward"),
    )


def backward_gradient_port_events(
    port_name: str,
    *,
    unreduced_port: str | None,
    is_parameter: bool = False,
) -> tuple[ResourceEvent, ...]:
    """Emit backward auxiliary resource events with correct peak semantics."""
    events: list[ResourceEvent] = []
    if unreduced_port is not None:
        events.extend(unreduced_grad_events(unreduced_port))
    if is_parameter:
        events.extend(weight_grad_accum_events(port_name))
    else:
        events.extend(activation_grad_events(port_name))
    return tuple(events)


def emit_active_auxiliary_events(
    result: OperationResult,
    event_builder: Callable[[str], tuple[ResourceEvent, ...]],
) -> tuple[ResourceEvent, ...]:
    """Call ``event_builder`` for each name in ``active_auxiliary_ports``."""
    events: list[ResourceEvent] = []
    for port_name in result.active_auxiliary_ports:
        events.extend(event_builder(port_name))
    return tuple(events)


def emit_binary_backward_resource_events(
    result: OperationResult,
    *,
    materializes_vjp: bool,
    parameter_grad_ports: frozenset[str] = frozenset(),
) -> tuple[ResourceEvent, ...]:
    """Emit backward auxiliary resource events for binary operations."""
    active = set(result.active_auxiliary_ports)
    events: list[ResourceEvent] = []
    for port_name in (GRAD_LEFT, GRAD_RIGHT):
        if port_name not in active:
            continue
        unreduced_port: str | None = None
        if materializes_vjp:
            candidate = f"{port_name}_unreduced"
            if candidate in active:
                unreduced_port = candidate
        events.extend(
            backward_gradient_port_events(
                port_name,
                unreduced_port=unreduced_port if materializes_vjp else None,
                is_parameter=port_name in parameter_grad_ports,
            )
        )
    return tuple(events)
