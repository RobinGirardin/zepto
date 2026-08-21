"""Validation for lowered operations."""

from __future__ import annotations

from .errors import LoweringError
from ..lowered import LoweredOperation
from ..operation.records import ResourceEvent


class LoweredOperationValidator:
    """Validate one lowered operation's events and FLOP counts."""

    def validate(self, operation: LoweredOperation) -> None:
        self.validate_flops(operation)
        self.validate_events(operation)

    def validate_flops(self, operation: LoweredOperation) -> None:
        for name, value in (
            ("forward_flops", operation.forward_flops),
            ("backward_flops", operation.backward_flops),
        ):
            if not isinstance(value, int) or value < 0:
                raise LoweringError(
                    f"{operation.implementation}: {name} must be a non-negative int"
                )

    def validate_events(self, operation: LoweredOperation) -> None:
        events = operation.resource_events
        if not isinstance(events, tuple) or not all(
            isinstance(event, ResourceEvent) for event in events
        ):
            raise LoweringError(
                f"{operation.implementation}: resource_events must be a tuple"
            )
        known = set(
            operation.input_tensors
            + operation.output_tensors
            + operation.auxiliary_tensors
        )
        for event in events:
            if event.value not in known:
                raise LoweringError(
                    f"{operation.implementation}: event target {event.value!r} "
                    "is not a known lowered tensor"
                )
            if event.storage is not None and event.storage not in known:
                raise LoweringError(
                    f"{operation.implementation}: event storage {event.storage!r} "
                    "is not a known lowered tensor"
                )
