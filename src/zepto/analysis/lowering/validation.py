"""Validation for lowered nodes."""

from __future__ import annotations

from .errors import LoweringError
from ..lowered import LoweredNode
from zepto.semantic.operations.records import ResourceEvent


class LoweredNodeValidator:
    """Validate one lowered node's events and FLOP counts."""

    def validate(self, node: LoweredNode) -> None:
        self.validate_flops(node)
        self.validate_events(node)

    def validate_flops(self, node: LoweredNode) -> None:
        for name, value in (
            ("forward_flops", node.forward_flops),
            ("backward_flops", node.backward_flops),
        ):
            if not isinstance(value, int) or value < 0:
                raise LoweringError(
                    f"{node.implementation}: {name} must be a non-negative int"
                )

    def validate_events(self, node: LoweredNode) -> None:
        events = node.resource_events
        if not isinstance(events, tuple) or not all(
            isinstance(event, ResourceEvent) for event in events
        ):
            raise LoweringError(
                f"{node.implementation}: resource_events must be a tuple"
            )
        known = set(
            node.input_edges + node.output_edges + node.auxiliary_edges
        )
        for event in events:
            if event.value not in known:
                raise LoweringError(
                    f"{node.implementation}: event target {event.value!r} "
                    "is not a known lowered edge"
                )
            if event.storage is not None and event.storage not in known:
                raise LoweringError(
                    f"{node.implementation}: event storage {event.storage!r} "
                    "is not a known lowered edge"
                )
