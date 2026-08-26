"""Multi-input concatenation operation declaration."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import allocate, persist_only_events, reduced_gradient_metadata
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"concat axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class Concat(Operation):
    """Concatenate two or more tensors along one axis."""

    axis: int
    input_count: int

    @property
    def family(self) -> str:
        return "concat"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return tuple(
            PortSpec(f"input_{index}") for index in range(self.input_count)
        )

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return tuple(
            PortSpec(f"grad_input_{index}", ValueKind.GRADIENT)
            for index in range(self.input_count)
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return tuple(reduced_gradient_metadata(value) for value in inputs)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return tuple(
            f"grad_input_{index}"
            for index, value in enumerate(inputs)
            if value.requires_grad
        )

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=tuple(
                f"input_{index}" for index in range(self.input_count)
            ),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        if self.input_count < 2:
            raise ValueError("concat requires at least two input tensors")
        if len(inputs) != self.input_count:
            raise ValueError(
                f"concat expects {self.input_count} inputs, got {len(inputs)}"
            )
        rank = len(inputs[0].shape)
        if rank == 0:
            raise ValueError("concat requires ranked input tensors")
        axis = _normalize_axis(self.axis, rank)
        output_shape = list(inputs[0].shape)
        output_shape[axis] = 0
        for index, value in enumerate(inputs):
            if len(value.shape) != rank:
                raise ValueError("concat inputs must share rank")
            for dim_index, (left, right) in enumerate(
                zip(inputs[0].shape, value.shape, strict=True)
            ):
                if dim_index == axis:
                    output_shape[axis] += right
                elif left != right:
                    raise ValueError(
                        f"concat non-concat dimensions mismatch on input_{index}: "
                        f"{inputs[0].shape} vs {value.shape}"
                    )
        requires_grad = any(value.requires_grad for value in inputs)
        return (
            ValueMetadata(
                tuple(output_shape),
                inputs[0].semantic_type,
                requires_grad=requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)


__all__ = ["Concat"]
