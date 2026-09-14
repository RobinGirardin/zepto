"""Multi-input concatenation operation declaration."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import allocate, activation_grad_events, reduced_gradient_tensor
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
    def input_ports(self) -> tuple[Port, ...]:
        return tuple(Port(f"input_{index}") for index in range(self.input_count))

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return tuple(
            Port(f"grad_input_{index}", ValueKind.GRADIENT)
            for index in range(self.input_count)
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        return tuple(reduced_gradient_tensor(value) for value in inputs)

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
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
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
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
        return (
            Tensor(
                shape=tuple(output_shape),
                dtype=inputs[0].dtype,
                semantic_type=inputs[0].semantic_type,
                requires_grad=any(value.requires_grad for value in inputs),
                persistent=inputs[0].persistent,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["Concat"]
