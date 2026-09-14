"""NumPy-take / index-select operation declaration."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import allocate, activation_grad_events, reduced_gradient_tensor
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"gather axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class Gather(Operation):
    """Select slices from input along axis (NumPy take / embedding lookup)."""

    axis: int

    @property
    def family(self) -> str:
        return "gather"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"), Port("index"))

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (Port(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        input_tensor, _index = inputs
        return (reduced_gradient_tensor(input_tensor),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("index",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        input_tensor, index = inputs
        if not input_tensor.shape:
            raise ValueError("gather requires a ranked input tensor")
        axis = _normalize_axis(self.axis, len(input_tensor.shape))
        output_shape = list(input_tensor.shape)
        output_shape[axis : axis + 1] = list(index.shape)
        return (
            Tensor(
                shape=tuple(output_shape),
                dtype=input_tensor.dtype,
                semantic_type=input_tensor.semantic_type,
                requires_grad=input_tensor.requires_grad,
                persistent=input_tensor.persistent,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("index",)
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


__all__ = ["Gather"]
