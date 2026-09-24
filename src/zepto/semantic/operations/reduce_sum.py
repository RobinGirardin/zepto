"""Axis reduction by sum."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import includes_backward, allocate, activation_grad_events, numel, reduced_gradient_tensor
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


def _normalize_axes(axis: int | tuple[int, ...], rank: int) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, int) else axis
    normalized: list[int] = []
    for dim in axes:
        resolved = dim if dim >= 0 else rank + dim
        if resolved < 0 or resolved >= rank:
            raise ValueError(f"reduce_sum axis {dim} out of range for rank {rank}")
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def _reduce_output_shape(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    *,
    keepdim: bool,
) -> tuple[int, ...]:
    axes = set(_normalize_axes(axis, len(shape)))
    if keepdim:
        return tuple(1 if index in axes else dim for index, dim in enumerate(shape))
    return tuple(dim for index, dim in enumerate(shape) if index not in axes)


@dataclass(frozen=True, slots=True)
class ReduceSum(Operation):
    """Sum tensor elements along one or more axes."""

    axis: int | tuple[int, ...]
    keepdim: bool = False

    @property
    def family(self) -> str:
        return "reduce_sum"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"),)

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
        return (reduced_gradient_tensor(inputs[0]),)

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
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (input_tensor,) = inputs
        if not input_tensor.shape:
            raise ValueError("reduce_sum requires a ranked input tensor")
        output_shape = _reduce_output_shape(
            input_tensor.shape, self.axis, keepdim=self.keepdim
        )
        return (
            Tensor(
                shape=output_shape,
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
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        input_tensor = context.tensor_for("input")
        output = context.tensor_for("output")
        if input_tensor is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        return numel(input_tensor) - numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if not includes_backward(context.phase):
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["ReduceSum"]
