"""Broadcast multiply of an activation by a 1-D parameter vector."""

from zepto.compose.values import Tensor
from zepto.semantic.metadata import PortContract
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import (
    GRAD_LEFT,
    GRAD_LEFT_UNREDUCED,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    active_binary_auxiliary_ports,
    allocate,
    broadcast_reduction_flops,
    emit_binary_backward_resource_events,
    numel,
    reduced_gradient_tensor,
    unreduced_gradient_tensor,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class ParameterScale(Operation):
    """Elementwise scale ``input`` by a 1-D ``weight`` parameter (γ in norms)."""

    @property
    def family(self) -> str:
        return "parameter_scale"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"),)

    @property
    def parameter_ports(self) -> tuple[Port, ...]:
        return (
            Port(
                "weight",
                value_kind=ValueKind.PARAMETER,
                contract=PortContract(semantic_type="weight"),
            ),
        )

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (
            Port(GRAD_LEFT, ValueKind.GRADIENT),
            Port(GRAD_RIGHT, ValueKind.GRADIENT),
            Port(GRAD_LEFT_UNREDUCED, ValueKind.GRADIENT),
            Port(GRAD_RIGHT_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (activation,) = inputs
        (weight,) = parameters
        (output,) = outputs
        return (
            reduced_gradient_tensor(activation),
            reduced_gradient_tensor(weight),
            unreduced_gradient_tensor(output),
            unreduced_gradient_tensor(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        (activation,) = inputs
        (weight,) = parameters
        (output,) = outputs
        return active_binary_auxiliary_ports(
            activation, weight, output, materializes_vjp=True
        )

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("input", "weight"),
            gradient_inputs=("output",),
            gradient_outputs=("input", "weight"),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (activation,) = inputs
        (weight,) = parameters
        if len(weight.shape) != 1:
            raise ValueError(
                f"parameter_scale weight must be rank-1, got {weight.shape}"
            )
        if activation.shape[-1] != weight.shape[0]:
            raise ValueError(
                f"parameter_scale last dim {activation.shape[-1]} must match "
                f"weight length {weight.shape[0]}"
            )
        return (
            Tensor(
                shape=activation.shape,
                dtype=activation.dtype,
                semantic_type=activation.semantic_type,
                requires_grad=activation.requires_grad or weight.requires_grad,
                persistent=activation.persistent,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        (activation,) = inputs
        (weight,) = parameters
        saved: list[str] = []
        if weight.requires_grad:
            saved.append("input")
        if activation.requires_grad:
            saved.append("weight")
        return tuple(saved)

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        activation = context.tensor_for("input")
        weight = context.tensor_for("weight")
        output = context.tensor_for("output")
        if activation is None or weight is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input', 'weight', and 'output' ports"
            )
        flops = 0
        if activation.requires_grad:
            flops += numel(output)
            if weight.shape != output.shape:
                flops += broadcast_reduction_flops(weight, output)
        if weight.requires_grad:
            flops += numel(output)
            if activation.shape != output.shape:
                flops += broadcast_reduction_flops(activation, output)
        return flops

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        events.extend(
            emit_binary_backward_resource_events(result, materializes_vjp=True)
        )
        return tuple(events)


__all__ = ["ParameterScale"]
