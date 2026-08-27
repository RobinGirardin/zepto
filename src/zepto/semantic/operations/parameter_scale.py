"""Broadcast multiply of an activation by a 1-D parameter vector."""

from ..metadata import ValueMetadata, TensorRole
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    GRAD_LEFT,
    GRAD_LEFT_UNREDUCED,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    active_binary_auxiliary_ports,
    allocate,
    broadcast_metadata,
    broadcast_reduction_flops,
    emit_binary_backward_resource_events,
    numel,
    reduced_gradient_metadata,
    unreduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class ParameterScale(Operation):
    """Elementwise scale ``input`` by a 1-D ``weight`` parameter (γ in norms)."""

    @property
    def family(self) -> str:
        return "parameter_scale"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def parameter_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(
                "weight",
                value_kind=ValueKind.PARAMETER,
                metadata=ValueMetadata((), semantic_type="weight"),
            ),
        )

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(GRAD_LEFT, ValueKind.GRADIENT),
            PortSpec(GRAD_RIGHT, ValueKind.GRADIENT),
            PortSpec(GRAD_LEFT_UNREDUCED, ValueKind.GRADIENT),
            PortSpec(GRAD_RIGHT_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (input_meta,) = inputs
        (weight,) = parameters
        (output,) = outputs
        return (
            reduced_gradient_metadata(input_meta),
            reduced_gradient_metadata(weight),
            unreduced_gradient_metadata(output),
            unreduced_gradient_metadata(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        (input_meta,) = inputs
        (weight,) = parameters
        (output,) = outputs
        return active_binary_auxiliary_ports(
            input_meta, weight, output, materializes_vjp=True
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
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (input_meta,) = inputs
        (weight,) = parameters
        if len(weight.shape) != 1:
            raise ValueError(
                f"parameter_scale weight must be rank-1, got {weight.shape}"
            )
        if input_meta.shape[-1] != weight.shape[0]:
            raise ValueError(
                f"parameter_scale last dim {input_meta.shape[-1]} must match "
                f"weight length {weight.shape[0]}"
            )
        return (
            ValueMetadata(
                input_meta.shape,
                input_meta.semantic_type,
                requires_grad=input_meta.requires_grad or weight.requires_grad,
                role=TensorRole.ACTIVATION,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        (input_meta,) = inputs
        (weight,) = parameters
        saved: list[str] = []
        if weight.requires_grad:
            saved.append("input")
        if input_meta.requires_grad:
            saved.append("weight")
        return tuple(saved)

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        input_meta = context.metadata_for("input")
        weight = context.metadata_for("weight")
        output = context.metadata_for("output")
        if input_meta is None or weight is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input', 'weight', and 'output' ports"
            )
        flops = 0
        if input_meta.requires_grad:
            flops += numel(output)
            if weight.shape != output.shape:
                flops += broadcast_reduction_flops(weight, output)
        if weight.requires_grad:
            flops += numel(output)
            if input_meta.shape != output.shape:
                flops += broadcast_reduction_flops(input_meta, output)
        return flops

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        events.extend(
            emit_binary_backward_resource_events(result, materializes_vjp=True)
        )
        return tuple(events)


__all__ = ["ParameterScale"]
