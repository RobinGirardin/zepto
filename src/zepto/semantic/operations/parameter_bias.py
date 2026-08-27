"""Broadcast add of a bias parameter to an activation."""

from ..metadata import ValueMetadata, TensorRole
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    broadcast_metadata,
    broadcast_reduction_flops,
    numel,
    persist_only_events,
    reduced_gradient_metadata,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"
GRAD_BIAS = "grad_bias"


class ParameterBias(Operation):
    """Elementwise add ``input + bias`` with a 1-D ``bias`` parameter."""

    @property
    def family(self) -> str:
        return "parameter_bias"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def parameter_ports(self) -> tuple[PortSpec, ...]:
        return (
            PortSpec(
                "bias",
                value_kind=ValueKind.PARAMETER,
                metadata=ValueMetadata((), semantic_type="bias"),
            ),
        )

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        from ..ports import ValueKind as VK

        return (
            PortSpec(GRAD_INPUT, VK.GRADIENT),
            PortSpec(GRAD_BIAS, VK.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (input_meta,) = inputs
        (bias,) = parameters
        return (
            reduced_gradient_metadata(input_meta),
            reduced_gradient_metadata(bias),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        (input_meta,) = inputs
        (bias,) = parameters
        active: list[str] = []
        if input_meta.requires_grad:
            active.append(GRAD_INPUT)
        if bias.requires_grad:
            active.append(GRAD_BIAS)
        return tuple(active)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input", "bias"),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        (input_meta,) = inputs
        (bias,) = parameters
        if len(bias.shape) != 1:
            raise ValueError(f"parameter_bias bias must be rank-1, got {bias.shape}")
        output = broadcast_metadata(
            (input_meta, bias),
            family=self.family,
            semantic_type=input_meta.semantic_type,
        )
        return (
            ValueMetadata(
                output.shape,
                input_meta.semantic_type,
                requires_grad=input_meta.requires_grad or bias.requires_grad,
                role=TensorRole.ACTIVATION,
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
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        input_meta = context.metadata_for("input")
        bias = context.metadata_for("bias")
        output = context.metadata_for("output")
        if input_meta is None or bias is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input', 'bias', and 'output' ports"
            )
        flops = 0
        if input_meta.requires_grad:
            flops += numel(output)
        if bias.requires_grad:
            flops += numel(output)
            if bias.shape != output.shape:
                flops += broadcast_reduction_flops(bias, output)
        return flops

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)


__all__ = ["ParameterBias"]
