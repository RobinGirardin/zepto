"""Broadcast add of a bias parameter to an activation."""

from zepto.compose.values import Tensor
from zepto.semantic.metadata import PortContract
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    broadcast_tensor,
    broadcast_reduction_flops,
    numel,
    activation_grad_events,
    weight_grad_accum_events,
    reduced_gradient_tensor,
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
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"),)

    @property
    def parameter_ports(self) -> tuple[Port, ...]:
        return (
            Port(
                "bias",
                value_kind=ValueKind.PARAMETER,
                contract=PortContract(semantic_type="bias"),
            ),
        )

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (
            Port(GRAD_INPUT, ValueKind.GRADIENT),
            Port(GRAD_BIAS, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (activation,) = inputs
        (bias,) = parameters
        return (
            reduced_gradient_tensor(activation),
            reduced_gradient_tensor(bias),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        (activation,) = inputs
        (bias,) = parameters
        active: list[str] = []
        if activation.requires_grad:
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
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (activation,) = inputs
        (bias,) = parameters
        if len(bias.shape) != 1:
            raise ValueError(f"parameter_bias bias must be rank-1, got {bias.shape}")
        return (
            broadcast_tensor(
                (activation, bias),
                family=self.family,
                semantic_type=activation.semantic_type,
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
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        activation = context.tensor_for("input")
        bias = context.tensor_for("bias")
        output = context.tensor_for("output")
        if activation is None or bias is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input', 'bias', and 'output' ports"
            )
        flops = 0
        if activation.requires_grad:
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
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            if port_name == GRAD_BIAS:
                events.extend(weight_grad_accum_events(port_name))
            else:
                events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["ParameterBias"]
