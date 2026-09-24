"""Elementwise power operation declaration."""

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import (
    includes_backward,
    GRAD_LEFT,
    GRAD_LEFT_UNREDUCED,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    active_binary_auxiliary_ports,
    allocate,
    broadcast_tensor,
    broadcast_reduction_flops,
    backward_gradient_port_events,
    numel,
    activation_grad_events,
    reduced_gradient_tensor,
    unreduced_gradient_tensor,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_BASE = "grad_base"
GRAD_EXPONENT = "grad_exponent"
GRAD_BASE_UNREDUCED = "grad_base_unreduced"
GRAD_EXPONENT_UNREDUCED = "grad_exponent_unreduced"


class Pow(Operation):
    """Raise base to exponent with NumPy-style broadcast semantics."""

    @property
    def family(self) -> str:
        return "pow"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("base"), Port("exponent"))

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (
            Port(GRAD_BASE, ValueKind.GRADIENT),
            Port(GRAD_EXPONENT, ValueKind.GRADIENT),
            Port(GRAD_BASE_UNREDUCED, ValueKind.GRADIENT),
            Port(GRAD_EXPONENT_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        base, exponent = inputs
        (output,) = outputs
        return (
            reduced_gradient_tensor(base),
            reduced_gradient_tensor(exponent),
            unreduced_gradient_tensor(output),
            unreduced_gradient_tensor(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        base, exponent = inputs
        (output,) = outputs
        rename = {
            GRAD_LEFT: GRAD_BASE,
            GRAD_RIGHT: GRAD_EXPONENT,
            GRAD_LEFT_UNREDUCED: GRAD_BASE_UNREDUCED,
            GRAD_RIGHT_UNREDUCED: GRAD_EXPONENT_UNREDUCED,
        }
        return tuple(
            rename[name]
            for name in active_binary_auxiliary_ports(
                base, exponent, output, materializes_vjp=True
            )
        )

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("base", "exponent", "output"),
            gradient_inputs=("output",),
            gradient_outputs=("base", "exponent"),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        base, exponent = inputs
        return (
            broadcast_tensor(
                (base, exponent),
                family=self.family,
                semantic_type="tensor",
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        base, exponent = inputs
        names: list[str] = []
        if base.requires_grad:
            names.extend(("base", "exponent"))
        if exponent.requires_grad:
            names.extend(("base", "output"))
        return tuple(dict.fromkeys(names))

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        base = context.tensor_for("base")
        exponent = context.tensor_for("exponent")
        output = context.tensor_for("output")
        if base is None or exponent is None or output is None:
            raise ValueError(
                "Estimation context must provide 'base', 'exponent', and 'output' ports"
            )
        flops = 0
        if base.requires_grad:
            flops += 3 * numel(output)
            if base.shape != output.shape:
                flops += broadcast_reduction_flops(base, output)
        if exponent.requires_grad:
            flops += 3 * numel(output)
            if exponent.shape != output.shape:
                flops += broadcast_reduction_flops(exponent, output)
        return flops

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if not includes_backward(context.phase):
            return tuple(events)
        unreduced_by_port = {
            GRAD_BASE: GRAD_BASE_UNREDUCED,
            GRAD_EXPONENT: GRAD_EXPONENT_UNREDUCED,
        }
        active = set(result.active_auxiliary_ports)
        for port_name in (GRAD_BASE, GRAD_EXPONENT):
            if port_name not in active:
                continue
            unreduced = unreduced_by_port[port_name]
            if unreduced in active:
                events.extend(
                    backward_gradient_port_events(
                        port_name, unreduced_port=unreduced
                    )
                )
            else:
                events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["Pow"]
