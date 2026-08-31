"""Elementwise conditional selection operation declaration."""

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import (
    allocate,
    backward_gradient_port_events,
    broadcast_reduction_flops,
    broadcast_tensor_n,
    numel,
    persist_only_events,
    reduced_gradient_tensor,
    unreduced_gradient_tensor,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_ON_TRUE = "grad_on_true"
GRAD_ON_TRUE_UNREDUCED = "grad_on_true_unreduced"
GRAD_ON_FALSE = "grad_on_false"
GRAD_ON_FALSE_UNREDUCED = "grad_on_false_unreduced"


class Where(Operation):
    """Select ``on_true`` or ``on_false`` per condition element."""

    @property
    def family(self) -> str:
        return "where"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (
            Port("condition"),
            Port("on_true"),
            Port("on_false"),
        )

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (
            Port(GRAD_ON_TRUE, ValueKind.GRADIENT),
            Port(GRAD_ON_FALSE, ValueKind.GRADIENT),
            Port(GRAD_ON_TRUE_UNREDUCED, ValueKind.GRADIENT),
            Port(GRAD_ON_FALSE_UNREDUCED, ValueKind.GRADIENT),
        )

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        _condition, on_true, on_false = inputs
        (output,) = outputs
        return (
            reduced_gradient_tensor(on_true),
            reduced_gradient_tensor(on_false),
            unreduced_gradient_tensor(output),
            unreduced_gradient_tensor(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        _condition, on_true, on_false = inputs
        (output,) = outputs
        active: list[str] = []
        if on_true.requires_grad:
            if on_true.shape != output.shape:
                active.extend((GRAD_ON_TRUE, GRAD_ON_TRUE_UNREDUCED))
            else:
                active.append(GRAD_ON_TRUE)
        if on_false.requires_grad:
            if on_false.shape != output.shape:
                active.extend((GRAD_ON_FALSE, GRAD_ON_FALSE_UNREDUCED))
            else:
                active.append(GRAD_ON_FALSE)
        return tuple(active)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("condition",),
            gradient_inputs=("output",),
            gradient_outputs=("on_true", "on_false"),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        condition, on_true, on_false = inputs
        output = broadcast_tensor_n(
            (condition, on_true, on_false),
            family=self.family,
        )
        return (
            Tensor(
                shape=output.shape,
                semantic_type=output.semantic_type,
                requires_grad=on_true.requires_grad or on_false.requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        _condition, on_true, on_false = inputs
        if on_true.requires_grad or on_false.requires_grad:
            return ("condition",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        on_true = context.tensor_for("on_true")
        on_false = context.tensor_for("on_false")
        output = context.tensor_for("output")
        if on_true is None or on_false is None or output is None:
            raise ValueError(
                "Estimation context must provide 'on_true', 'on_false', and 'output' ports"
            )
        flops = 0
        if on_true.requires_grad:
            flops += numel(output)
            if on_true.shape != output.shape:
                flops += broadcast_reduction_flops(on_true, output)
        if on_false.requires_grad:
            flops += numel(output)
            if on_false.shape != output.shape:
                flops += broadcast_reduction_flops(on_false, output)
        return flops

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        unreduced_by_port = {
            GRAD_ON_TRUE: GRAD_ON_TRUE_UNREDUCED,
            GRAD_ON_FALSE: GRAD_ON_FALSE_UNREDUCED,
        }
        active = set(result.active_auxiliary_ports)
        for port_name in (GRAD_ON_TRUE, GRAD_ON_FALSE):
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
                events.extend(persist_only_events(port_name))
        return tuple(events)


__all__ = ["Where"]
