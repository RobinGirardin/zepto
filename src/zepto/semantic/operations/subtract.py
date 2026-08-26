"""Elementwise subtraction operation declaration."""

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import (
    GRAD_LEFT,
    GRAD_LEFT_UNREDUCED,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    active_binary_auxiliary_ports,
    allocate,
    broadcast_tensor,
    broadcast_reduction_flops,
    numel,
    persist_only_events,
    reduced_gradient_tensor,
    unreduced_gradient_tensor,
)
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class Subtract(Operation):
    """Subtract the right tensor from the left with broadcast semantics."""

    @property
    def family(self) -> str:
        """Return the stable subtraction family name."""
        return "subtract"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        """Return the left and right input port declarations."""
        return (Port("left"), Port("right"))

    @property
    def output_ports(self) -> tuple[Port, ...]:
        """Return the output port declaration."""
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
        left, right = inputs
        (output,) = outputs
        return (
            reduced_gradient_tensor(left),
            reduced_gradient_tensor(right),
            unreduced_gradient_tensor(output),
            unreduced_gradient_tensor(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        left, right = inputs
        (output,) = outputs
        return active_binary_auxiliary_ports(
            left, right, output, materializes_vjp=False
        )

    @property
    def backward(self) -> BackwardSpec:
        """Declare upstream output gradients for both operand gradients."""
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("left", "right"),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        """Infer the broadcast-compatible output tensor."""
        left, right = inputs
        output = broadcast_tensor(
            (left, right),
            family=self.family,
            semantic_type="tensor",
        )
        return (output,)

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        """Save nothing because subtraction gradients pass through (negated)."""
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return one arithmetic FLOP per output element."""
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        """Return broadcast reduction and negation FLOPs."""
        left = context.tensor_for("left")
        right = context.tensor_for("right")
        output = context.tensor_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flop = 0
        if left.requires_grad and left.shape != output.shape:
            flop += broadcast_reduction_flops(left, output)
        if right.requires_grad:
            if right.shape != output.shape:
                flop += broadcast_reduction_flops(right, output)
            flop += numel(right)
        return flop

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report forward output allocation and backward gradient aux events."""
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)


__all__ = ["Subtract"]
