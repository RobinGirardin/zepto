"""Elementwise maximum operation declaration."""

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
    emit_binary_backward_resource_events,
    numel,
    reduced_gradient_tensor,
    unreduced_gradient_tensor,
)
from .records import (
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
)


class Maximum(Operation):
    """Take the elementwise maximum of two tensors with broadcast semantics."""

    @property
    def family(self) -> str:
        """Return the stable maximum family name."""
        return "maximum"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        """Return the left and right operand ports."""
        return (Port("left"), Port("right"))

    @property
    def output_ports(self) -> tuple[Port, ...]:
        """Return the elementwise maximum output port."""
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
            left, right, output, materializes_vjp=True
        )

    @property
    def backward(self) -> BackwardSpec:
        """Declare mask-routed gradients and their possible saved values."""
        return BackwardSpec(
            supported=True,
            saved_for_backward=("left", "right"),
            gradient_inputs=("output",),
            gradient_outputs=("left", "right"),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        """Infer the broadcast elementwise maximum tensor."""
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
        """Save both operands whenever any gradient is requested."""
        left, right = inputs
        if left.requires_grad or right.requires_grad:
            return ("left", "right")
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Comparisons are not billed as arithmetic FLOPs."""
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        """Return masked multiply and broadcast reduction FLOPs."""
        left = context.tensor_for("left")
        right = context.tensor_for("right")
        output = context.tensor_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flops = 0
        if left.requires_grad:
            flops += numel(output)
            if left.shape != output.shape:
                flops += broadcast_reduction_flops(left, output)
        if right.requires_grad:
            flops += numel(output)
            if right.shape != output.shape:
                flops += broadcast_reduction_flops(right, output)
        return flops

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        """Report forward output allocation and backward gradient aux events."""
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        events.extend(
            emit_binary_backward_resource_events(result, materializes_vjp=True)
        )
        return tuple(events)


__all__ = ["Maximum"]
