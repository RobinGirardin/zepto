"""Elementwise multiplication operation declaration."""

from ..metadata import TensorMetadata
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
from .records import (
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
)


class Multiply(Operation):
    """Multiply two tensors using elementwise broadcast semantics."""

    @property
    def family(self) -> str:
        """Return the stable multiplication family name."""
        return "multiply"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return the left and right operand ports."""
        return (PortSpec("left"), PortSpec("right"))

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return the product output port."""
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
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[TensorMetadata, ...]:
        left, right = inputs
        (output,) = outputs
        return (
            reduced_gradient_metadata(left),
            reduced_gradient_metadata(right),
            unreduced_gradient_metadata(output),
            unreduced_gradient_metadata(output),
        )

    def active_auxiliary_ports(
        self,
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[str, ...]:
        left, right = inputs
        (output,) = outputs
        return active_binary_auxiliary_ports(
            left, right, output, materializes_vjp=True
        )

    @property
    def backward(self) -> BackwardSpec:
        """Declare product-rule gradients and their possible saved values."""
        return BackwardSpec(
            supported=True,
            saved_for_backward=("left", "right"),
            gradient_inputs=("output",),
            gradient_outputs=("left", "right"),
        )

    def infer_outputs(
        self,
        inputs: tuple[TensorMetadata, ...],
    ) -> tuple[TensorMetadata, ...]:
        """Infer the broadcast product metadata."""
        left, right = inputs
        output = broadcast_metadata(
            (left, right),
            family=self.family,
            semantic_type="tensor",
        )
        return (output,)

    def saved_for_backward(
        self,
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[str, ...]:
        """Save the opposite operand for each requested product-rule VJP."""
        left, right = inputs
        names: list[str] = []
        if right.requires_grad:
            names.append("left")
        if left.requires_grad:
            names.append("right")
        return tuple(names)

    def forward_flops(self, context: EstimationContext) -> int:
        """Return one multiplication per output element."""
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        """Return VJP and broadcast reduction FLOPs."""
        left = context.metadata_for("left")
        right = context.metadata_for("right")
        output = context.metadata_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flop = 0
        if left.requires_grad:
            flop += numel(output)
            if left.shape != output.shape:
                flop += broadcast_reduction_flops(left, output)
        if right.requires_grad:
            flop += numel(output)
            if right.shape != output.shape:
                flop += broadcast_reduction_flops(right, output)
        return flop

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        """Report forward output allocation and backward gradient aux events."""
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        events.extend(
            emit_binary_backward_resource_events(result, materializes_vjp=True)
        )
        return tuple(events)


__all__ = ["Multiply"]
