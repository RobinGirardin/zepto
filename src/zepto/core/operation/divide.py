"""Elementwise division operation declaration."""

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .helpers import allocate, broadcast_metadata, numel
from .records import (
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
)


class Divide(Operation):
    """Divide the left tensor by the right with broadcast semantics."""

    @property
    def family(self) -> str:
        """Return the stable division family name."""
        return "divide"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return the left (dividend) and right (divisor) operand ports."""
        return (PortSpec("left"), PortSpec("right"))

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return the quotient output port."""
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        """Declare quotient-rule gradients and their possible saved values."""
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
        """Infer the broadcast quotient metadata."""
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
        """Save the operands each requested quotient-rule VJP depends on.

        The left gradient is ``grad / right`` and only needs the divisor.
        The right gradient is ``-grad * left / right**2`` and needs both
        operands.
        """
        left, right = inputs
        names: list[str] = []
        if right.requires_grad:
            names.append("left")
        if left.requires_grad or right.requires_grad:
            names.append("right")
        return tuple(names)

    def forward_flops(self, context: EstimationContext) -> int:
        """Return one division per output element."""
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        """Return the elementwise cost of each requested operand gradient.
        """
        left = context.metadata_for("left")
        right = context.metadata_for("right")
        output = context.metadata_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flops = 0
        if left.requires_grad:
            flops += 2 * numel(output)
        if right.requires_grad:
            flops += 4 * numel(output)
        return flops

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the quotient output."""
        return allocate(result)


__all__ = ["Divide"]
