"""Elementwise subtraction operation declaration."""

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .helpers import allocate, broadcast_metadata, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class Subtract(Operation):
    """Subtract the right tensor from the left with broadcast semantics."""

    @property
    def family(self) -> str:
        """Return the stable subtraction family name."""
        return "subtract"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return the left and right input port declarations.

        Their metadata remains ``None`` until graph construction binds the
        concrete input tensor metadata.
        """
        return (PortSpec("left"), PortSpec("right"))

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return the output port declaration.

        Its metadata remains ``None`` until graph construction binds the
        inferred output tensor metadata.
        """
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        """Declare upstream output gradients for both operand gradients."""
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("left", "right"),
        )

    def infer_outputs(
        self, inputs: tuple[TensorMetadata, ...]
    ) -> tuple[TensorMetadata, ...]:
        """Infer the broadcast-compatible output metadata."""
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
        """Save nothing because subtraction gradients pass through (negated)."""
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return one arithmetic FLOP per output element."""
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        """Return one elementwise gradient operation per operand."""
        left = context.metadata_for("left")
        right = context.metadata_for("right")
        output = context.metadata_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flop = 0
        if left.requires_grad:
            flop += 0
        if right.requires_grad:
            flop += 0
        return flop

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the output."""
        return allocate(result)


__all__ = ["Subtract"]
