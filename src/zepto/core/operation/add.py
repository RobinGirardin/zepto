"""Elementwise addition operation declaration."""

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .helpers import allocate, broadcast_metadata, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class Add(Operation):
    """Add two equal-rank tensors with singleton-dimension broadcasting."""

    @property
    def family(self) -> str:
        """Return the stable addition family name."""
        return "add"

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
        """Save nothing because addition gradients pass through unchanged."""
        return ()

    def forward_flops(self, context: EstimationContext, result: OperationResult) -> int:
        """Return one arithmetic FLOP per output element."""
        return numel(result.outputs[0])

    def backward_flops(self, context: EstimationContext, result: OperationResult) -> int:
        """Return one elementwise gradient operation per operand."""
        return 2 * self.forward_flops(context, result)

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the output."""
        return allocate(result)

__all__ = ["Add"]
