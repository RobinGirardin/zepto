"""Elementwise maximum operation declaration."""

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


class Maximum(Operation):
    """Take the elementwise maximum of two tensors with broadcast semantics."""

    @property
    def family(self) -> str:
        """Return the stable maximum family name."""
        return "maximum"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return the left and right operand ports."""
        return (PortSpec("left"), PortSpec("right"))

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return the elementwise maximum output port."""
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        """Declare mask-routed gradients and their possible saved values.

        The gradient routes to whichever operand was larger:
        ``dL/dA = dL/dY * 1[A > B]`` and ``dL/dB = dL/dY * 1[B > A]``,
        so both masks require both operands.
        """
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
        """Infer the broadcast elementwise maximum metadata."""
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
        """Save both operands whenever any gradient is requested.

        Each backward mask (``1[left > right]`` or ``1[right > left]``)
        compares both operands, so a single requested gradient still needs
        the pair.
        """
        left, right = inputs
        if left.require_grad or right.require_grad:
            return ("left", "right")
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return one comparison per output element."""
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        """Return one conditional multiply per element per requested gradient."""
        left = context.metadata_for("left")
        right = context.metadata_for("right")
        output = context.metadata_for("output")
        if left is None or right is None or output is None:
            raise ValueError(
                "Estimation context must provide 'left', 'right', and 'output' ports"
            )
        flops = 0
        if left.require_grad:
            flops += numel(output)
        if right.require_grad:
            flops += numel(output)
        return flops

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the elementwise maximum output."""
        return allocate(result)


__all__ = ["Maximum"]
