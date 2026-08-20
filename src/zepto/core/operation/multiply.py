"""Elementwise multiplication operation declaration."""

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

    @property
    def backward(self) -> BackwardSpec:
        """Declare product-rule gradients and their possible saved values."""
        return BackwardSpec(
            supported=True,
            saved_for_backward=("left", "right"),
            gradient_inputs=("output",),
            gradient_outputs=("left", "right"),
        )

    def infer_result(
        self,
        inputs: tuple[TensorMetadata, ...],
    ) -> OperationResult:
        """Infer product metadata and values required by product-rule VJPs."""
        left, right = inputs
        output = broadcast_metadata(
            (left, right),
            family=self.family,
            semantic_type="tensor",
        )

        saved_for_backward: list[str] = []
        if right.require_grad:
            saved_for_backward.append("left")
        if left.require_grad:
            saved_for_backward.append("right")

        return OperationResult(
            outputs=(output,),
            saved_for_backward=tuple(saved_for_backward),
        )

    def forward_flops(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> int:
        """Return one multiplication per output element."""
        return numel(result.outputs[0])

    def backward_flops(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> int:
        """Return one multiplication per requested operand gradient."""
        left = context.metadata_for("left")
        right = context.metadata_for("right")
        if left is None or right is None:
            raise ValueError(
                "Estimation context must provide both a 'left' and 'right' port"
            )
        gradient_count = sum(
            value.require_grad for value in (left, right)
        )
        return gradient_count * numel(result.outputs[0])

    def resource_events(
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the product output."""
        return allocate(result)


__all__ = ["Multiply"]
