"""Rectified-linear-unit operation declaration."""

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class ReLU(Operation):
    """Apply ReLU and retain its output for backward masking."""

    @property
    def family(self) -> str:
        """Return the stable ReLU family name."""
        return "relu"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return the input port declaration.

        Its metadata remains ``None`` until graph construction binds the
        concrete input tensor metadata.
        """
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return the output port declaration.

        Its metadata remains ``None`` until graph construction binds the
        inferred output tensor metadata.
        """
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        """Declare the output used to recover the backward mask."""
        return BackwardSpec(
            supported=True,
            saved_for_backward=("output",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_result(self, inputs: tuple[TensorMetadata, ...]) -> OperationResult:
        """Preserve metadata and retain the post-activation output."""
        return OperationResult(
            outputs=inputs,
            saved_for_backward=("output",),
        )

    def forward_flops(self, context: EstimationContext, result: OperationResult) -> int:
        """Return one conditional multiply per element, as in Atto."""
        return numel(result.outputs[0])

    def backward_flops(self, context: EstimationContext, result: OperationResult) -> int:
        """Return one conditional multiply per element for the backward mask."""
        return numel(result.outputs[0])

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the output."""
        return allocate(result)

__all__ = ["ReLU"]
