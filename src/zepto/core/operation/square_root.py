"""Elementwise square-root operation declaration."""

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class SquareRoot(Operation):
    """Apply an elementwise square root and retain its output for backward."""

    @property
    def family(self) -> str:
        """Return the stable square-root family name."""
        return "square_root"

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
        """Declare the output reused by the backward pass.

        The input gradient is ``grad / (2 * output)``, so saving the forward
        output avoids recomputing the square root.
        """
        return BackwardSpec(
            supported=True,
            saved_for_backward=("output",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self, inputs: tuple[TensorMetadata, ...]
    ) -> tuple[TensorMetadata, ...]:
        """Preserve the input metadata for the square-root output."""
        return inputs

    def saved_for_backward(
        self,
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[str, ...]:
        """Save the output only when the input gradient is requested."""
        if inputs[0].require_grad:
            return ("output",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return one square root per output element."""
        output = context.metadata_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        """Return two operations per element for ``grad / (2 * output)``."""
        metadata = context.metadata_for("input")
        output = context.metadata_for("output")
        if metadata is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        flop = 0
        if metadata.require_grad:
            flop += 3 * numel(output)
        return flop

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for the output."""
        return allocate(result)


__all__ = ["SquareRoot"]
