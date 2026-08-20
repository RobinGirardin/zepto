"""Reshape view operation declaration."""

from dataclasses import dataclass

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .records import (
    AliasSpec,
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
    ResourceEventKind,
)


@dataclass(frozen=True, slots=True)
class Reshape(Operation):
    """Present an input tensor with a different shape as a view."""

    shape: tuple[int, ...]

    @property
    def family(self) -> str:
        """Return the stable reshape family name."""
        return "reshape"

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
        inferred reshaped metadata.
        """
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        """Declare the view gradient without saved forward state."""
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_result(self, inputs: tuple[TensorMetadata, ...]) -> OperationResult:
        """Infer the requested shape and preserve input storage."""
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError("reshape dimensions must be positive and non-empty")
        input_numel = 1
        for dimension in inputs[0].shape:
            input_numel *= dimension
        output_numel = 1
        for dimension in self.shape:
            output_numel *= dimension
        if input_numel != output_numel:
            raise ValueError(
                f"reshape element count mismatch: {inputs[0].shape} -> {self.shape}"
            )
        return OperationResult(
            (TensorMetadata(
                self.shape,
                inputs[0].semantic_type,
                require_grad=inputs[0].require_grad,
            ),),
            (AliasSpec("input"),),
        )

    def forward_flops(self, context: EstimationContext, result: OperationResult) -> int:
        """Return zero because reshape performs no arithmetic."""
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report the output view alias event."""
        return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)

__all__ = ["Reshape"]
