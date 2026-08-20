"""Identity operation declaration."""

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


class Identity(Operation):
    """Return an input tensor as an aliased output."""

    @property
    def family(self) -> str:
        """Return the stable identity family name."""
        return "identity"

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
        """Declare the identity gradient without saved forward state."""
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_result(self, inputs: tuple[TensorMetadata, ...]) -> OperationResult:
        """Infer an output that aliases the input storage."""
        return OperationResult(inputs, (AliasSpec("input"),))

    def forward_flops(self, context: EstimationContext, result: OperationResult) -> int:
        """Return zero because identity performs no arithmetic."""
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report the output alias event."""
        return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)

__all__ = ["Identity"]
