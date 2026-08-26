"""Identity operation declaration."""

from zepto.compose.values import Tensor
from ..ports import Port
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
    def input_ports(self) -> tuple[Port, ...]:
        """Return the input port declaration."""
        return (Port("input"),)

    @property
    def output_ports(self) -> tuple[Port, ...]:
        """Return the output port declaration."""
        return (Port("output"),)

    @property
    def backward(self) -> BackwardSpec:
        """Declare the identity gradient without saved forward state."""
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        """Preserve the input tensor description unchanged."""
        return inputs

    def output_aliases(self) -> tuple[AliasSpec | None, ...]:
        """Declare the output as a view of the input storage."""
        return (AliasSpec("input"),)

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        """Save nothing because the identity gradient passes through."""
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return zero because identity performs no arithmetic."""
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        """Return zero because identity derivative performs no arithmetic."""
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report the output alias event."""
        return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)

__all__ = ["Identity"]
