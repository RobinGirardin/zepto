"""Reshape view operation declaration."""

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class Reshape(Operation):
    """Present an input tensor with a different shape as a view."""

    shape: tuple[int, ...]

    @property
    def family(self) -> str:
        """Return the stable reshape family name."""
        return "reshape"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        """Return the input port declaration."""
        return (Port("input"),)

    @property
    def output_ports(self) -> tuple[Port, ...]:
        """Return the output port declaration."""
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return ()

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return ()

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    @property
    def backward(self) -> BackwardSpec:
        """Declare the view gradient without saved forward state."""
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
        """Infer the requested shape from the input tensor."""
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
        return (
            Tensor(
                shape=self.shape,
                dtype=inputs[0].dtype,
                semantic_type=inputs[0].semantic_type,
                requires_grad=inputs[0].requires_grad,
                persistent=inputs[0].persistent,
            ),
        )

    def output_aliases(self) -> tuple[AliasSpec | None, ...]:
        """Declare the output as a view of the input storage."""
        return (AliasSpec("input"),)

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        """Save nothing because the backward reshape needs only shapes."""
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return zero because reshape performs no arithmetic."""
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        """Return zero because reshape derivative performs no arithmetic."""
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report the output view alias event."""
        return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)

__all__ = ["Reshape"]
