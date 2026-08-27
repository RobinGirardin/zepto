"""Transpose view operation declaration."""

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
class Transpose(Operation):
    """Permute tensor dimensions while preserving input storage."""

    permutation: tuple[int, ...]

    @property
    def family(self) -> str:
        """Return the stable transpose family name."""
        return "transpose"

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
        """Declare the inverse view gradient without saved forward state."""
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
        """Infer the permuted shape from the input tensor."""
        rank = len(inputs[0].shape)
        normalized = tuple(
            axis if axis >= 0 else axis + rank for axis in self.permutation
        )
        if len(normalized) != rank or sorted(normalized) != list(range(rank)):
            raise ValueError("transpose permutation must cover every dimension once")
        shape = tuple(inputs[0].shape[index] for index in normalized)
        return (
            Tensor(
                shape=shape,
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
        """Save nothing because the inverse permutation is declarative."""
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return zero because transpose performs no arithmetic."""
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        """Return zero because transpose derivative performs no arithmetic."""
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report the output view alias event."""
        return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)

__all__ = ["Transpose"]
