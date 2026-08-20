"""Transpose view operation declaration."""

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
class Transpose(Operation):
    """Permute tensor dimensions while preserving input storage."""

    permutation: tuple[int, ...]

    @property
    def family(self) -> str:
        """Return the stable transpose family name."""
        return "transpose"

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
        inferred transposed metadata.
        """
        return (PortSpec("output"),)

    @property
    def backward(self) -> BackwardSpec:
        """Declare the inverse view gradient without saved forward state."""
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_result(self, inputs: tuple[TensorMetadata, ...]) -> OperationResult:
        """Infer the permuted shape and preserve input storage."""
        rank = len(inputs[0].shape)
        normalized = tuple(
            axis if axis >= 0 else axis + rank for axis in self.permutation
        )
        if len(normalized) != rank or sorted(normalized) != list(range(rank)):
            raise ValueError("transpose permutation must cover every dimension once")
        shape = tuple(inputs[0].shape[index] for index in normalized)
        return OperationResult(
            (TensorMetadata(
                shape,
                inputs[0].semantic_type,
                require_grad=inputs[0].require_grad,
            ),),
            (AliasSpec("input"),),
        )

    def forward_flops(self, context: EstimationContext, result: OperationResult) -> int:
        """Return zero because transpose performs no arithmetic."""
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report the output view alias event."""
        return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)

__all__ = ["Transpose"]
