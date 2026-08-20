"""Multi-output split operation declaration."""

from dataclasses import dataclass

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .base import Operation
from .helpers import allocate
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


@dataclass(frozen=True, slots=True)
class Split(Operation):
    """Split a tensor along its leading dimension into materialized outputs."""

    sizes: tuple[int, ...]

    @property
    def family(self) -> str:
        """Return the stable split family name."""
        return "split"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return the input port declaration.

        Its metadata remains ``None`` until graph construction binds the
        concrete input tensor metadata.
        """
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return one output port per configured split size.

        Output metadata remains ``None`` until graph construction binds the
        inferred metadata for each split result.
        """
        return tuple(PortSpec(f"output_{index}") for index in range(len(self.sizes)))

    @property
    def backward(self) -> BackwardSpec:
        """Declare reduction of output gradients into the input gradient."""
        return BackwardSpec(
            supported=True,
            gradient_inputs=tuple(
                f"output_{index}" for index in range(len(self.sizes))
            ),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self, inputs: tuple[TensorMetadata, ...]
    ) -> tuple[TensorMetadata, ...]:
        """Infer the ordered metadata for each split output."""
        if not self.sizes or any(size < 0 for size in self.sizes):
            raise ValueError("split sizes must be non-negative and non-empty")
        if (
            not inputs[0].shape
            or not isinstance(inputs[0].shape[0], int)
            or sum(self.sizes) != inputs[0].shape[0]
        ):
            raise ValueError("split sizes must sum to the leading dimension")
        return tuple(
            TensorMetadata(
                (size, *inputs[0].shape[1:]),
                inputs[0].semantic_type,
                require_grad=inputs[0].require_grad,
            )
            for size in self.sizes
        )

    def saved_for_backward(
        self,
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[str, ...]:
        """Save nothing because split gradients only concatenate."""
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        """Return zero because splitting performs no arithmetic."""
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        """Return zero because splitting derivative performs no arithmetic."""
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Report allocation events for all split outputs."""
        return allocate(result)

__all__ = ["Split"]
