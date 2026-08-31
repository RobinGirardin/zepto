"""Multi-output split operation declaration."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port
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
    def input_ports(self) -> tuple[Port, ...]:
        """Return the input port declaration."""
        return (Port("input"),)

    @property
    def output_ports(self) -> tuple[Port, ...]:
        """Return one output port per configured split size."""
        return tuple(Port(f"output_{index}") for index in range(len(self.sizes)))

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
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        """Infer the ordered tensors for each split output."""
        if not self.sizes or any(size < 0 for size in self.sizes):
            raise ValueError("split sizes must be non-negative and non-empty")
        if (
            not inputs[0].shape
            or not isinstance(inputs[0].shape[0], int)
            or sum(self.sizes) != inputs[0].shape[0]
        ):
            raise ValueError("split sizes must sum to the leading dimension")
        return tuple(
            Tensor(
                shape=(size, *inputs[0].shape[1:]),
                dtype=inputs[0].dtype,
                semantic_type=inputs[0].semantic_type,
                requires_grad=inputs[0].requires_grad,
                persistent=inputs[0].persistent,
            )
            for size in self.sizes
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
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
        return allocate(len(self.output_ports))

__all__ = ["Split"]
