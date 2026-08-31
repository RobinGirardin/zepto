"""Elementwise greater-than comparison operation declaration."""

from zepto.compose.values import Tensor
from ..ports import Port
from .base import Operation
from .helpers import allocate, broadcast_tensor, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


class GreaterThan(Operation):
    """Return an elementwise boolean mask where ``left > right``."""

    @property
    def family(self) -> str:
        return "greater_than"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("left"), Port("right"))

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(supported=False)

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        left, right = inputs
        output = broadcast_tensor(
            (left, right),
            family=self.family,
            semantic_type="comparison_mask",
        )
        return (
            Tensor(
                shape=output.shape,
                semantic_type=output.semantic_type,
                requires_grad=False,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return tuple(allocate(len(self.output_ports)))


__all__ = ["GreaterThan"]
