"""Replace rows in a base tensor at indexed positions (multimodal fusion)."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent
from .scatter_add import _normalize_axis, _validate_scatter_shapes


@dataclass(frozen=True, slots=True)
class ScatterUpdate(Operation):
    """Return a new tensor with rows along ``axis`` replaced by ``updates``.

    Unlike :class:`ScatterAdd`, duplicate indices do **not** accumulate — each
    row is overwritten (Qwen/Muse/Gemma soft-token insertion).
    """

    axis: int = 0

    @property
    def family(self) -> str:
        return "scatter_update"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"), Port("indices"), Port("updates"))

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
        base, indices, updates = inputs
        axis = _normalize_axis(self.axis, len(base.shape))
        _validate_scatter_shapes(
            base, indices, updates, axis, family="scatter_update"
        )
        return (
            Tensor(
                shape=base.shape,
                dtype=base.dtype,
                semantic_type=base.semantic_type,
                requires_grad=base.requires_grad or updates.requires_grad,
                persistent=base.persistent,
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
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return tuple(allocate(len(self.output_ports)))


__all__ = ["ScatterUpdate"]
