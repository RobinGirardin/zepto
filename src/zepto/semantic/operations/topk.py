"""Partial top-k selection along one axis (torch.topk semantics)."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port
from .base import Operation
from .helpers import allocate
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"topk dim {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class TopK(Operation):
    """Return the k largest elements and their indices along ``dim``.

    Matches PyTorch ``torch.topk`` with ``largest=True, sorted=True``.
    Indices are 0-based positions along ``dim`` (expert ids for MoE routers).
    """

    k: int
    dim: int = -1

    @property
    def family(self) -> str:
        return "topk"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"),)

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("values"), Port("indices"))

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(supported=False)

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        (scores,) = inputs
        if not scores.shape:
            raise ValueError("topk requires a ranked input tensor")
        if self.k <= 0:
            raise ValueError(f"topk k must be positive, got {self.k}")
        axis = _normalize_axis(self.dim, len(scores.shape))
        axis_size = scores.shape[axis]
        if self.k > axis_size:
            raise ValueError(
                f"topk k ({self.k}) must not exceed size along dim ({axis_size})"
            )
        out_shape = list(scores.shape)
        out_shape[axis] = self.k
        out_shape_tuple = tuple(out_shape)
        return (
            Tensor(
                shape=out_shape_tuple,
                dtype=scores.dtype,
                semantic_type=scores.semantic_type,
                requires_grad=scores.requires_grad,
                persistent=scores.persistent,
            ),
            Tensor(
                shape=out_shape_tuple,
                dtype=scores.dtype,
                semantic_type="routing_index",
                requires_grad=False,
                persistent=False,
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


__all__ = ["TopK"]
