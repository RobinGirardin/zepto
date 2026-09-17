"""Replace rows in a base tensor at indexed positions (multimodal fusion)."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent
from .scatter_add import _normalize_axis


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
        if not base.shape or not updates.shape:
            raise ValueError("scatter_update requires ranked input and updates tensors")
        axis = _normalize_axis(self.axis, len(base.shape))
        if indices.shape and len(indices.shape) != 1:
            raise ValueError("scatter_update indices must be rank-1")
        if len(updates.shape) != len(base.shape):
            raise ValueError("scatter_update updates rank must match input rank")
        for dim, (u, b) in enumerate(zip(updates.shape, base.shape)):
            if dim == axis:
                if indices.shape and u != indices.shape[0]:
                    raise ValueError(
                        f"scatter_update updates size on axis {axis} ({u}) must "
                        f"match len(indices) ({indices.shape[0] if indices.shape else 0})"
                    )
            elif u != b:
                raise ValueError(
                    f"scatter_update updates shape {updates.shape} incompatible with "
                    f"input shape {base.shape} on dim {dim}"
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
        updates = context.tensor_for("updates")
        if updates is None:
            raise ValueError("Estimation context must provide an 'updates' port")
        return numel(updates)

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        return tuple(allocate(len(self.output_ports)))


__all__ = ["ScatterUpdate"]
