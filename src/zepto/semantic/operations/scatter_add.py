"""Accumulate indexed source rows into a base tensor (torch.index_add semantics)."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port
from .base import Operation
from .helpers import allocate, numel
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"scatter_add axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class ScatterAdd(Operation):
    """Add ``updates`` into ``input`` at ``indices`` along ``axis``.

    Zepto graphs are immutable: this op **returns a new tensor** with the
    accumulated result (conceptually ``out = input; out.index_add_(…)``).
    Duplicate indices **sum** contributions (same as ``torch.index_add``).
    """

    axis: int = 0

    @property
    def family(self) -> str:
        return "scatter_add"

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
            raise ValueError("scatter_add requires ranked input and updates tensors")
        axis = _normalize_axis(self.axis, len(base.shape))
        if indices.shape and len(indices.shape) != 1:
            raise ValueError(
                "scatter_add indices must be rank-1 (v1 MoE contract)"
            )
        if len(updates.shape) != len(base.shape):
            raise ValueError(
                "scatter_add updates rank must match input rank"
            )
        for dim, (u, b) in enumerate(zip(updates.shape, base.shape)):
            if dim == axis:
                if indices.shape and u != indices.shape[0]:
                    raise ValueError(
                        f"scatter_add updates size on axis {axis} ({u}) must "
                        f"match len(indices) ({indices.shape[0] if indices.shape else 0})"
                    )
            elif u != b:
                raise ValueError(
                    f"scatter_add updates shape {updates.shape} incompatible with "
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


__all__ = ["ScatterAdd"]
