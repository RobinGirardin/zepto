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


def _scatter_index_count(indices: Tensor, base: Tensor, axis: int, *, family: str) -> int:
    """Number of indexed positions along ``axis`` (rank-1 or batched rank-2 indices)."""
    if not indices.shape:
        return 0
    rank = len(indices.shape)
    if rank == 1:
        return indices.shape[0]
    if rank == 2:
        if len(base.shape) < 2:
            raise ValueError(
                f"{family} rank-2 indices (B, N) require input rank >= 2"
            )
        if axis == 0:
            raise ValueError(
                f"{family} rank-2 indices (B, N) cannot scatter along batch axis 0"
            )
        if indices.shape[0] != base.shape[0]:
            raise ValueError(
                f"{family} indices batch dim {indices.shape[0]} must match "
                f"input batch dim {base.shape[0]}"
            )
        return indices.shape[1]
    raise ValueError(
        f"{family} indices must be rank-1 (N,) or rank-2 (B, N), got rank {rank}"
    )


def _validate_scatter_shapes(
    base: Tensor,
    indices: Tensor,
    updates: Tensor,
    axis: int,
    *,
    family: str,
) -> None:
    if not base.shape or not updates.shape:
        raise ValueError(f"{family} requires ranked input and updates tensors")
    if len(updates.shape) != len(base.shape):
        raise ValueError(f"{family} updates rank must match input rank")
    index_count = _scatter_index_count(indices, base, axis, family=family)
    for dim, (u, b) in enumerate(zip(updates.shape, base.shape, strict=True)):
        if dim == axis:
            if indices.shape and u != index_count:
                raise ValueError(
                    f"{family} updates size on axis {axis} ({u}) must "
                    f"match index count ({index_count})"
                )
        elif u != b:
            raise ValueError(
                f"{family} updates shape {updates.shape} incompatible with "
                f"input shape {base.shape} on dim {dim}"
            )


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
        axis = _normalize_axis(self.axis, len(base.shape))
        _validate_scatter_shapes(
            base, indices, updates, axis, family="scatter_add"
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
