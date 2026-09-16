"""Shared additive sliding-window causal mask allocation."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port
from .base import Operation
from .records import (
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
    ResourceEventKind,
)


@dataclass(frozen=True, slots=True)
class MaterializedSlidingWindowCausalMask(Operation):
    """Allocate a persistent additive mask of shape ``(1, S, S)``.

    Combines standard causal masking (query row ``i`` may not attend to keys
    ``j > i``) with a sliding band (keys more than ``window_size - 1`` positions
    behind the query are forbidden). Values are not stored in the structural
    graph — only shape, persistence, and allocation semantics are modeled.
    """

    seq_len: int
    window_size: int

    @property
    def family(self) -> str:
        return "materialized_sliding_window_causal_mask"

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return ()

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
        if self.seq_len <= 0:
            raise ValueError(
                "materialized_sliding_window_causal_mask seq_len must be positive"
            )
        if self.window_size <= 0:
            raise ValueError(
                "materialized_sliding_window_causal_mask window_size must be positive"
            )
        if self.window_size > self.seq_len:
            raise ValueError(
                "materialized_sliding_window_causal_mask window_size must not "
                f"exceed seq_len ({self.window_size} > {self.seq_len})"
            )
        return (
            Tensor(
                shape=(1, self.seq_len, self.seq_len),
                semantic_type="tensor",
                requires_grad=False,
                persistent=True,
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
        return (
            ResourceEvent(ResourceEventKind.ALLOCATE, "output:0"),
            ResourceEvent(ResourceEventKind.PERSIST, "output:0"),
        )


__all__ = ["MaterializedSlidingWindowCausalMask"]
