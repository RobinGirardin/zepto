"""Shared additive causal mask allocation."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec
from .base import Operation
from .records import (
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
    ResourceEventKind,
)


@dataclass(frozen=True, slots=True)
class MaterializedCausalMask(Operation):
    """Allocate a persistent additive causal mask of shape ``(1, S, S)``."""

    seq_len: int

    @property
    def family(self) -> str:
        return "materialized_causal_mask"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return ()

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return ()

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return ()

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(supported=False)

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        if self.seq_len <= 0:
            raise ValueError("materialized_causal_mask seq_len must be positive")
        return (
            ValueMetadata(
                (1, self.seq_len, self.seq_len),
                semantic_type="tensor",
                requires_grad=False,
                persistent=True,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
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


__all__ = ["MaterializedCausalMask"]
