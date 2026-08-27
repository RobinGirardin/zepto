"""KV-head replication along one axis."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import allocate, numel, persist_only_events, reduced_gradient_metadata
from .records import (
    AliasSpec,
    BackwardSpec,
    EstimationContext,
    OperationResult,
    ResourceEvent,
    ResourceEventKind,
)

GRAD_INPUT = "grad_input"


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"repeat_kv axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class RepeatKV(Operation):
    """Repeat the selected axis ``n_rep`` times (GQA KV-head expand)."""

    n_rep: int
    axis: int = 0

    @property
    def family(self) -> str:
        return "repeat_kv"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"),)

    @property
    def output_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("output"),)

    def auxiliary_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        return (reduced_gradient_metadata(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad and self.n_rep > 1:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        if self.n_rep < 1:
            raise ValueError("repeat_kv n_rep must be >= 1")
        (input_meta,) = inputs
        if not input_meta.shape:
            raise ValueError("repeat_kv requires a ranked input tensor")
        axis = _normalize_axis(self.axis, len(input_meta.shape))
        shape = list(input_meta.shape)
        shape[axis] *= self.n_rep
        return (
            ValueMetadata(
                tuple(shape),
                input_meta.semantic_type,
                requires_grad=input_meta.requires_grad,
            ),
        )

    def output_aliases(self) -> tuple[AliasSpec | None, ...]:
        if self.n_rep == 1:
            return (AliasSpec("input"),)
        return ()

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
        input_meta = context.metadata_for("input")
        output = context.metadata_for("output")
        if input_meta is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        if input_meta.requires_grad and self.n_rep > 1:
            return numel(output) - numel(input_meta)
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        if self.n_rep == 1 and result.aliases:
            return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)


__all__ = ["RepeatKV"]
