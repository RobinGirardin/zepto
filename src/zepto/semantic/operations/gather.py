"""NumPy-take / index-select operation declaration."""

from dataclasses import dataclass

from ..metadata import ValueMetadata
from ..ports import PortSpec, ValueKind
from .base import Operation
from .helpers import allocate, persist_only_events, reduced_gradient_metadata
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


def _normalize_axis(axis: int, rank: int) -> int:
    resolved = axis if axis >= 0 else rank + axis
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"gather axis {axis} out of range for rank {rank}")
    return resolved


@dataclass(frozen=True, slots=True)
class Gather(Operation):
    """Select slices from input along axis (NumPy take / embedding lookup)."""

    axis: int

    @property
    def family(self) -> str:
        return "gather"

    @property
    def input_ports(self) -> tuple[PortSpec, ...]:
        return (PortSpec("input"), PortSpec("index"))

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
        input_meta, _index = inputs
        return (reduced_gradient_metadata(input_meta),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return (GRAD_INPUT,)
        return ()

    @property
    def backward(self) -> BackwardSpec:
        return BackwardSpec(
            supported=True,
            saved_for_backward=("index",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
    ) -> tuple[ValueMetadata, ...]:
        input_meta, index_meta = inputs
        if not input_meta.shape:
            raise ValueError("gather requires a ranked input tensor")
        axis = _normalize_axis(self.axis, len(input_meta.shape))
        output_shape = list(input_meta.shape)
        output_shape[axis : axis + 1] = list(index_meta.shape)
        return (
            ValueMetadata(
                tuple(output_shape),
                input_meta.semantic_type,
                requires_grad=input_meta.requires_grad,
            ),
        )

    def saved_for_backward(
        self,
        inputs: tuple[ValueMetadata, ...],
        parameters: tuple[ValueMetadata, ...] = (),
        outputs: tuple[ValueMetadata, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("index",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        return 0

    def backward_flops(self, context: EstimationContext) -> int:
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(result))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(persist_only_events(port_name))
        return tuple(events)


__all__ = ["Gather"]
