"""KV-head replication along one axis."""

from dataclasses import dataclass

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import includes_backward, allocate, activation_grad_events, numel, reduced_gradient_tensor
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
    def input_ports(self) -> tuple[Port, ...]:
        return (Port("input"),)

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return (Port("output"),)

    def auxiliary_ports(self) -> tuple[Port, ...]:
        return (Port(GRAD_INPUT, ValueKind.GRADIENT),)

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        return (reduced_gradient_tensor(inputs[0]),)

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
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
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        if self.n_rep < 1:
            raise ValueError("repeat_kv n_rep must be >= 1")
        (input_tensor,) = inputs
        if not input_tensor.shape:
            raise ValueError("repeat_kv requires a ranked input tensor")
        axis = _normalize_axis(self.axis, len(input_tensor.shape))
        shape = list(input_tensor.shape)
        shape[axis] *= self.n_rep
        return (
            Tensor(
                shape=tuple(shape),
                dtype=input_tensor.dtype,
                semantic_type=input_tensor.semantic_type,
                requires_grad=input_tensor.requires_grad,
                persistent=input_tensor.persistent,
            ),
        )

    def output_aliases(self) -> tuple[AliasSpec | None, ...]:
        if self.n_rep == 1:
            return (AliasSpec("input"),)
        return ()

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
        input_tensor = context.tensor_for("input")
        output = context.tensor_for("output")
        if input_tensor is None or output is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        if input_tensor.requires_grad and self.n_rep > 1:
            return numel(output) - numel(input_tensor)
        return 0

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        if self.n_rep == 1 and result.aliases:
            return (ResourceEvent(ResourceEventKind.ALIAS, "output:0"),)
        events = list(allocate(len(self.output_ports)))
        if not includes_backward(context.phase):
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["RepeatKV"]
