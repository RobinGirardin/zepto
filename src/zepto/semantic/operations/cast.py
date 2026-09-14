"""Elementwise dtype conversion operation declaration."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose.values import Tensor
from zepto.semantic.metadata import DType
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import allocate, activation_grad_events, reduced_gradient_tensor
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"


@dataclass(frozen=True, slots=True)
class Cast(Operation):
    """Convert an activation to an explicit element type.

    Models PyTorch ``tensor.to(dtype)`` as a materializing conversion: same
    shape, new ``dtype``, fresh storage (not a view). Used by HF-accurate
    eager paths such as RMSNorm fp32 variance accumulation.

    Backward passes the upstream gradient through a reverse cast to the
    input dtype. No forward values are saved.
    """

    to_dtype: DType

    @property
    def family(self) -> str:
        return "cast"

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
        if inputs[0].requires_grad:
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
        (input_tensor,) = inputs
        if self.to_dtype is DType.UNKNOWN:
            raise ValueError("cast requires a concrete to_dtype")
        if not input_tensor.shape:
            raise ValueError("cast requires a ranked input tensor")
        return (
            Tensor(
                shape=input_tensor.shape,
                dtype=self.to_dtype,
                semantic_type=input_tensor.semantic_type,
                requires_grad=input_tensor.requires_grad,
                persistent=input_tensor.persistent,
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
        self,
        context: EstimationContext,
        result: OperationResult,
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["Cast"]
