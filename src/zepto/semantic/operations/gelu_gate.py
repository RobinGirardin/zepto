"""Atomic GELU gate-branch operations for GeGLU graph discovery."""

from __future__ import annotations

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .base import Operation
from .helpers import allocate, activation_grad_events, numel, reduced_gradient_tensor
from .records import BackwardSpec, EstimationContext, OperationResult, ResourceEvent

GRAD_INPUT = "grad_input"

# Kernel-accurate per-element totals (GELU + no extra mul); see docs/kernel/geglu.md §4/§5.
_GELU_TANH_FWD = 12
_GELU_TANH_BWD = 20
_GELU_ERF_FWD = 8
_GELU_ERF_BWD = 17


class GeluTanhGate(Operation):
    """Atomic tanh-GELU on the gate branch (one graph node for ``region/geglu``)."""

    @property
    def family(self) -> str:
        return "gelu"

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
            saved_for_backward=("input",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        return inputs

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("input",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return _GELU_TANH_FWD * numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        input_tensor = context.tensor_for("input")
        if output is None or input_tensor is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        if not input_tensor.requires_grad:
            return 0
        return _GELU_TANH_BWD * numel(output)

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(activation_grad_events(port_name))
        return tuple(events)


class GeluErfGate(Operation):
    """Atomic exact-erf GELU on the gate branch (one graph node for ``region/geglu``)."""

    @property
    def family(self) -> str:
        return "gelu_erf"

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
            saved_for_backward=("input",),
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        return inputs

    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        if inputs[0].requires_grad:
            return ("input",)
        return ()

    def forward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        if output is None:
            raise ValueError("Estimation context must provide an 'output' port")
        return _GELU_ERF_FWD * numel(output)

    def backward_flops(self, context: EstimationContext) -> int:
        output = context.tensor_for("output")
        input_tensor = context.tensor_for("input")
        if output is None or input_tensor is None:
            raise ValueError(
                "Estimation context must provide 'input' and 'output' ports"
            )
        if not input_tensor.requires_grad:
            return 0
        return _GELU_ERF_BWD * numel(output)

    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        events = list(allocate(len(self.output_ports)))
        if context.phase != "backward":
            return tuple(events)
        for port_name in result.active_auxiliary_ports:
            events.extend(activation_grad_events(port_name))
        return tuple(events)


__all__ = ["GeluErfGate", "GeluTanhGate"]
