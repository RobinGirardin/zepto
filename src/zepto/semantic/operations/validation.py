"""Dedicated validators for operation declarations and invocations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zepto.compose.values import Tensor
from ..ports import Port, ValueKind
from .records import (
    BackwardSpec,
    Materialization,
    OperationError,
    OperationResult,
)

if TYPE_CHECKING:
    from .base import Operation


class DeclarationValidator:
    """Validate an operation's static declaration."""

    def validate(self, operation: Operation) -> None:
        self.validate_family(operation)
        self.validate_ports(operation)
        self.validate_backward(operation)

    def validate_family(self, operation: Operation) -> None:
        if not isinstance(operation.family, str) or not operation.family:
            raise OperationError("Operation family cannot be empty")

    def validate_ports(self, operation: Operation) -> None:
        inputs = operation.input_ports
        parameters = operation.parameter_ports
        outputs = operation.output_ports
        auxiliary = operation.auxiliary_ports()
        if not isinstance(inputs, tuple) or not isinstance(outputs, tuple):
            raise OperationError("Operation ports must be tuples")
        ports = (*inputs, *parameters, *outputs, *auxiliary)
        if not all(isinstance(port, Port) and port.name for port in ports):
            raise OperationError("Operations require named Port declarations")
        names = [port.name for port in ports]
        if len(names) != len(set(names)):
            raise OperationError("Operation port names must be unique")
        for port in parameters:
            if port.value_kind is not ValueKind.PARAMETER:
                raise OperationError(
                    f"Parameter port {port.name!r} must use ValueKind.PARAMETER"
                )

    def validate_backward(self, operation: Operation) -> None:
        backward = operation.backward
        if not isinstance(backward, BackwardSpec):
            raise OperationError("backward must return BackwardSpec")
        known = {
            port.name
            for port in (
                *operation.input_ports,
                *operation.parameter_ports,
                *operation.output_ports,
                *operation.auxiliary_ports(),
            )
        }
        for reference in (
            *backward.saved_for_backward,
            *backward.gradient_inputs,
            *backward.gradient_outputs,
        ):
            if reference not in known:
                raise OperationError(f"Backward references unknown port {reference!r}")
        if not backward.supported and (
            backward.saved_for_backward
            or backward.gradient_inputs
            or backward.gradient_outputs
        ):
            raise OperationError("Unsupported backward operation declares requirements")


class InvocationValidator:
    """Validate one concrete invocation's inputs and inferred result."""

    def validate_inputs(
        self,
        operation: Operation,
        inputs: tuple[Tensor, ...],
    ) -> None:
        if len(inputs) != len(operation.input_ports):
            raise OperationError(
                f"{operation.family!r} expects {len(operation.input_ports)} inputs, "
                f"got {len(inputs)}"
            )
        if not all(isinstance(value, Tensor) for value in inputs):
            raise OperationError("Operation inputs must be Tensor")

    def validate_parameters(
        self,
        operation: Operation,
        parameters: tuple[Tensor, ...],
    ) -> None:
        if len(parameters) != len(operation.parameter_ports):
            raise OperationError(
                f"{operation.family!r} expects "
                f"{len(operation.parameter_ports)} parameters, got {len(parameters)}"
            )
        if not all(isinstance(value, Tensor) for value in parameters):
            raise OperationError("Operation parameters must be Tensor")

    def validate_result(
        self,
        operation: Operation,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        auxiliary_outputs: tuple[Tensor, ...],
        result: OperationResult,
    ) -> None:
        self.validate_outputs(operation, outputs, result)
        self.validate_aliases(operation, inputs, outputs, result)
        self.validate_saved_state(operation, inputs, parameters, outputs, result)
        self.validate_auxiliary_state(operation, inputs, auxiliary_outputs, result)

    def validate_outputs(
        self,
        operation: Operation,
        outputs: tuple[Tensor, ...],
        result: OperationResult,
    ) -> None:
        if not isinstance(result, OperationResult):
            raise OperationError("Inference must return OperationResult")
        if len(outputs) != len(operation.output_ports):
            raise OperationError(
                f"{operation.family!r} inferred {len(outputs)} outputs, "
                f"declares {len(operation.output_ports)}"
            )
        if not all(isinstance(value, Tensor) for value in outputs):
            raise OperationError("Operation outputs must be Tensor")
        if result.aliases and len(result.aliases) != len(operation.output_ports):
            raise OperationError("Alias declarations must match output arity")

    def validate_aliases(
        self,
        operation: Operation,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        result: OperationResult,
    ) -> None:
        aliases = result.aliases or (None,) * len(outputs)
        if len(aliases) != len(outputs):
            raise OperationError("Alias declarations must match output arity")
        input_names = {port.name for port in operation.input_ports}
        for index, alias in enumerate(aliases):
            if alias is None:
                continue
            if alias.source_port not in input_names:
                raise OperationError(f"Unknown alias source port {alias.source_port!r}")
            if alias.materialization is Materialization.VIEW:
                source = inputs[
                    next(
                        i
                        for i, port in enumerate(operation.input_ports)
                        if port.name == alias.source_port
                    )
                ]
                source_size = _numel(source)
                output_size = _numel(outputs[index])
                if (
                    source_size is not None
                    and output_size is not None
                    and source_size != output_size
                ):
                    raise OperationError("View aliases must preserve tensor shape")
            elif alias.materialization is not Materialization.CONTIGUOUS_COPY:
                raise OperationError(
                    f"Unknown materialization {alias.materialization!r}"
                )

    def validate_saved_state(
        self,
        operation: Operation,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        result: OperationResult,
    ) -> None:
        selection = operation.saved_for_backward(inputs, parameters, outputs)
        if not isinstance(selection, tuple) or not all(
            isinstance(name, str) and name for name in selection
        ):
            raise OperationError(
                "saved_for_backward must return a tuple of port names"
            )
        if len(selection) != len(set(selection)):
            raise OperationError("saved_for_backward selection must be unique")
        if not operation.backward.supported and selection:
            raise OperationError(
                "Unsupported backward operation selects saved values"
            )
        if result.saved_for_backward != selection:
            raise OperationError(
                "Result saved values do not match the operation's "
                "saved_for_backward selection"
            )
        saved_names = set(result.saved_for_backward)
        if not saved_names.issubset(set(operation.backward.saved_for_backward)):
            raise OperationError("Result saves an undeclared backward value")
        known_ports = {
            port.name
            for port in (
                *operation.input_ports,
                *operation.parameter_ports,
                *operation.output_ports,
            )
        }
        if not saved_names.issubset(known_ports):
            raise OperationError("Result saves an unknown operation port")

    def validate_auxiliary_state(
        self,
        operation: Operation,
        inputs: tuple[Tensor, ...],
        auxiliary_outputs: tuple[Tensor, ...],
        result: OperationResult,
    ) -> None:
        aux_ports = operation.auxiliary_ports()
        if len(auxiliary_outputs) != len(aux_ports):
            raise OperationError(
                f"{operation.family!r} inferred {len(auxiliary_outputs)} "
                f"auxiliary outputs, declares {len(aux_ports)}"
            )
        if result.auxiliary_aliases and len(result.auxiliary_aliases) != len(aux_ports):
            raise OperationError(
                "Auxiliary alias declarations must match auxiliary arity"
            )
        allowed = {port.name for port in aux_ports}
        if not set(result.active_auxiliary_ports).issubset(allowed):
            raise OperationError("Result activates an undeclared auxiliary port")
        gradient_outputs = set(operation.backward.gradient_outputs)
        input_by_name = {
            port.name: tensor
            for port, tensor in zip(operation.input_ports, inputs, strict=True)
        }
        for port, tensor in zip(aux_ports, auxiliary_outputs, strict=True):
            if port.value_kind is not ValueKind.GRADIENT:
                continue
            if port.name.endswith("_unreduced"):
                continue
            operand_name = port.name.removeprefix("grad_")
            if operand_name not in gradient_outputs:
                continue
            operand = input_by_name.get(operand_name)
            if operand is not None and tensor.shape != operand.shape:
                raise OperationError(
                    f"Gradient auxiliary port {port.name!r} shape "
                    f"{tensor.shape} must match operand shape {operand.shape}"
                )


def _numel(tensor: Tensor) -> int:
    """Return the element count for a concrete shape."""
    result = 1
    for dimension in tensor.shape:
        result *= dimension
    return result
