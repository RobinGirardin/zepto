"""Dedicated validators for operation declarations and invocations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..metadata import TensorMetadata
from ..ports import PortSpec, ValueKind
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
        outputs = operation.output_ports
        auxiliary = operation.auxiliary_ports()
        if not isinstance(inputs, tuple) or not isinstance(outputs, tuple):
            raise OperationError("Operation ports must be tuples")
        ports = (*inputs, *outputs, *auxiliary)
        if not all(isinstance(port, PortSpec) and port.name for port in ports):
            raise OperationError("Operations require named PortSpec declarations")
        names = [port.name for port in ports]
        if len(names) != len(set(names)):
            raise OperationError("Operation port names must be unique")

    def validate_backward(self, operation: Operation) -> None:
        backward = operation.backward
        if not isinstance(backward, BackwardSpec):
            raise OperationError("backward must return BackwardSpec")
        known = {
            port.name
            for port in (
                *operation.input_ports,
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
        inputs: tuple[TensorMetadata, ...],
    ) -> None:
        if len(inputs) != len(operation.input_ports):
            raise OperationError(
                f"{operation.family!r} expects {len(operation.input_ports)} inputs, "
                f"got {len(inputs)}"
            )
        if not all(isinstance(value, TensorMetadata) for value in inputs):
            raise OperationError("Operation inputs must be TensorMetadata")

    def validate_result(
        self,
        operation: Operation,
        inputs: tuple[TensorMetadata, ...],
        result: OperationResult,
    ) -> None:
        self.validate_metadata(operation, result)
        self.validate_aliases(operation, inputs, result)
        self.validate_saved_state(operation, inputs, result)
        self.validate_auxiliary_state(operation, inputs, result)

    def validate_metadata(self, operation: Operation, result: OperationResult) -> None:
        if not isinstance(result, OperationResult):
            raise OperationError("Inference must return OperationResult")
        if len(result.outputs) != len(operation.output_ports):
            raise OperationError(
                f"{operation.family!r} inferred {len(result.outputs)} outputs, "
                f"declares {len(operation.output_ports)}"
            )
        if not all(isinstance(value, TensorMetadata) for value in result.outputs):
            raise OperationError("Operation outputs must be TensorMetadata")

    def validate_aliases(
        self,
        operation: Operation,
        inputs: tuple[TensorMetadata, ...],
        result: OperationResult,
    ) -> None:
        aliases = result.aliases or (None,) * len(result.outputs)
        if len(aliases) != len(result.outputs):
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
                output_size = _numel(result.outputs[index])
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
        inputs: tuple[TensorMetadata, ...],
        result: OperationResult,
    ) -> None:
        selection = operation.saved_for_backward(inputs, result.outputs)
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
        if not saved_names.issubset(
            {port.name for port in (*operation.input_ports, *operation.output_ports)}
        ):
            raise OperationError("Result saves an unknown operation port")

    def validate_auxiliary_state(
        self,
        operation: Operation,
        inputs: tuple[TensorMetadata, ...],
        result: OperationResult,
    ) -> None:
        aux_ports = operation.auxiliary_ports()
        if len(result.auxiliary_outputs) != len(aux_ports):
            raise OperationError(
                f"{operation.family!r} inferred {len(result.auxiliary_outputs)} "
                f"auxiliary outputs, declares {len(aux_ports)}"
            )
        allowed = {port.name for port in aux_ports}
        if not set(result.active_auxiliary_ports).issubset(allowed):
            raise OperationError("Result activates an undeclared auxiliary port")
        gradient_outputs = set(operation.backward.gradient_outputs)
        input_by_name = {
            port.name: metadata
            for port, metadata in zip(operation.input_ports, inputs, strict=True)
        }
        for port, metadata in zip(aux_ports, result.auxiliary_outputs, strict=True):
            if port.value_kind is not ValueKind.GRADIENT:
                continue
            if port.name.endswith("_unreduced"):
                continue
            operand_name = port.name.removeprefix("grad_")
            if operand_name not in gradient_outputs:
                continue
            operand = input_by_name.get(operand_name)
            if operand is not None and metadata.shape != operand.shape:
                raise OperationError(
                    f"Gradient auxiliary port {port.name!r} shape "
                    f"{metadata.shape} must match operand shape {operand.shape}"
                )


def _numel(metadata: TensorMetadata) -> int:
    """Return the element count for a concrete shape."""
    result = 1
    for dimension in metadata.shape:
        result *= dimension
    return result
