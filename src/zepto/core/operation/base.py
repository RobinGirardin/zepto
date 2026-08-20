"""Base interfaces and validation for backend-neutral operations."""

from abc import ABC, abstractmethod
from dataclasses import replace

from ..metadata import TensorMetadata
from ..ports import PortSpec
from .records import (
    AliasSpec,
    BackwardSpec,
    EstimationContext,
    OperationError,
    OperationResult,
    ResourceEvent,
)


class SemanticOperation(ABC):
    """Describe the semantic behavior of a structural operation."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Return the stable operation family name."""
        ...

    @property
    @abstractmethod
    def input_ports(self) -> tuple[PortSpec, ...]:
        """Return ordered input port declarations.

        Port metadata may be ``None`` at declaration time. The composition
        context binds the concrete input tensor metadata while building a
        structural graph.
        """
        ...

    @property
    @abstractmethod
    def output_ports(self) -> tuple[PortSpec, ...]:
        """Return ordered output port declarations.

        Port metadata may be ``None`` at declaration time. The composition
        context binds the inferred output metadata while building a
        structural graph.
        """
        ...

    @abstractmethod
    def infer_outputs(
        self, inputs: tuple[TensorMetadata, ...]
    ) -> tuple[TensorMetadata, ...]:
        """Infer the output tensor metadata for one operation invocation."""
        ...

    def output_aliases(self) -> tuple[AliasSpec | None, ...]:
        """Return per-output alias declarations, in output-port order.

        The default empty tuple means every output is materialized in fresh
        storage. View and copy operations override this to declare their
        storage relationship to an input port.
        """
        return ()

    @property
    @abstractmethod
    def backward(self) -> BackwardSpec:
        """Return the operation's backend-neutral backward requirements."""
        return BackwardSpec()

    @abstractmethod
    def saved_for_backward(
        self,
        inputs: tuple[TensorMetadata, ...],
        outputs: tuple[TensorMetadata, ...],
    ) -> tuple[str, ...]:
        """Select the port names saved for this concrete invocation.

        ``BackwardSpec.saved_for_backward`` declares the names an operation
        is *allowed* to save. This method selects the subset actually needed
        for one invocation, based on which inputs participate in gradients.
        Operations whose backward pass needs no saved forward values return
        an empty tuple.
        """
        ...


class EstimationOperation(ABC):
    """Describe theoretical cost and resource behavior."""

    @abstractmethod
    def forward_flops(self, context: EstimationContext) -> int:
        """Return the theoretical forward FLOP count.

        The context carries resolved metadata for every input and output
        port, so estimators look up outputs by port name (for example
        ``context.metadata_for("output")``) just like inputs.
        """
        ...

    def backward_flops(self, context: EstimationContext) -> int:
        """Return the theoretical backward FLOP count."""
        return 0

    @abstractmethod
    def resource_events(
        self, context: EstimationContext, result: OperationResult
    ) -> tuple[ResourceEvent, ...]:
        """Return ordered logical resource-lifetime events."""
        ...


class Operation(SemanticOperation, EstimationOperation, ABC):
    """Immutable-by-convention declaration of one semantic operation family."""

    def validate_declaration(self) -> None:
        """Validate family, ports, and backward declarations."""
        if not isinstance(self.family, str) or not self.family:
            raise OperationError("Operation family cannot be empty")
        inputs = self.input_ports
        outputs = self.output_ports
        if not isinstance(inputs, tuple) or not isinstance(outputs, tuple):
            raise OperationError("Operation ports must be tuples")
        ports = (*inputs, *outputs)
        if not all(isinstance(port, PortSpec) and port.name for port in ports):
            raise OperationError("Operations require named PortSpec declarations")
        names = [port.name for port in ports]
        if len(names) != len(set(names)):
            raise OperationError("Operation port names must be unique")
        backward = self.backward
        if not isinstance(backward, BackwardSpec):
            raise OperationError("backward must return BackwardSpec")
        known = set(names)
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

    def infer_result(self, inputs: tuple[TensorMetadata, ...]) -> OperationResult:
        """Validate inputs, compose one result, and validate that result.

        The result is constructed exactly once from the operation's hooks:
        ``infer_outputs()`` supplies the output metadata, ``output_aliases()``
        supplies the storage relationships, and ``saved_for_backward()``
        selects the invocation-specific saved values.
        """
        self.validate_declaration()
        if len(inputs) != len(self.input_ports):
            raise OperationError(
                f"{self.family!r} expects {len(self.input_ports)} inputs, got {len(inputs)}"
            )
        if not all(isinstance(value, TensorMetadata) for value in inputs):
            raise OperationError("Operation inputs must be TensorMetadata")
        outputs = self.infer_outputs(inputs)
        if not isinstance(outputs, tuple):
            raise OperationError("infer_outputs must return a tuple")
        result = OperationResult(
            outputs=outputs,
            aliases=self.output_aliases(),
            saved_for_backward=self.saved_for_backward(inputs, outputs),
        )
        self.validate_result(inputs, result)
        return result

    def validate_result(
        self, inputs: tuple[TensorMetadata, ...], result: OperationResult
    ) -> None:
        """Validate metadata, aliases, backward state, costs, and events."""
        if not isinstance(result, OperationResult):
            raise OperationError("Inference must return OperationResult")
        if len(result.outputs) != len(self.output_ports):
            raise OperationError(
                f"{self.family!r} inferred {len(result.outputs)} outputs, "
                f"declares {len(self.output_ports)}"
            )
        if not all(isinstance(value, TensorMetadata) for value in result.outputs):
            raise OperationError("Operation outputs must be TensorMetadata")
        aliases = result.aliases or (None,) * len(result.outputs)
        if len(aliases) != len(result.outputs):
            raise OperationError("Alias declarations must match output arity")
        input_names = {port.name for port in self.input_ports}
        for index, alias in enumerate(aliases):
            if alias is None:
                continue
            if alias.source_port not in input_names:
                raise OperationError(f"Unknown alias source port {alias.source_port!r}")
            if alias.materialization.value == "view":
                source = inputs[
                    next(i for i, port in enumerate(self.input_ports)
                         if port.name == alias.source_port)
                ]
                source_size = _numel(source)
                output_size = _numel(result.outputs[index])
                if source_size is not None and output_size is not None and source_size != output_size:
                    raise OperationError("View aliases must preserve tensor shape")
        selection = self.saved_for_backward(inputs, result.outputs)
        if not isinstance(selection, tuple) or not all(
            isinstance(name, str) and name for name in selection
        ):
            raise OperationError(
                "saved_for_backward must return a tuple of port names"
            )
        if len(selection) != len(set(selection)):
            raise OperationError("saved_for_backward selection must be unique")
        if not self.backward.supported and selection:
            raise OperationError(
                "Unsupported backward operation selects saved values"
            )
        if result.saved_for_backward != selection:
            raise OperationError(
                "Result saved values do not match the operation's "
                "saved_for_backward selection"
            )
        saved_names = set(result.saved_for_backward)
        if not saved_names.issubset(set(self.backward.saved_for_backward)):
            raise OperationError("Result saves an undeclared backward value")
        if not saved_names.issubset({port.name for port in (*self.input_ports, *self.output_ports)}):
            raise OperationError("Result saves an unknown operation port")
        context = EstimationContext(
            port_metadata=(
                *(
                    (port.name, metadata)
                    for port, metadata in zip(self.input_ports, inputs, strict=True)
                ),
                *(
                    (port.name, metadata)
                    for port, metadata in zip(
                        self.output_ports, result.outputs, strict=True
                    )
                ),
            )
        )
        for flops in (
            self.forward_flops(context),
            self.backward_flops(replace(context, phase="backward")),
        ):
            if not isinstance(flops, int) or flops < 0:
                raise OperationError("FLOP counts must be non-negative integers")
        events = self.resource_events(context, result)
        if not isinstance(events, tuple) or not all(isinstance(event, ResourceEvent) for event in events):
            raise OperationError("Resource events must be an ordered tuple")


def _numel(metadata: TensorMetadata) -> int:
    """Return the element count for a concrete shape."""
    result = 1
    for dimension in metadata.shape:
        result *= dimension
    return result
