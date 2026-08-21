"""Base interfaces and validation for backend-neutral operations."""

from abc import ABC, abstractmethod

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
from .validation import DeclarationValidator, InvocationValidator

_DECLARATION_VALIDATOR = DeclarationValidator()
_INVOCATION_VALIDATOR = InvocationValidator()


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
        _DECLARATION_VALIDATOR.validate(self)

    def infer_result(self, inputs: tuple[TensorMetadata, ...]) -> OperationResult:
        """Validate inputs, compose one result, and validate that result.

        The result is constructed exactly once from the operation's hooks:
        ``infer_outputs()`` supplies the output metadata, ``output_aliases()``
        supplies the storage relationships, and ``saved_for_backward()``
        selects the invocation-specific saved values.
        """
        self.validate_declaration()
        _INVOCATION_VALIDATOR.validate_inputs(self, inputs)
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
        """Validate metadata, aliases, and backward state."""
        _INVOCATION_VALIDATOR.validate_result(self, inputs, result)
