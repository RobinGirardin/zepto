"""Base interfaces and validation for backend-neutral operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from zepto.compose.values import Tensor
from ..ports import Port
from .records import (
    AliasSpec,
    BackwardSpec,
    EstimationContext,
    InferenceBundle,
    OperationError,
    OperationResult,
    ResourceEvent,
)
from .validation import DeclarationValidator, InvocationValidator

if TYPE_CHECKING:
    from zepto.compose.values import Parameter

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
    def parameter_ports(self) -> tuple[Port, ...]:
        """Return ordered parameter port declarations."""
        return ()

    @property
    @abstractmethod
    def input_ports(self) -> tuple[Port, ...]:
        """Return ordered input port declarations.

        Port contracts may be ``None`` at declaration time. The composition
        context checks supplied tensors against any declared contract.
        """
        ...

    @property
    @abstractmethod
    def output_ports(self) -> tuple[Port, ...]:
        """Return ordered output port declarations.

        Port contracts may be ``None`` at declaration time. The composition
        context checks inferred output tensors against any declared contract.
        """
        ...

    @abstractmethod
    def infer_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        """Infer the output tensors for one operation invocation."""
        ...

    def output_aliases(self) -> tuple[AliasSpec | None, ...]:
        """Return per-output alias declarations, in output-port order.

        The default empty tuple means every output is materialized in fresh
        storage. View and copy operations override this to declare their
        storage relationship to an input port.
        """
        return ()

    def auxiliary_ports(self) -> tuple[Port, ...]:
        """Return internal tensor ports not exposed as public functional outputs."""
        return ()

    def infer_auxiliary_outputs(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[Tensor, ...]:
        """Return auxiliary tensors in ``auxiliary_ports`` order."""
        return ()

    def active_auxiliary_ports(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
    ) -> tuple[str, ...]:
        """Return the subset of auxiliary ports live for this invocation."""
        return ()

    def auxiliary_aliases(self) -> tuple[AliasSpec | None, ...]:
        """Return per-auxiliary alias declarations (default: fresh storage)."""
        return ()

    @property
    @abstractmethod
    def backward(self) -> BackwardSpec:
        """Return the operation's backend-neutral backward requirements."""
        return BackwardSpec()

    @abstractmethod
    def saved_for_backward(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
        outputs: tuple[Tensor, ...] = (),
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

        The context carries resolved values for every input and output
        port, so estimators look up outputs by port name (for example
        ``context.tensor_for("output")``) just like inputs.
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

    def infer_result(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...] = (),
    ) -> InferenceBundle:
        """Validate inputs, infer tensors, and compose one semantics-only result.

        ``infer_outputs()`` supplies the output tensors, ``output_aliases()``
        supplies the storage relationships, and ``saved_for_backward()``
        selects the invocation-specific saved values.
        """
        self.validate_declaration()
        _INVOCATION_VALIDATOR.validate_inputs(self, inputs)
        _INVOCATION_VALIDATOR.validate_parameters(self, parameters)
        outputs = self.infer_outputs(inputs, parameters)
        if not isinstance(outputs, tuple):
            raise OperationError("infer_outputs must return a tuple")
        auxiliary_outputs = self.infer_auxiliary_outputs(inputs, parameters, outputs)
        result = OperationResult(
            aliases=self.output_aliases(),
            auxiliary_aliases=self.auxiliary_aliases(),
            saved_for_backward=self.saved_for_backward(inputs, parameters, outputs),
            active_auxiliary_ports=self.active_auxiliary_ports(
                inputs, parameters, outputs
            ),
        )
        self.validate_result(inputs, parameters, outputs, auxiliary_outputs, result)
        return InferenceBundle(outputs, auxiliary_outputs, result)

    def validate_result(
        self,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        auxiliary_outputs: tuple[Tensor, ...],
        result: OperationResult,
    ) -> None:
        """Validate tensors, aliases, and backward state."""
        _INVOCATION_VALIDATOR.validate_result(
            self, inputs, parameters, outputs, auxiliary_outputs, result
        )

    def __call__(
        self,
        *inputs: Tensor,
        parameters: tuple[Parameter, ...] = (),
    ) -> Tensor | tuple[Tensor, ...]:
        """Record this operation in the active graph composition context.

        Args:
            *inputs: Symbolic input tensors already present in the graph.
            parameters: Parameter handles referenced by the operation.

        Returns:
            One graph output or a tuple of graph outputs.

        Raises:
            ComposeError: If no composition context is active.
        """
        from zepto.compose.context import Compose
        from zepto.graph.errors import ComposeError

        context = Compose.current()
        if context is None:
            raise ComposeError("Operation calls require an active Compose context")
        return context._apply(self, *inputs, parameters=parameters)
