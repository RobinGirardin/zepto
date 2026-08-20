"""Eager graph composition context and user-facing module base class."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .errors import GraphCompositionError, MetadataMismatchError, PortArityError
from .graph import StructuralGraph, StructuralGraphBuilder
from .metadata import TensorMetadata, metadata_compatible
from .ports import PortSpec
from .provenance import Provenance
from .ids import ParameterId, TensorId
from .operation import Operation


@dataclass(frozen=True, slots=True)
class GraphTensor:
    """Symbolic tensor handle passed through user-defined module methods."""

    id: TensorId
    metadata: TensorMetadata


class Module:
    """User-facing composition and provenance boundary."""

    def __init__(self) -> None:
        """Create an empty module composition boundary."""
        self._parameters: dict[str, ParameterId] = {}
        self._modules: dict[str, Module] = {}

    def register_parameter(self, name: str, parameter: ParameterId) -> None:
        """Register a graph parameter under a module-local name.

        Args:
            name: Unique name used in module parameter paths.
            parameter: Graph-owned parameter identity.
        """
        if not name or name in self._parameters or name in self._modules:
            raise ValueError(f"Invalid or duplicate module member {name!r}")
        self._parameters[name] = parameter

    def register_module(self, name: str, module: Module) -> None:
        """Register a nested module under a module-local name.

        Args:
            name: Unique name used in nested provenance paths.
            module: Nested module to register.
        """
        if not name or name in self._parameters or name in self._modules:
            raise ValueError(f"Invalid or duplicate module member {name!r}")
        self._modules[name] = module

    def __call__(self, *inputs: GraphTensor) -> GraphTensor | tuple[GraphTensor, ...]:
        """Compose this module in the active graph context.

        Args:
            *inputs: Symbolic graph inputs.

        Returns:
            One graph output or a tuple of graph outputs.

        Raises:
            GraphCompositionError: If no composition context is active.
        """
        context = GraphCompositionContext.current()
        if context is None:
            raise GraphCompositionError(
                "Module calls require an active GraphCompositionContext"
            )
        return context.call_module(self, inputs)

    def forward(
        self, *inputs: GraphTensor
    ) -> GraphTensor | tuple[GraphTensor, ...]:
        """Define the module's eager structural composition.

        Subclasses override this method to call functional operations or
        nested modules.
        """
        raise NotImplementedError

    def named_parameters(self, prefix: str = "") -> Iterable[tuple[str, ParameterId]]:
        """Iterate over registered parameters using dotted paths.

        Args:
            prefix: Path prefix applied to emitted names.

        Yields:
            ``(name, parameter_id)`` pairs in registration order.
        """
        for name, parameter in self._parameters.items():
            yield f"{prefix}{name}", parameter
        for name, module in self._modules.items():
            yield from module.named_parameters(f"{prefix}{name}.")


class GraphCompositionContext:
    """Active composition context that owns one graph builder."""

    _active: GraphCompositionContext | None = None

    def __init__(self, builder: StructuralGraphBuilder | None = None) -> None:
        """Create a context around a builder.

        Args:
            builder: Optional existing builder to populate.
        """
        self.builder = builder or StructuralGraphBuilder()
        self._module_path: tuple[str, ...] = ()
        self._instance_counts: dict[tuple[str, str], int] = {}
        self._previous: GraphCompositionContext | None = None

    def __enter__(self) -> "GraphCompositionContext":
        """Make this context the active composition context."""
        if GraphCompositionContext._active is not None:
            raise GraphCompositionError("A graph composition context is already active")
        self._previous = GraphCompositionContext._active
        GraphCompositionContext._active = self
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Restore the previously active composition context."""
        GraphCompositionContext._active = self._previous

    @classmethod
    def current(cls) -> GraphCompositionContext | None:
        """Return the currently active composition context, if any."""
        return cls._active

    def input(self, metadata: TensorMetadata) -> GraphTensor:
        """Declare and return a graph input handle."""
        tensor_id = self.builder.add_input(metadata)
        return GraphTensor(tensor_id, metadata)

    def parameter(
        self, metadata: TensorMetadata, *, requires_grad: bool = True
    ) -> ParameterId:
        """Declare a parameter in the owned graph."""
        return self.builder.add_parameter(metadata, requires_grad=requires_grad)

    def apply(
        self,
        operation: Operation,
        *inputs: GraphTensor,
        parameters: tuple[ParameterId, ...] = (),
        module_path: tuple[str, ...] | None = None,
        component_type: str | None = None,
        source_label: str | None = None,
    ) -> GraphTensor | tuple[GraphTensor, ...]:
        """Apply an operation specification and record its graph node.

        Args:
            operation: Structural operation declaration.
            *inputs: Symbolic input values.
            parameters: Parameter identities referenced by the operation.
            module_path: Optional provenance path override.
            component_type: Optional reportable component type.
            source_label: Optional source-level label.

        Returns:
            One graph output or a tuple of graph outputs.

        Raises:
            MetadataMismatchError: If a declared port contract is not met.
            PortArityError: If port, input, or inferred output counts differ.
        """
        input_metadata = tuple(value.metadata for value in inputs)
        result = operation.infer(input_metadata)
        output_metadata = result.outputs
        input_ports, output_ports = self._bind_ports(
            operation,
            input_metadata,
            output_metadata,
        )
        key = (operation.family, ".".join(module_path or self._module_path))
        instance = self._instance_counts.get(key, 0)
        self._instance_counts[key] = instance + 1
        provenance = Provenance(
            module_path=module_path or self._module_path,
            component_type=component_type,
            operation_family=operation.family,
            operation_instance=instance,
            source_label=source_label,
        )
        output_ids = self.builder.add_operation(
            operation_family=operation.family,
            input_ports=input_ports,
            output_ports=output_ports,
            input_tensors=tuple(value.id for value in inputs),
            output_metadata=output_metadata,
            parameter_ids=parameters,
            provenance=provenance,
            operation=operation,
            result=result,
        )
        values = tuple(
            GraphTensor(tensor_id, metadata)
            for tensor_id, metadata in zip(output_ids, output_metadata, strict=True)
        )
        return values[0] if len(values) == 1 else values

    def _bind_ports(
        self,
        operation: Operation,
        input_metadata: tuple[TensorMetadata, ...],
        output_metadata: tuple[TensorMetadata, ...],
    ) -> tuple[tuple[PortSpec, ...], tuple[PortSpec, ...]]:
        """Validate and bind actual metadata onto an operation's ports.

        The operation declaration remains reusable and unchanged. The returned
        port declarations are immutable copies containing the actual metadata used
        by this graph invocation.

        Args:
            operation: Unbound operation declaration.
            input_metadata: Metadata from the supplied graph inputs.
            output_metadata: Metadata produced by the inference rule.

        Returns:
            Bound input and output ports in declaration order.

        Raises:
            MetadataMismatchError: If a declared contract is incompatible.
            PortArityError: If declaration and metadata counts differ.
        """
        if len(operation.input_ports) != len(input_metadata):
            raise PortArityError(
                f"{operation.family!r} expects "
                f"{len(operation.input_ports)} inputs, got {len(input_metadata)}"
            )
        if len(operation.output_ports) != len(output_metadata):
            raise PortArityError(
                f"{operation.family!r} declares "
                f"{len(operation.output_ports)} outputs, inferred "
                f"{len(output_metadata)}"
            )

        bound_inputs = tuple(
            self._bind_port(
                operation,
                port,
                metadata,
                direction="input",
            )
            for port, metadata in zip(
                operation.input_ports,
                input_metadata,
                strict=True,
            )
        )
        bound_outputs = tuple(
            self._bind_port(
                operation,
                port,
                metadata,
                direction="output",
            )
            for port, metadata in zip(
                operation.output_ports,
                output_metadata,
                strict=True,
            )
        )
        return bound_inputs, bound_outputs

    @staticmethod
    def _bind_port(
        operation: Operation,
        port: PortSpec,
        actual: TensorMetadata,
        *,
        direction: str,
    ) -> PortSpec:
        """Validate one port and return it with resolved metadata."""
        if port.metadata is not None and not metadata_compatible(
            port.metadata,
            actual,
        ):
            raise MetadataMismatchError(
                f"{operation.family!r} {direction} port {port.name!r} "
                f"expects {port.metadata!r}, got {actual!r}"
            )
        return replace(port, metadata=actual)

    def call_module(
        self, module: Module, inputs: tuple[GraphTensor, ...]
    ) -> GraphTensor | tuple[GraphTensor, ...]:
        """Invoke a module while recording its provenance path."""
        previous_path = self._module_path
        name = module.__class__.__name__
        self._module_path = (*previous_path, name)
        try:
            return module.forward(*inputs)
        finally:
            self._module_path = previous_path

    def mark_output(self, value: GraphTensor) -> None:
        """Expose a graph value as a graph output."""
        self.builder.mark_output(value.id)

    def build(self) -> StructuralGraph:
        """Finalize and return the immutable structural graph."""
        return self.builder.build()


def build_graph(
    module: Module,
    input_metadata: tuple[TensorMetadata, ...],
) -> StructuralGraph:
    """Build a structural graph by eagerly composing one module.

    Args:
        module: Root module whose ``forward`` method is composed.
        input_metadata: Metadata for root module inputs in positional order.

    Returns:
        The finalized immutable structural graph.
    """
    with GraphCompositionContext() as context:
        inputs = tuple(context.input(metadata) for metadata in input_metadata)
        outputs = module(*inputs)
        values = outputs if isinstance(outputs, tuple) else (outputs,)
        for value in values:
            context.mark_output(value)
        return context.build()
