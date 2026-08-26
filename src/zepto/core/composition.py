"""Eager graph composition context and user-facing module base class."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import ClassVar, Iterable

from .errors import GraphCompositionError, MetadataMismatchError, PortArityError
from .graph import StructuralGraph, StructuralGraphBuilder
from .metadata import ValueMetadata, metadata_compatible
from .parameter import bound_parameter_metadata
from .ports import PortSpec
from .provenance import Provenance
from .ids import ParameterId, TensorId
from .operation import Operation


@dataclass(frozen=True, slots=True)
class GraphTensor:
    """Symbolic tensor handle passed through user-defined module methods."""

    id: TensorId
    metadata: ValueMetadata


@dataclass(frozen=True, slots=True)
class GraphParameter:
    """Symbolic parameter handle registered during module construction."""

    id: ParameterId
    metadata: ValueMetadata
    trainable: bool = True


class Module:
    """User-facing composition and provenance boundary."""

    module_kind: ClassVar[str | None] = None

    def __init__(self) -> None:
        """Create an empty module composition boundary."""
        self._parameters: dict[str, GraphParameter] = {}
        self._modules: dict[str, Module] = {}
        self._registered_name: str | None = None

    @property
    def component_type(self) -> str:
        """Stable reportable kind for provenance and region matching."""
        return self.module_kind or self.__class__.__name__

    def register_parameter(self, name: str, parameter: GraphParameter) -> None:
        """Register a graph parameter under a module-local name.

        Args:
            name: Unique name used in module parameter paths.
            parameter: Graph-owned parameter handle.
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
        module._registered_name = name

    def __setattr__(self, name: str, value: object) -> None:
        """Auto-register nested modules and parameters assigned as attributes."""
        if name in ("_parameters", "_modules", "_registered_name") or name.startswith(
            "_"
        ):
            object.__setattr__(self, name, value)
            return

        parameters = self.__dict__.get("_parameters")
        if parameters is not None:
            if isinstance(value, Module):
                self.register_module(name, value)
                object.__setattr__(self, name, value)
                return
            if isinstance(value, GraphParameter):
                self.register_parameter(name, value)
                object.__setattr__(self, name, value)
                return

        object.__setattr__(self, name, value)

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
        name = getattr(self, "_registered_name", None)
        return context.call_module(self, inputs, name=name)

    def forward(
        self, *inputs: GraphTensor
    ) -> GraphTensor | tuple[GraphTensor, ...]:
        """Define the module's eager structural composition.

        Subclasses override this method to call functional operations or
        nested modules.
        """
        raise NotImplementedError

    def named_parameters(
        self, prefix: str = ""
    ) -> Iterable[tuple[str, GraphParameter]]:
        """Iterate over registered parameters using dotted paths.

        Args:
            prefix: Path prefix applied to emitted names.

        Yields:
            ``(name, parameter)`` pairs in registration order.
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
        self._component_type: str | None = None
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

    def input(self, metadata: ValueMetadata) -> GraphTensor:
        """Declare and return a graph input handle."""
        tensor_id = self.builder.add_input(metadata)
        return GraphTensor(tensor_id, metadata)

    def parameter(
        self,
        metadata: ValueMetadata,
        *,
        trainable: bool = True,
    ) -> GraphParameter:
        """Declare a model parameter in the owned graph."""
        parameter_id = self.builder.add_parameter(metadata, trainable=trainable)
        return GraphParameter(parameter_id, metadata, trainable=trainable)

    def apply(
        self,
        operation: Operation,
        *inputs: GraphTensor,
        parameters: tuple[GraphParameter, ...] = (),
        module_path: tuple[str, ...] | None = None,
        component_type: str | None = None,
        source_label: str | None = None,
    ) -> GraphTensor | tuple[GraphTensor, ...]:
        """Apply an operation specification and record its graph node.

        Args:
            operation: Structural operation declaration.
            *inputs: Symbolic input values.
            parameters: Parameter handles referenced by the operation.
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
        parameter_metadata = tuple(
            bound_parameter_metadata(self.builder._parameters[param.id])
            for param in parameters
        )
        result = operation.infer_result(input_metadata, parameter_metadata)
        output_metadata = result.outputs
        input_ports, parameter_ports, output_ports = self._bind_ports(
            operation,
            input_metadata,
            parameter_metadata,
            output_metadata,
        )
        key = (operation.family, ".".join(module_path or self._module_path))
        instance = self._instance_counts.get(key, 0)
        self._instance_counts[key] = instance + 1
        provenance = Provenance(
            module_path=module_path or self._module_path,
            component_type=component_type or self._component_type,
            operation_family=operation.family,
            operation_instance=instance,
            source_label=source_label,
        )
        output_ids = self.builder.add_operation(
            operation_family=operation.family,
            input_ports=input_ports,
            parameter_ports=parameter_ports,
            output_ports=output_ports,
            input_tensors=tuple(value.id for value in inputs),
            parameter_ids=tuple(param.id for param in parameters),
            output_metadata=output_metadata,
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
        input_metadata: tuple[ValueMetadata, ...],
        parameter_metadata: tuple[ValueMetadata, ...],
        output_metadata: tuple[ValueMetadata, ...],
    ) -> tuple[tuple[PortSpec, ...], tuple[PortSpec, ...], tuple[PortSpec, ...]]:
        """Validate and bind actual metadata onto an operation's ports.

        The operation declaration remains reusable and unchanged. The returned
        port declarations are immutable copies containing the actual metadata used
        by this graph invocation.

        Args:
            operation: Unbound operation declaration.
            input_metadata: Metadata from the supplied graph inputs.
            parameter_metadata: Metadata from the supplied graph parameters.
            output_metadata: Metadata produced by the inference rule.

        Returns:
            Bound input, parameter, and output ports in declaration order.

        Raises:
            MetadataMismatchError: If a declared contract is incompatible.
            PortArityError: If declaration and metadata counts differ.
        """
        if len(operation.input_ports) != len(input_metadata):
            raise PortArityError(
                f"{operation.family!r} expects "
                f"{len(operation.input_ports)} inputs, got {len(input_metadata)}"
            )
        if len(operation.parameter_ports) != len(parameter_metadata):
            raise PortArityError(
                f"{operation.family!r} expects "
                f"{len(operation.parameter_ports)} parameters, "
                f"got {len(parameter_metadata)}"
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
        bound_parameters = tuple(
            self._bind_port(
                operation,
                port,
                metadata,
                direction="parameter",
            )
            for port, metadata in zip(
                operation.parameter_ports,
                parameter_metadata,
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
        return bound_inputs, bound_parameters, bound_outputs

    @staticmethod
    def _bind_port(
        operation: Operation,
        port: PortSpec,
        actual: ValueMetadata,
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
        self,
        module: Module,
        inputs: tuple[GraphTensor, ...],
        *,
        name: str | None = None,
    ) -> GraphTensor | tuple[GraphTensor, ...]:
        """Invoke a module while recording its provenance path."""
        previous_path = self._module_path
        previous_component = self._component_type
        segment = name or module.component_type
        self._module_path = (*previous_path, segment)
        self._component_type = module.component_type
        try:
            return module.forward(*inputs)
        finally:
            self._module_path = previous_path
            self._component_type = previous_component

    def mark_output(self, value: GraphTensor) -> None:
        """Expose a graph value as a graph output."""
        self.builder.mark_output(value.id)

    def build(self) -> StructuralGraph:
        """Finalize and return the immutable structural graph."""
        return self.builder.build()


def build_graph(
    module_or_factory: Module | Callable[[GraphCompositionContext], Module],
    input_metadata: tuple[ValueMetadata, ...],
) -> StructuralGraph:
    """Build a structural graph by eagerly composing one module.

    Args:
        module_or_factory: Root module or a factory that receives the active
            composition context and returns a module. Factories are required
            when module construction declares parameters.
        input_metadata: Metadata for root module inputs in positional order.

    Returns:
        The finalized immutable structural graph.
    """
    with GraphCompositionContext() as context:
        if isinstance(module_or_factory, Module):
            module = module_or_factory
        else:
            module = module_or_factory(context)
        inputs = tuple(context.input(metadata) for metadata in input_metadata)
        outputs = module(*inputs)
        values = outputs if isinstance(outputs, tuple) else (outputs,)
        for value in values:
            context.mark_output(value)
        return context.build()
