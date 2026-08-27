"""Eager graph composition context and user-facing module base class."""

from __future__ import annotations

from collections.abc import Callable
from typing import Iterable

from .values import Parameter, Tensor
from zepto.graph.errors import ComposeError, MetadataMismatchError, PortArityError
from zepto.graph.graph import Graph, GraphBuilder
from zepto.semantic.metadata import contract_satisfied
from zepto.semantic.ports import Port
from zepto.graph.provenance import Provenance
from zepto.semantic.operations import Operation


class Module:
    """User-facing composition and provenance boundary."""

    def __init__(self) -> None:
        """Create an empty module composition boundary."""
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}

    def register_parameter(self, name: str, parameter: Parameter) -> None:
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

    def __call__(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Compose this module in the active graph context.

        Args:
            *inputs: Symbolic graph inputs.

        Returns:
            One graph output or a tuple of graph outputs.

        Raises:
            ComposeError: If no composition context is active.
        """
        context = Compose.current()
        if context is None:
            raise ComposeError("Module calls require an active Compose context")
        return context.call_module(self, inputs)

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Define the module's eager structural composition.

        Subclasses override this method to call operations or nested modules.
        """
        raise NotImplementedError

    def named_parameters(
        self, prefix: str = ""
    ) -> Iterable[tuple[str, Parameter]]:
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


class Compose:
    """Active composition context that owns one graph builder."""

    _active: Compose | None = None

    def __init__(self, builder: GraphBuilder | None = None) -> None:
        """Create a context around a builder.

        Args:
            builder: Optional existing builder to populate.
        """
        self.builder = builder or GraphBuilder()
        self._module_path: tuple[str, ...] = ()
        self._instance_counts: dict[tuple[str, str], int] = {}
        self._previous: Compose | None = None

    def __enter__(self) -> Compose:
        """Make this context the active composition context."""
        if Compose._active is not None:
            raise ComposeError("A compose context is already active")
        self._previous = Compose._active
        Compose._active = self
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Restore the previously active composition context."""
        Compose._active = self._previous

    @classmethod
    def current(cls) -> Compose | None:
        """Return the currently active composition context, if any."""
        return cls._active

    def input(self, spec: Tensor) -> Tensor:
        """Declare and return a graph input handle."""
        if not isinstance(spec, Tensor):
            raise TypeError("Compose.input requires a Tensor")
        edge_id = self.builder.add_input(spec)
        return spec.registered(edge_id)

    def parameter(self, spec: Parameter) -> Parameter:
        """Declare a model parameter in the owned graph."""
        if not isinstance(spec, Parameter):
            raise TypeError("Compose.parameter requires a Parameter")
        parameter_id = self.builder.add_parameter(spec)
        return spec.registered(parameter_id)

    def _apply(
        self,
        operation: Operation,
        *inputs: Tensor,
        parameters: tuple[Parameter, ...] = (),
        module_path: tuple[str, ...] | None = None,
        component_type: str | None = None,
        source_label: str | None = None,
    ) -> Tensor | tuple[Tensor, ...]:
        """Record an operation node in the owned graph builder.

        This is an internal hook used by ``Operation.__call__``. Prefer calling
        operations directly during module composition.
        """
        parameter_tensors = tuple(parameter.as_tensor() for parameter in parameters)
        bundle = operation.infer_result(inputs, parameter_tensors)
        outputs = bundle.outputs
        result = bundle.result
        input_ports, parameter_ports, output_ports = self._bind_ports(
            operation,
            inputs,
            parameter_tensors,
            outputs,
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
            parameter_ports=parameter_ports,
            output_ports=output_ports,
            input_edges=tuple(value.id for value in inputs),
            parameter_ids=tuple(param.id for param in parameters),
            output_tensors=outputs,
            provenance=provenance,
            operation=operation,
            result=result,
        )
        values = tuple(
            tensor.registered(edge_id)
            for edge_id, tensor in zip(output_ids, outputs, strict=True)
        )
        return values[0] if len(values) == 1 else values

    def _bind_ports(
        self,
        operation: Operation,
        inputs: tuple[Tensor, ...],
        parameters: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
    ) -> tuple[tuple[Port, ...], tuple[Port, ...], tuple[Port, ...]]:
        """Validate tensors against declared port contracts."""
        if len(operation.input_ports) != len(inputs):
            raise PortArityError(
                f"{operation.family!r} expects "
                f"{len(operation.input_ports)} inputs, got {len(inputs)}"
            )
        if len(operation.parameter_ports) != len(parameters):
            raise PortArityError(
                f"{operation.family!r} expects "
                f"{len(operation.parameter_ports)} parameters, "
                f"got {len(parameters)}"
            )
        if len(operation.output_ports) != len(outputs):
            raise PortArityError(
                f"{operation.family!r} declares "
                f"{len(operation.output_ports)} outputs, inferred "
                f"{len(outputs)}"
            )

        for port, tensor in zip(operation.input_ports, inputs, strict=True):
            self._bind_port(operation, port, tensor, direction="input")
        for port, tensor in zip(operation.parameter_ports, parameters, strict=True):
            self._bind_port(operation, port, tensor, direction="parameter")
        for port, tensor in zip(operation.output_ports, outputs, strict=True):
            self._bind_port(operation, port, tensor, direction="output")
        return operation.input_ports, operation.parameter_ports, operation.output_ports

    @staticmethod
    def _bind_port(
        operation: Operation,
        port: Port,
        actual: Tensor,
        *,
        direction: str,
    ) -> None:
        """Reject a tensor that does not satisfy a declared port contract."""
        if port.contract is not None and not contract_satisfied(port.contract, actual):
            raise MetadataMismatchError(
                f"{operation.family!r} {direction} port {port.name!r} "
                f"expects {port.contract!r}, got {actual!r}"
            )

    def call_module(
        self, module: Module, inputs: tuple[Tensor, ...]
    ) -> Tensor | tuple[Tensor, ...]:
        """Invoke a module while recording its provenance path."""
        previous_path = self._module_path
        name = module.__class__.__name__
        self._module_path = (*previous_path, name)
        try:
            return module.forward(*inputs)
        finally:
            self._module_path = previous_path

    def mark_output(self, value: Tensor) -> None:
        """Expose a graph value as a graph output."""
        self.builder.mark_output(value.id)

    def build(self) -> Graph:
        """Finalize and return the immutable graph."""
        return self.builder.build()


def compose_graph(
    module_or_factory: Module | Callable[[Compose], Module],
    inputs: tuple[Tensor, ...],
) -> Graph:
    """Build a graph by eagerly composing one module.

    Args:
        module_or_factory: Root module or a factory that receives the active
            compose context and returns a module. Factories are required
            when module construction declares parameters.
        inputs: Root module input tensors in positional order.

    Returns:
        The finalized immutable graph.
    """
    with Compose() as context:
        if isinstance(module_or_factory, Module):
            module = module_or_factory
        else:
            module = module_or_factory(context)
        registered = tuple(context.input(tensor) for tensor in inputs)
        outputs = module(*registered)
        values = outputs if isinstance(outputs, tuple) else (outputs,)
        for value in values:
            context.mark_output(value)
        return context.build()
