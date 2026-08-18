from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from .errors import (
    CrossGraphReferenceError,
    DuplicatePortError,
    ForeignDimensionScopeError,
    GraphAlreadyFinalizedError,
    InvalidGraphOutputError,
    InvalidOperationOrderError,
    InvalidPortReferenceError,
    PortArityError,
    UnknownParameterError,
    UnknownOperationError,
    UnknownTensorError,
)
from .ids import (
    DimensionScope,
    GraphId,
    OperationId,
    ParameterId,
    TensorId,
)
from .metadata import SymbolicDim, TensorMetadata
from .operation import StructuralOperation
from .parameter import Parameter
from .ports import PortRef, PortSpec
from .provenance import Provenance
from .tensor import Tensor


@dataclass(frozen=True, slots=True)
class StructuralGraph:
    """Immutable backend-neutral structural graph.

    Attributes:
        id: Stable identity of this graph.
        inputs: Graph input tensor identities in declaration order.
        outputs: Graph output tensor identities in declaration order.
        operations: Operation identities in eager composition order.
        tensors: Read-only mapping of tensor identities to tensors.
        parameters: Read-only mapping of parameter identities to parameters.
        operation_nodes: Read-only mapping of operation identities to operations.
    """

    id: GraphId
    inputs: tuple[TensorId, ...]
    outputs: tuple[TensorId, ...]
    operations: tuple[OperationId, ...]
    tensors: Mapping[TensorId, Tensor]
    parameters: Mapping[ParameterId, Parameter]
    operation_nodes: Mapping[OperationId, StructuralOperation]

    def tensor(self, tensor_id: TensorId) -> Tensor:
        """Return a tensor by identity.

        Args:
            tensor_id: Identity of the tensor to retrieve.

        Returns:
            The corresponding graph tensor.

        Raises:
            UnknownTensorError: If the tensor is not in this graph.
        """
        try:
            return self.tensors[tensor_id]
        except KeyError as error:
            raise UnknownTensorError(tensor_id) from error

    def operation(self, operation_id: OperationId) -> StructuralOperation:
        """Return an operation by its graph-local identity.

        Args:
            operation_id: Identity of the operation to retrieve.

        Returns:
            The corresponding structural operation.

        Raises:
            UnknownOperationError: If the operation is not in this graph.
        """
        try:
            return self.operation_nodes[operation_id]
        except KeyError as error:
            raise UnknownOperationError(operation_id) from error

    def parameter(self, parameter_id: ParameterId) -> Parameter:
        """Return a parameter by identity.

        Args:
            parameter_id: Identity of the parameter to retrieve.

        Returns:
            The corresponding graph parameter.

        Raises:
            UnknownParameterError: If the parameter is not in this graph.
        """
        try:
            return self.parameters[parameter_id]
        except KeyError as error:
            raise UnknownParameterError(parameter_id) from error

    def validate(self) -> None:
        """Validate graph ownership, connectivity, and operation ordering.

        Raises:
            GraphError: If the graph violates a structural invariant.
        """
        operation_positions = {
            operation_id: position
            for position, operation_id in enumerate(self.operations)
        }
        for tensor_id in (*self.inputs, *self.outputs, *self.tensors):
            if tensor_id.graph_id != self.id:
                raise CrossGraphReferenceError(
                    f"Tensor {tensor_id} does not belong to graph {self.id}"
                )
        for operation_id, operation in self.operation_nodes.items():
            if operation_id != operation.id or operation.id.graph_id != self.id:
                raise CrossGraphReferenceError(
                    f"Operation {operation_id} does not belong to graph {self.id}"
                )
            if len(operation.input_ports) != len(operation.input_tensors):
                raise PortArityError(f"Input arity mismatch for {operation_id}")
            if len(operation.output_ports) != len(operation.output_tensors):
                raise PortArityError(f"Output arity mismatch for {operation_id}")
            for tensor_id in operation.input_tensors:
                tensor = self.tensor(tensor_id)
                if tensor.producer is not None:
                    producer_position = operation_positions[tensor.producer.operation_id]
                    if producer_position >= operation_positions[operation_id]:
                        raise InvalidOperationOrderError(
                            f"{operation_id} consumes a later operation's tensor"
                        )
        for tensor_id in self.outputs:
            if tensor_id not in self.tensors:
                raise InvalidGraphOutputError(f"Unknown graph output {tensor_id}")


class StructuralGraphBuilder:
    """Mutable builder that finalizes one immutable structural graph.

    The builder owns all graph-local identities and validates references as
    declarations are added. Calling :meth:`build` freezes the current state;
    subsequent mutations raise ``GraphAlreadyFinalizedError``.
    """

    def __init__(self, graph_id: GraphId | None = None) -> None:
        """Create an empty structural graph builder.

        Args:
            graph_id: Optional identity to use for the graph. A new identity
                is generated when omitted.
        """
        self._graph_id = graph_id or GraphId.new()
        self._dimension_scope = DimensionScope.new()
        self._dimensions: dict[str, SymbolicDim] = {}
        self._next_tensor = 0
        self._next_parameter = 0
        self._next_operation = 0
        self._inputs: list[TensorId] = []
        self._outputs: list[TensorId] = []
        self._tensors: dict[TensorId, Tensor] = {}
        self._parameters: dict[ParameterId, Parameter] = {}
        self._operations: dict[OperationId, StructuralOperation] = {}
        self._operation_order: list[OperationId] = []
        self._finalized = False

    @property
    def graph_id(self) -> GraphId:
        return self._graph_id

    def symbolic_dim(self, name: str) -> SymbolicDim:
        """Return the canonical symbolic dimension for ``name``.

        Args:
            name: Human-readable name within this builder's dimension scope.

        Returns:
            A stable symbolic dimension reused by subsequent calls with the
            same name.
        """
        self._ensure_open()
        if name not in self._dimensions:
            self._dimensions[name] = SymbolicDim(name, self._dimension_scope)
        return self._dimensions[name]

    def add_input(
        self,
        metadata: TensorMetadata,
        *,
        provenance: Provenance | None = None,
    ) -> TensorId:
        """Declare a graph input tensor.

        Args:
            metadata: Backend-neutral shape and semantic metadata.
            provenance: Optional origin metadata for the input.

        Returns:
            The newly allocated graph-local tensor identity.
        """
        self._ensure_open()
        self._validate_metadata(metadata)
        tensor_id = TensorId(self._graph_id, self._next_tensor)
        self._next_tensor += 1
        self._tensors[tensor_id] = Tensor(
            tensor_id, metadata, provenance, None, ()
        )
        self._inputs.append(tensor_id)
        return tensor_id

    def add_parameter(
        self,
        metadata: TensorMetadata,
        *,
        requires_grad: bool = True,
    ) -> ParameterId:
        """Register a parameter that operations may reference.

        Args:
            metadata: Backend-neutral parameter shape and semantic metadata.
            requires_grad: Whether the parameter participates in gradients.

        Returns:
            The newly allocated graph-local parameter identity.
        """
        self._ensure_open()
        self._validate_metadata(metadata)
        parameter_id = ParameterId(self._graph_id, self._next_parameter)
        self._next_parameter += 1
        self._parameters[parameter_id] = Parameter(
            parameter_id, metadata, requires_grad
        )
        return parameter_id

    def add_operation(
        self,
        *,
        operation_family: str,
        input_ports: tuple[PortSpec, ...],
        output_ports: tuple[PortSpec, ...],
        input_tensors: tuple[TensorId, ...],
        output_metadata: tuple[TensorMetadata, ...],
        parameter_ids: tuple[ParameterId, ...] = (),
        provenance: Provenance,
        declaration: object | None = None,
    ) -> tuple[TensorId, ...]:
        """Append an operation and connect it to existing input tensors.

        This method registers the operation, records consumer references on
        the supplied input tensors, and allocates/registers its output
        tensors. Input tensors and parameters must already belong to this
        builder.

        Args:
            operation_family: Stable semantic family of the operation.
            input_ports: Ordered declarations for operation inputs.
            output_ports: Ordered declarations for operation outputs.
            input_tensors: Existing tensors bound to ``input_ports``.
            output_metadata: Metadata for the newly allocated outputs.
            parameter_ids: Existing parameters referenced by the operation.
            provenance: Structured origin metadata.
            declaration: Optional opaque operation declaration for later
                semantic and estimation contracts.

        Returns:
            Output tensor identities in output-port order.

        Raises:
            GraphError: If ownership, arity, ports, or metadata are invalid.
        """
        self._ensure_open()
        if len(input_ports) != len(input_tensors):
            raise PortArityError("Input port and tensor counts differ")
        if len(output_ports) != len(output_metadata):
            raise PortArityError("Output port and metadata counts differ")
        self._validate_ports(input_ports, output_ports)
        for tensor_id in input_tensors:
            self._require_tensor(tensor_id)
        for parameter_id in parameter_ids:
            self._require_parameter(parameter_id)
        for metadata in output_metadata:
            self._validate_metadata(metadata)

        operation_id = OperationId(self._graph_id, self._next_operation)
        self._next_operation += 1
        output_tensors = tuple(
            TensorId(self._graph_id, self._next_tensor + offset)
            for offset in range(len(output_metadata))
        )
        self._next_tensor += len(output_tensors)
        operation = StructuralOperation(
            operation_id,
            operation_family,
            input_ports,
            output_ports,
            input_tensors,
            output_tensors,
            parameter_ids,
            provenance,
            declaration,
        )
        self._operations[operation_id] = operation
        self._operation_order.append(operation_id)

        for tensor_id, port in zip(input_tensors, input_ports, strict=True):
            ref = PortRef(operation_id, port.name, "input")
            self._tensors[tensor_id] = replace(
                self._tensors[tensor_id],
                consumers=(*self._tensors[tensor_id].consumers, ref),
            )
        for tensor_id, metadata, port in zip(
            output_tensors, output_metadata, output_ports, strict=True
        ):
            self._tensors[tensor_id] = Tensor(
                tensor_id,
                metadata,
                provenance,
                PortRef(operation_id, port.name, "output"),
                (),
            )
        return output_tensors

    def mark_output(self, tensor_id: TensorId) -> None:
        """Mark an existing tensor as a graph output.

        Args:
            tensor_id: Existing tensor identity to expose as an output.
        """
        self._ensure_open()
        self._require_tensor(tensor_id)
        if tensor_id not in self._outputs:
            self._outputs.append(tensor_id)

    def build(self) -> StructuralGraph:
        """Validate and finalize the immutable structural graph.

        Returns:
            The finalized structural graph.

        Raises:
            GraphError: If a structural invariant is violated.
            GraphAlreadyFinalizedError: If this builder was already finalized.
        """
        self._ensure_open()
        graph = StructuralGraph(
            self._graph_id,
            tuple(self._inputs),
            tuple(self._outputs),
            tuple(self._operation_order),
            MappingProxyType(dict(self._tensors)),
            MappingProxyType(dict(self._parameters)),
            MappingProxyType(dict(self._operations)),
        )
        graph.validate()
        self._finalized = True
        return graph

    def _ensure_open(self) -> None:
        if self._finalized:
            raise GraphAlreadyFinalizedError(
                "Structural graph builder has already been finalized"
            )

    def _validate_metadata(self, metadata: TensorMetadata) -> None:
        for dimension in metadata.shape:
            if (
                isinstance(dimension, SymbolicDim)
                and dimension.scope != self._dimension_scope
            ):
                raise ForeignDimensionScopeError(
                    f"Dimension {dimension.name!r} belongs to another scope"
                )

    def _validate_ports(
        self,
        input_ports: tuple[PortSpec, ...],
        output_ports: tuple[PortSpec, ...],
    ) -> None:
        input_names = [port.name for port in input_ports]
        output_names = [port.name for port in output_ports]
        if len(input_names) != len(set(input_names)):
            raise DuplicatePortError("Duplicate input port name")
        if len(output_names) != len(set(output_names)):
            raise DuplicatePortError("Duplicate output port name")
        if set(input_names) & set(output_names):
            raise DuplicatePortError("Input and output port names must be distinct")
        for port in (*input_ports, *output_ports):
            if port.metadata is not None:
                self._validate_metadata(port.metadata)

    def _require_tensor(self, tensor_id: TensorId) -> None:
        if tensor_id.graph_id != self._graph_id:
            raise CrossGraphReferenceError(
                f"Tensor {tensor_id} belongs to another graph"
            )
        if tensor_id not in self._tensors:
            raise UnknownTensorError(tensor_id)

    def _require_parameter(self, parameter_id: ParameterId) -> None:
        if parameter_id.graph_id != self._graph_id:
            raise CrossGraphReferenceError(
                f"Parameter {parameter_id} belongs to another graph"
            )
        if parameter_id not in self._parameters:
            raise UnknownParameterError(parameter_id)
