"""Structural graph storage, validation, and construction."""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from .errors import (
    CrossGraphReferenceError,
    DuplicatePortError,
    GraphAlreadyFinalizedError,
    PortArityError,
    UnknownParameterError,
    UnknownOperationError,
    UnknownTensorError,
)
from .ids import (
    GraphId,
    OperationId,
    ParameterId,
    StorageId,
    TensorId,
)
from .metadata import ValueMetadata
from .graph_validation import GraphValidator
from .operation import Operation, OperationError, OperationResult, StructuralOperation
from .operation.records import Materialization
from .parameter import Parameter, bound_parameter_metadata
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
        GraphValidator().validate(self)


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
        self._next_tensor = 0
        self._next_parameter = 0
        self._next_operation = 0
        self._next_storage = 0
        self._inputs: list[TensorId] = []
        self._outputs: list[TensorId] = []
        self._tensors: dict[TensorId, Tensor] = {}
        self._parameters: dict[ParameterId, Parameter] = {}
        self._operations: dict[OperationId, StructuralOperation] = {}
        self._operation_order: list[OperationId] = []
        self._finalized = False

    @property
    def graph_id(self) -> GraphId:
        """Return the graph identity owned by this builder."""
        return self._graph_id

    def add_input(
        self,
        metadata: ValueMetadata,
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
        storage_id = StorageId(self._graph_id, self._next_storage)
        self._next_storage += 1
        self._tensors[tensor_id] = Tensor(
            tensor_id, metadata, provenance, None, (), storage_id
        )
        self._inputs.append(tensor_id)
        return tensor_id

    def add_parameter(
        self,
        metadata: ValueMetadata,
        *,
        trainable: bool = True,
    ) -> ParameterId:
        """Register a parameter that operations may reference.

        Args:
            metadata: Backend-neutral parameter shape and semantic metadata.
            trainable: Whether an optimizer should update this parameter.

        Returns:
            The newly allocated graph-local parameter identity.
        """
        self._ensure_open()
        self._validate_metadata(metadata)
        parameter_id = ParameterId(self._graph_id, self._next_parameter)
        self._next_parameter += 1
        self._parameters[parameter_id] = Parameter(
            parameter_id, metadata, trainable
        )
        return parameter_id

    def add_operation(
        self,
        *,
        operation_family: str,
        input_ports: tuple[PortSpec, ...],
        parameter_ports: tuple[PortSpec, ...] = (),
        output_ports: tuple[PortSpec, ...],
        input_tensors: tuple[TensorId, ...],
        parameter_ids: tuple[ParameterId, ...] = (),
        output_metadata: tuple[ValueMetadata, ...],
        provenance: Provenance,
        operation: Operation,
        result: OperationResult | None = None,
    ) -> tuple[TensorId, ...]:
        """Append an operation and connect it to existing input tensors.

        This method registers the operation, records consumer references on the supplied input tensors, and allocates/registers its output tensors. Input tensors and parameters must already belong to this builder.

        Args:
            operation_family: Stable semantic family of the operation.
            input_ports: Ordered declarations for operation inputs.
            output_ports: Ordered declarations for operation outputs.
            input_tensors: Existing tensors bound to ``input_ports``.
            output_metadata: Metadata for the newly allocated outputs.
            parameter_ids: Existing parameters referenced by the operation.
            provenance: Structured origin metadata.
            operation: Complete semantic and estimation operation declaration.

        Returns:
            Output tensor identities in output-port order.

        Raises:
            GraphError: If ownership, arity, ports, or metadata are invalid.
        """
        self._ensure_open()
        if len(input_ports) != len(input_tensors):
            raise PortArityError("Input port and tensor counts differ")
        if len(parameter_ports) != len(parameter_ids):
            raise PortArityError("Parameter port and id counts differ")
        if len(output_ports) != len(output_metadata):
            raise PortArityError("Output port and metadata counts differ")
        self._validate_ports(input_ports, parameter_ports, output_ports)
        for tensor_id in input_tensors:
            self._require_tensor(tensor_id)
        for parameter_id in parameter_ids:
            self._require_parameter(parameter_id)
        for metadata in output_metadata:
            self._validate_metadata(metadata)
        operation.validate_declaration()
        if operation.family != operation_family:
            raise OperationError("Operation family does not match its declaration")
        if (
            tuple((p.name, p.value_kind) for p in operation.input_ports)
            != tuple((p.name, p.value_kind) for p in input_ports)
            or tuple((p.name, p.value_kind) for p in operation.parameter_ports)
            != tuple((p.name, p.value_kind) for p in parameter_ports)
            or tuple((p.name, p.value_kind) for p in operation.output_ports)
            != tuple((p.name, p.value_kind) for p in output_ports)
        ):
            raise OperationError("Bound ports do not match the operation declaration")
        input_metadata = tuple(
            self._tensors[tensor_id].metadata for tensor_id in input_tensors
        )
        parameter_metadata = tuple(
            bound_parameter_metadata(self._parameters[parameter_id])
            for parameter_id in parameter_ids
        )

        if result is None:
            result = operation.infer_result(input_metadata, parameter_metadata)
        operation.validate_result(input_metadata, parameter_metadata, result)

        operation_id = OperationId(self._graph_id, self._next_operation)
        self._next_operation += 1
        input_port_names = {port.name for port in input_ports}
        parameter_port_names = {port.name for port in parameter_ports}
        saved_for_backward = tuple(
            PortRef(
                operation_id,
                saved_name,
                (
                    "input"
                    if saved_name in input_port_names
                    else "parameter"
                    if saved_name in parameter_port_names
                    else "output"
                ),
            )
            for saved_name in result.saved_for_backward
        )
        output_tensors = tuple(
            TensorId(self._graph_id, self._next_tensor + offset)
            for offset in range(len(output_metadata))
        )
        self._next_tensor += len(output_tensors)
        auxiliary_port_decls = operation.auxiliary_ports()
        aux_by_name = {port.name: port for port in auxiliary_port_decls}
        auxiliary_tensors: dict[str, TensorId] = {}
        for port_name in result.active_auxiliary_ports:
            port = aux_by_name[port_name]
            index = next(
                i for i, p in enumerate(auxiliary_port_decls) if p.name == port_name
            )
            aux_metadata = result.auxiliary_outputs[index]
            aux_alias = (
                result.auxiliary_aliases[index]
                if result.auxiliary_aliases
                else None
            )
            aux_id = TensorId(self._graph_id, self._next_tensor)
            self._next_tensor += 1
            storage_id = None
            if aux_alias is not None and aux_alias.materialization is Materialization.VIEW:
                source_index = next(
                    position
                    for position, input_port in enumerate(input_ports)
                    if input_port.name == aux_alias.source_port
                )
                storage_id = self._tensors[input_tensors[source_index]].storage_id
            if storage_id is None:
                storage_id = StorageId(self._graph_id, self._next_storage)
                self._next_storage += 1
            auxiliary_tensors[port_name] = aux_id
            self._tensors[aux_id] = Tensor(
                aux_id,
                aux_metadata,
                provenance,
                PortRef(operation_id, port.name, "auxiliary"),
                (),
                storage_id,
            )
        self._operations[operation_id] = StructuralOperation(
            operation_id,
            operation_family,
            input_ports,
            parameter_ports,
            output_ports,
            input_tensors,
            parameter_ids,
            output_tensors,
            provenance,
            operation,
            result,
            saved_for_backward,
            auxiliary_port_decls,
            auxiliary_tensors,
        )
        self._operation_order.append(operation_id)

        for tensor_id, port in zip(input_tensors, input_ports, strict=True):
            ref = PortRef(operation_id, port.name, "input")
            self._tensors[tensor_id] = replace(
                self._tensors[tensor_id],
                consumers=(*self._tensors[tensor_id].consumers, ref),
            )
        for index, (tensor_id, metadata, port) in enumerate(
            zip(
                output_tensors, output_metadata, output_ports, strict=True
            )
        ):
            alias = result.aliases[index] if result.aliases else None
            storage_id = None
            if alias is not None and alias.materialization is Materialization.VIEW:
                source_index = next(
                    position
                    for position, input_port in enumerate(input_ports)
                    if input_port.name == alias.source_port
                )
                storage_id = self._tensors[input_tensors[source_index]].storage_id
            if storage_id is None:
                storage_id = StorageId(self._graph_id, self._next_storage)
                self._next_storage += 1
            self._tensors[tensor_id] = Tensor(
                tensor_id,
                metadata,
                provenance,
                PortRef(operation_id, port.name, "output"),
                (),
                storage_id,
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
        """Raise when the builder has already been finalized."""
        if self._finalized:
            raise GraphAlreadyFinalizedError(
                "Structural graph builder has already been finalized"
            )

    def _validate_metadata(self, metadata: ValueMetadata) -> None:
        """Validate concrete metadata before adding it to this graph."""
        if not isinstance(metadata, ValueMetadata):
            raise TypeError("Graph metadata must be ValueMetadata")

    def _validate_ports(
        self,
        input_ports: tuple[PortSpec, ...],
        parameter_ports: tuple[PortSpec, ...],
        output_ports: tuple[PortSpec, ...],
    ) -> None:
        """Validate port uniqueness and concrete metadata ownership."""
        input_names = [port.name for port in input_ports]
        parameter_names = [port.name for port in parameter_ports]
        output_names = [port.name for port in output_ports]
        if len(input_names) != len(set(input_names)):
            raise DuplicatePortError("Duplicate input port name")
        if len(parameter_names) != len(set(parameter_names)):
            raise DuplicatePortError("Duplicate parameter port name")
        if len(output_names) != len(set(output_names)):
            raise DuplicatePortError("Duplicate output port name")
        all_names = input_names + parameter_names + output_names
        if len(all_names) != len(set(all_names)):
            raise DuplicatePortError("Port names must be unique across directions")
        for port in (*input_ports, *parameter_ports, *output_ports):
            if port.metadata is not None:
                self._validate_metadata(port.metadata)

    def _require_tensor(self, tensor_id: TensorId) -> None:
        """Require that a tensor belongs to this builder."""
        if tensor_id.graph_id != self._graph_id:
            raise CrossGraphReferenceError(
                f"Tensor {tensor_id} belongs to another graph"
            )
        if tensor_id not in self._tensors:
            raise UnknownTensorError(tensor_id)

    def _require_parameter(self, parameter_id: ParameterId) -> None:
        """Require that a parameter belongs to this builder."""
        if parameter_id.graph_id != self._graph_id:
            raise CrossGraphReferenceError(
                f"Parameter {parameter_id} belongs to another graph"
            )
        if parameter_id not in self._parameters:
            raise UnknownParameterError(parameter_id)
