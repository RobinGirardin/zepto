"""Immutable graph storage, validation, and construction."""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from zepto.compose.values import Parameter, Tensor
from .edge import Edge
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
    EdgeId,
    GraphId,
    NodeId,
    ParameterId,
    StorageId,
)
from .validation import GraphValidator
from zepto.semantic.operations import Operation, OperationError, OperationResult
from .node import Node
from zepto.semantic.operations.records import Materialization
from .parameter import parameter_as_tensor
from zepto.semantic.ports import PortLink, Port
from .provenance import Provenance


def _make_edge(
    edge_id: EdgeId,
    tensor: Tensor,
    *,
    provenance: Provenance | None,
    producer: PortLink | None,
    consumers: tuple[PortLink, ...],
    storage_id: StorageId | None,
) -> Edge:
    """Build an immutable edge from a structural tensor snapshot."""
    return Edge(
        edge_id,
        tensor.registered(edge_id),
        provenance,
        producer,
        consumers,
        storage_id,
    )


@dataclass(frozen=True, slots=True)
class Graph:
    """Immutable backend-neutral structural graph."""

    id: GraphId
    inputs: tuple[EdgeId, ...]
    outputs: tuple[EdgeId, ...]
    node_order: tuple[NodeId, ...]
    edges: Mapping[EdgeId, Edge]
    parameters: Mapping[ParameterId, Parameter]
    nodes: Mapping[NodeId, Node]

    def edge(self, edge_id: EdgeId) -> Edge:
        """Return an edge by identity."""
        try:
            return self.edges[edge_id]
        except KeyError as error:
            raise UnknownTensorError(edge_id) from error

    def node(self, node_id: NodeId) -> Node:
        """Return a node by identity."""
        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise UnknownOperationError(node_id) from error

    def parameter(self, parameter_id: ParameterId) -> Parameter:
        """Return a stored parameter by identity."""
        try:
            return self.parameters[parameter_id]
        except KeyError as error:
            raise UnknownParameterError(parameter_id) from error

    def validate(self) -> None:
        """Validate graph ownership, connectivity, and node ordering."""
        GraphValidator().validate(self)


class GraphBuilder:
    """Mutable builder that finalizes one immutable graph."""

    def __init__(self, graph_id: GraphId | None = None) -> None:
        self._graph_id = graph_id or GraphId.new()
        self._next_edge = 0
        self._next_parameter = 0
        self._next_node = 0
        self._next_storage = 0
        self._inputs: list[EdgeId] = []
        self._outputs: list[EdgeId] = []
        self._edges: dict[EdgeId, Edge] = {}
        self._parameters: dict[ParameterId, Parameter] = {}
        self._nodes: dict[NodeId, Node] = {}
        self._node_order: list[NodeId] = []
        self._finalized = False

    @property
    def graph_id(self) -> GraphId:
        return self._graph_id

    def add_input(
        self,
        tensor: Tensor,
        *,
        provenance: Provenance | None = None,
    ) -> EdgeId:
        """Declare a graph input edge."""
        self._ensure_open()
        self._validate_tensor(tensor)
        edge_id = EdgeId(self._graph_id, self._next_edge)
        self._next_edge += 1
        storage_id = StorageId(self._graph_id, self._next_storage)
        self._next_storage += 1
        self._edges[edge_id] = _make_edge(
            edge_id,
            tensor,
            provenance=provenance,
            producer=None,
            consumers=(),
            storage_id=storage_id,
        )
        self._inputs.append(edge_id)
        return edge_id

    def add_parameter(self, parameter: Parameter) -> ParameterId:
        """Register a parameter that nodes may reference."""
        self._ensure_open()
        if not isinstance(parameter, Parameter):
            raise TypeError("Graph parameters must be Parameter")
        parameter_id = ParameterId(self._graph_id, self._next_parameter)
        self._next_parameter += 1
        self._parameters[parameter_id] = parameter.registered(parameter_id)
        return parameter_id

    def add_operation(
        self,
        *,
        operation_family: str,
        input_ports: tuple[Port, ...],
        parameter_ports: tuple[Port, ...] = (),
        output_ports: tuple[Port, ...],
        input_edges: tuple[EdgeId, ...],
        parameter_ids: tuple[ParameterId, ...] = (),
        output_tensors: tuple[Tensor, ...],
        provenance: Provenance,
        operation: Operation,
        result: OperationResult | None = None,
    ) -> tuple[EdgeId, ...]:
        """Append a node and connect it to existing input edges."""
        self._ensure_open()
        if len(input_ports) != len(input_edges):
            raise PortArityError("Input port and edge counts differ")
        if len(parameter_ports) != len(parameter_ids):
            raise PortArityError("Parameter port and id counts differ")
        if len(output_ports) != len(output_tensors):
            raise PortArityError("Output port and tensor counts differ")
        self._validate_ports(input_ports, parameter_ports, output_ports)
        for edge_id in input_edges:
            self._require_edge(edge_id)
        for parameter_id in parameter_ids:
            self._require_parameter(parameter_id)
        for tensor in output_tensors:
            self._validate_tensor(tensor)
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
        input_tensors = tuple(self._edges[edge_id].tensor for edge_id in input_edges)
        parameter_tensors = tuple(
            parameter_as_tensor(self._parameters[parameter_id])
            for parameter_id in parameter_ids
        )

        if result is None:
            bundle = operation.infer_result(input_tensors, parameter_tensors)
            result = bundle.result
            auxiliary_outputs = bundle.auxiliary_outputs
        else:
            auxiliary_outputs = operation.infer_auxiliary_outputs(
                input_tensors, parameter_tensors, output_tensors
            )
            operation.validate_result(
                input_tensors,
                parameter_tensors,
                output_tensors,
                auxiliary_outputs,
                result,
            )

        node_id = NodeId(self._graph_id, self._next_node)
        self._next_node += 1
        input_port_names = {port.name for port in input_ports}
        parameter_port_names = {port.name for port in parameter_ports}
        saved_for_backward = tuple(
            PortLink(
                node_id,
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
        output_edges = tuple(
            EdgeId(self._graph_id, self._next_edge + offset)
            for offset in range(len(output_tensors))
        )
        self._next_edge += len(output_edges)
        auxiliary_port_decls = operation.auxiliary_ports()
        aux_by_name = {port.name: port for port in auxiliary_port_decls}
        auxiliary_edges: dict[str, EdgeId] = {}
        for port_name in result.active_auxiliary_ports:
            port = aux_by_name[port_name]
            index = next(
                i for i, p in enumerate(auxiliary_port_decls) if p.name == port_name
            )
            aux_tensor = auxiliary_outputs[index]
            aux_alias = (
                result.auxiliary_aliases[index]
                if result.auxiliary_aliases
                else None
            )
            aux_id = EdgeId(self._graph_id, self._next_edge)
            self._next_edge += 1
            storage_id = None
            if aux_alias is not None and aux_alias.materialization is Materialization.VIEW:
                source_index = next(
                    position
                    for position, input_port in enumerate(input_ports)
                    if input_port.name == aux_alias.source_port
                )
                storage_id = self._edges[input_edges[source_index]].storage_id
            if storage_id is None:
                storage_id = StorageId(self._graph_id, self._next_storage)
                self._next_storage += 1
            auxiliary_edges[port_name] = aux_id
            self._edges[aux_id] = _make_edge(
                aux_id,
                aux_tensor,
                provenance=provenance,
                producer=PortLink(node_id, port.name, "auxiliary"),
                consumers=(),
                storage_id=storage_id,
            )
        self._nodes[node_id] = Node(
            node_id,
            operation_family,
            input_ports,
            parameter_ports,
            output_ports,
            input_edges,
            parameter_ids,
            output_edges,
            provenance,
            operation,
            result,
            saved_for_backward,
            auxiliary_port_decls,
            auxiliary_edges,
        )
        self._node_order.append(node_id)

        for edge_id, port in zip(input_edges, input_ports, strict=True):
            ref = PortLink(node_id, port.name, "input")
            edge = self._edges[edge_id]
            self._edges[edge_id] = replace(
                edge,
                consumers=(*edge.consumers, ref),
            )
        for index, (edge_id, tensor, port) in enumerate(
            zip(output_edges, output_tensors, output_ports, strict=True)
        ):
            alias = result.aliases[index] if result.aliases else None
            storage_id = None
            if alias is not None and alias.materialization is Materialization.VIEW:
                source_index = next(
                    position
                    for position, input_port in enumerate(input_ports)
                    if input_port.name == alias.source_port
                )
                storage_id = self._edges[input_edges[source_index]].storage_id
            if storage_id is None:
                storage_id = StorageId(self._graph_id, self._next_storage)
                self._next_storage += 1
            self._edges[edge_id] = _make_edge(
                edge_id,
                tensor,
                provenance=provenance,
                producer=PortLink(node_id, port.name, "output"),
                consumers=(),
                storage_id=storage_id,
            )
        return output_edges

    def mark_output(self, edge_id: EdgeId) -> None:
        """Mark an existing edge as a graph output."""
        self._ensure_open()
        self._require_edge(edge_id)
        if edge_id not in self._outputs:
            self._outputs.append(edge_id)

    def build(self) -> Graph:
        """Validate and finalize the immutable graph."""
        self._ensure_open()
        graph = Graph(
            self._graph_id,
            tuple(self._inputs),
            tuple(self._outputs),
            tuple(self._node_order),
            MappingProxyType(dict(self._edges)),
            MappingProxyType(dict(self._parameters)),
            MappingProxyType(dict(self._nodes)),
        )
        graph.validate()
        self._finalized = True
        return graph

    def _ensure_open(self) -> None:
        if self._finalized:
            raise GraphAlreadyFinalizedError(
                "Graph builder has already been finalized"
            )

    def _validate_tensor(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError("Graph values must be Tensor")

    def _validate_ports(
        self,
        input_ports: tuple[Port, ...],
        parameter_ports: tuple[Port, ...],
        output_ports: tuple[Port, ...],
    ) -> None:
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

    def _require_edge(self, edge_id: EdgeId) -> None:
        if edge_id.graph_id != self._graph_id:
            raise CrossGraphReferenceError(
                f"Edge {edge_id} belongs to another graph"
            )
        if edge_id not in self._edges:
            raise UnknownTensorError(edge_id)

    def _require_parameter(self, parameter_id: ParameterId) -> None:
        if parameter_id.graph_id != self._graph_id:
            raise CrossGraphReferenceError(
                f"Parameter {parameter_id} belongs to another graph"
            )
        if parameter_id not in self._parameters:
            raise UnknownParameterError(parameter_id)
