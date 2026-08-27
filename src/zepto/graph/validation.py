"""Structural graph validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import (
    CrossGraphReferenceError,
    InvalidGraphOutputError,
    InvalidOperationOrderError,
    PortArityError,
)
from zepto.semantic.metadata import contract_satisfied
from zepto.semantic.operations.records import Materialization
from .parameter import parameter_as_tensor

if TYPE_CHECKING:
    from .graph import Graph


class GraphValidator:
    """Validate structural graph invariants."""

    def validate(self, graph: Graph) -> None:
        self.validate_ownership(graph)
        self.validate_port_bindings(graph)
        self.validate_operation_order(graph)
        self.validate_storage_links(graph)
        self.validate_graph_outputs(graph)

    def validate_ownership(self, graph: Graph) -> None:
        for edge_id in (*graph.inputs, *graph.outputs, *graph.edges):
            if edge_id.graph_id != graph.id:
                raise CrossGraphReferenceError(
                    f"Edge {edge_id} does not belong to graph {graph.id}"
                )
        for node_id, node in graph.nodes.items():
            if node_id != node.id or node.id.graph_id != graph.id:
                raise CrossGraphReferenceError(
                    f"Node {node_id} does not belong to graph {graph.id}"
                )

    def validate_port_bindings(self, graph: Graph) -> None:
        for node_id, node in graph.nodes.items():
            if node.declaration is None:
                raise ValueError(f"Node {node_id} has no complete contract")
            node.declaration.validate_declaration()
            if node.result is not None:
                input_tensors = tuple(
                    graph.edge(edge_id).tensor for edge_id in node.input_edges
                )
                parameter_tensors = tuple(
                    parameter_as_tensor(graph.parameter(parameter_id))
                    for parameter_id in node.parameter_ids
                )
                output_tensors = tuple(
                    graph.edge(edge_id).tensor for edge_id in node.output_edges
                )
                auxiliary_outputs = node.declaration.infer_auxiliary_outputs(
                    input_tensors, parameter_tensors, output_tensors
                )
                node.declaration.validate_result(
                    input_tensors,
                    parameter_tensors,
                    output_tensors,
                    auxiliary_outputs,
                    node.result,
                )
            for port, edge_id in zip(node.input_ports, node.input_edges, strict=True):
                self._check_contract(port.contract, graph.edge(edge_id).tensor)
            for port, parameter_id in zip(
                node.parameter_ports, node.parameter_ids, strict=True
            ):
                self._check_contract(
                    port.contract,
                    parameter_as_tensor(graph.parameter(parameter_id)),
                )
            for port, edge_id in zip(node.output_ports, node.output_edges, strict=True):
                self._check_contract(port.contract, graph.edge(edge_id).tensor)
            for saved_ref in node.saved_for_backward:
                saved_ref.resolve(node)
            for port_name, edge_id in (node.auxiliary_edges or {}).items():
                if edge_id not in graph.edges:
                    raise ValueError(
                        f"Unknown auxiliary edge {edge_id} for {node_id}"
                    )
                port_ref = graph.edge(edge_id).producer
                if (
                    port_ref is None
                    or port_ref.node_id != node_id
                    or port_ref.role != "auxiliary"
                ):
                    raise ValueError(
                        f"Auxiliary edge {edge_id} has invalid provenance "
                        f"for node {node_id}"
                    )
                if port_ref.port_name != port_name:
                    raise ValueError(
                        f"Auxiliary edge {edge_id} port name mismatch "
                        f"for node {node_id}"
                    )
            if len(node.input_ports) != len(node.input_edges):
                raise PortArityError(f"Input arity mismatch for {node_id}")
            if len(node.parameter_ports) != len(node.parameter_ids):
                raise PortArityError(f"Parameter arity mismatch for {node_id}")
            if len(node.output_ports) != len(node.output_edges):
                raise PortArityError(f"Output arity mismatch for {node_id}")

    def validate_operation_order(self, graph: Graph) -> None:
        node_positions = {
            node_id: position for position, node_id in enumerate(graph.node_order)
        }
        for node_id, node in graph.nodes.items():
            for edge_id in node.input_edges:
                edge = graph.edge(edge_id)
                if edge.producer is not None:
                    producer_position = node_positions[edge.producer.node_id]
                    if producer_position >= node_positions[node_id]:
                        raise InvalidOperationOrderError(
                            f"{node_id} consumes a later node's edge"
                        )

    def validate_storage_links(self, graph: Graph) -> None:
        for node_id, node in graph.nodes.items():
            if node.result is None:
                continue
            aliases = node.result.aliases or (None,) * len(node.output_edges)
            input_storage = {
                graph.edge(edge_id).storage_id for edge_id in node.input_edges
            }
            for index, (edge_id, alias) in enumerate(
                zip(node.output_edges, aliases, strict=True)
            ):
                output_storage = graph.edge(edge_id).storage_id
                if alias is not None and alias.materialization is Materialization.VIEW:
                    source_index = next(
                        position
                        for position, port in enumerate(node.input_ports)
                        if port.name == alias.source_port
                    )
                    source_storage = graph.edge(
                        node.input_edges[source_index]
                    ).storage_id
                    if output_storage != source_storage:
                        raise ValueError(
                            f"View output {edge_id} must share source storage "
                            f"for node {node_id}"
                        )
                elif output_storage in input_storage:
                    raise ValueError(
                        f"Non-view output {edge_id} illegally shares input "
                        f"storage for node {node_id}"
                    )
            if node.auxiliary_edges and node.result is not None:
                aux_aliases = node.result.auxiliary_aliases or (
                    None,
                ) * len(node.auxiliary_ports)
                aux_ports = node.auxiliary_ports
                for port_name, edge_id in node.auxiliary_edges.items():
                    index = next(
                        i for i, port in enumerate(aux_ports) if port.name == port_name
                    )
                    alias = aux_aliases[index] if aux_aliases else None
                    if alias is not None and alias.materialization is Materialization.VIEW:
                        aux_storage = graph.edge(edge_id).storage_id
                        source_index = next(
                            position
                            for position, port in enumerate(node.input_ports)
                            if port.name == alias.source_port
                        )
                        source_storage = graph.edge(
                            node.input_edges[source_index]
                        ).storage_id
                        if aux_storage != source_storage:
                            raise ValueError(
                                f"View auxiliary {edge_id} must share source "
                                f"storage for node {node_id}"
                            )

    def validate_graph_outputs(self, graph: Graph) -> None:
        for edge_id in graph.outputs:
            if edge_id not in graph.edges:
                raise InvalidGraphOutputError(f"Unknown graph output {edge_id}")

    @staticmethod
    def _check_contract(contract, tensor) -> None:
        if contract is not None and not contract_satisfied(contract, tensor):
            raise ValueError(
                f"Port contract {contract!r} is not satisfied by {tensor!r}"
            )
