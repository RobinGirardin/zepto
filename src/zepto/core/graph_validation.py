"""Structural graph validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import (
    CrossGraphReferenceError,
    InvalidGraphOutputError,
    InvalidOperationOrderError,
    PortArityError,
)
from .operation.records import Materialization

if TYPE_CHECKING:
    from .graph import StructuralGraph


class GraphValidator:
    """Validate structural graph invariants."""

    def validate(self, graph: StructuralGraph) -> None:
        self.validate_ownership(graph)
        self.validate_port_bindings(graph)
        self.validate_operation_order(graph)
        self.validate_storage_links(graph)
        self.validate_graph_outputs(graph)

    def validate_ownership(self, graph: StructuralGraph) -> None:
        for tensor_id in (*graph.inputs, *graph.outputs, *graph.tensors):
            if tensor_id.graph_id != graph.id:
                raise CrossGraphReferenceError(
                    f"Tensor {tensor_id} does not belong to graph {graph.id}"
                )
        for operation_id, operation in graph.operation_nodes.items():
            if operation_id != operation.id or operation.id.graph_id != graph.id:
                raise CrossGraphReferenceError(
                    f"Operation {operation_id} does not belong to graph {graph.id}"
                )

    def validate_port_bindings(self, graph: StructuralGraph) -> None:
        for operation_id, operation in graph.operation_nodes.items():
            if operation.declaration is None:
                raise ValueError(f"Operation {operation_id} has no complete contract")
            operation.declaration.validate_declaration()
            if operation.result is not None:
                operation.declaration.validate_result(
                    tuple(
                        graph.tensor(tensor_id).metadata
                        for tensor_id in operation.input_tensors
                    ),
                    operation.result,
                )
            for saved_ref in operation.saved_for_backward:
                saved_ref.resolve(operation)
            for port_name, tensor_id in (operation.auxiliary_tensors or {}).items():
                if tensor_id not in graph.tensors:
                    raise ValueError(
                        f"Unknown auxiliary tensor {tensor_id} for {operation_id}"
                    )
                port_ref = graph.tensor(tensor_id).producer
                if (
                    port_ref is None
                    or port_ref.operation_id != operation_id
                    or port_ref.role != "auxiliary"
                ):
                    raise ValueError(
                        f"Auxiliary tensor {tensor_id} has invalid provenance "
                        f"for operation {operation_id}"
                    )
                if port_ref.port_name != port_name:
                    raise ValueError(
                        f"Auxiliary tensor {tensor_id} port name mismatch "
                        f"for operation {operation_id}"
                    )
            if len(operation.input_ports) != len(operation.input_tensors):
                raise PortArityError(f"Input arity mismatch for {operation_id}")
            if len(operation.output_ports) != len(operation.output_tensors):
                raise PortArityError(f"Output arity mismatch for {operation_id}")

    def validate_operation_order(self, graph: StructuralGraph) -> None:
        operation_positions = {
            operation_id: position
            for position, operation_id in enumerate(graph.operations)
        }
        for operation_id, operation in graph.operation_nodes.items():
            for tensor_id in operation.input_tensors:
                tensor = graph.tensor(tensor_id)
                if tensor.producer is not None:
                    producer_position = operation_positions[
                        tensor.producer.operation_id
                    ]
                    if producer_position >= operation_positions[operation_id]:
                        raise InvalidOperationOrderError(
                            f"{operation_id} consumes a later operation's tensor"
                        )

    def validate_storage_links(self, graph: StructuralGraph) -> None:
        for operation_id, operation in graph.operation_nodes.items():
            if operation.result is None:
                continue
            aliases = operation.result.aliases or (None,) * len(
                operation.output_tensors
            )
            input_storage = {
                graph.tensor(tensor_id).storage_id
                for tensor_id in operation.input_tensors
            }
            for index, (tensor_id, alias) in enumerate(
                zip(operation.output_tensors, aliases, strict=True)
            ):
                output_storage = graph.tensor(tensor_id).storage_id
                if alias is not None and alias.materialization is Materialization.VIEW:
                    source_index = next(
                        position
                        for position, port in enumerate(operation.input_ports)
                        if port.name == alias.source_port
                    )
                    source_storage = graph.tensor(
                        operation.input_tensors[source_index]
                    ).storage_id
                    if output_storage != source_storage:
                        raise ValueError(
                            f"View output {tensor_id} must share source storage "
                            f"for operation {operation_id}"
                        )
                elif output_storage in input_storage:
                    raise ValueError(
                        f"Non-view output {tensor_id} illegally shares input "
                        f"storage for operation {operation_id}"
                    )
            if operation.auxiliary_tensors and operation.result is not None:
                aux_aliases = operation.result.auxiliary_aliases or (
                    None,
                ) * len(operation.result.auxiliary_outputs)
                aux_ports = operation.auxiliary_ports
                for port_name, tensor_id in operation.auxiliary_tensors.items():
                    index = next(
                        i for i, port in enumerate(aux_ports) if port.name == port_name
                    )
                    alias = aux_aliases[index] if aux_aliases else None
                    if alias is not None and alias.materialization is Materialization.VIEW:
                        aux_storage = graph.tensor(tensor_id).storage_id
                        source_index = next(
                            position
                            for position, port in enumerate(operation.input_ports)
                            if port.name == alias.source_port
                        )
                        source_storage = graph.tensor(
                            operation.input_tensors[source_index]
                        ).storage_id
                        if aux_storage != source_storage:
                            raise ValueError(
                                f"View auxiliary {tensor_id} must share source "
                                f"storage for operation {operation_id}"
                            )

    def validate_graph_outputs(self, graph: StructuralGraph) -> None:
        for tensor_id in graph.outputs:
            if tensor_id not in graph.tensors:
                raise InvalidGraphOutputError(f"Unknown graph output {tensor_id}")
