"""Identity lowering implementations delegating to graph nodes."""

from __future__ import annotations

from dataclasses import dataclass, replace

from zepto.analysis.lowered import LoweredNode
from zepto.graph.node import Node
from zepto.graph.graph import Graph
from zepto.semantic.operations.base import Operation
from zepto.semantic.operations.records import EstimationContext
from ..context import InvocationContext
from ..helpers import (
    append_save_events,
    ensure_lowered_edge,
    remap_events,
)
from ..registry import ImplementationDescriptor


@dataclass(frozen=True, slots=True)
class IdentityImplementation:
    """Wrap one semantic Operation as a 1:1 lowered implementation."""

    operation: Operation
    descriptor: ImplementationDescriptor

    def compatible(
        self,
        node: Node,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        del graph, context
        if node.operation_family != self.descriptor.family:
            return "family mismatch"
        return None

    def lower(
        self,
        node: Node,
        graph: Graph,
        context: InvocationContext,
        edge_map: dict,
        *,
        estimation: EstimationContext,
        lowered_edges: dict,
    ) -> LoweredNode:
        op = node.declaration
        if op is None:
            op = self.operation
        result = node.result
        if result is None:
            raise ValueError(f"Node {node.id} has no result")

        for port, eid in zip(node.input_ports, node.input_edges, strict=True):
            ensure_lowered_edge(
                eid,
                graph,
                context,
                lowered_edges,
                edge_map,
                node=node,
                port_name=port.name,
                port_direction="input",
            )
        for port, eid in zip(node.output_ports, node.output_edges, strict=True):
            ensure_lowered_edge(
                eid,
                graph,
                context,
                lowered_edges,
                edge_map,
                node=node,
                port_name=port.name,
                port_direction="output",
            )
        if node.auxiliary_edges is not None:
            aux_by_name = {port.name: port for port in node.auxiliary_ports}
            for port_name, eid in node.auxiliary_edges.items():
                port = aux_by_name[port_name]
                ensure_lowered_edge(
                    eid,
                    graph,
                    context,
                    lowered_edges,
                    edge_map,
                    node=node,
                    port_name=port.name,
                    port_direction="auxiliary",
                )

        events = op.resource_events(estimation, result)
        forward = op.forward_flops(estimation)
        backward = op.backward_flops(replace(estimation, phase="backward"))

        if not isinstance(forward, int) or forward < 0:
            raise ValueError(f"{op.family}: forward_flops must be non-negative int")
        if not isinstance(backward, int) or backward < 0:
            raise ValueError(f"{op.family}: backward_flops must be non-negative int")

        remapped = remap_events(events, node, edge_map)
        with_saves = append_save_events(remapped, node, edge_map)

        input_ids = tuple(edge_map[bound_edge_id] for bound_edge_id in node.input_edges)
        output_ids = tuple(
            edge_map[bound_edge_id] for bound_edge_id in node.output_edges
        )
        auxiliary_ids = ()
        if node.auxiliary_edges is not None:
            auxiliary_ids = tuple(
                edge_map[bound_edge_id]
                for bound_edge_id in node.auxiliary_edges.values()
            )

        return LoweredNode(
            id=f"op{node.id.index}",
            node_id=node.id,
            implementation=self.descriptor.id,
            input_edges=input_ids,
            output_edges=output_ids,
            auxiliary_edges=auxiliary_ids,
            resource_events=with_saves,
            forward_flops=forward,
            backward_flops=backward,
        )
