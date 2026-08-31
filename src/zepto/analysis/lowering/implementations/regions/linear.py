"""Fused linear region lowering via provenance discovery."""

from __future__ import annotations

from dataclasses import dataclass, replace

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.graph.node import Node
from ...context import InvocationContext
from ...helpers import (
    RegionEstimationContext,
    append_save_events,
    build_estimation_context,
    ensure_lowered_edge,
    remap_events,
)
from ...region import ProvenanceMatchRule, Region
from ...registry import RegionImplementationDescriptor


@dataclass(frozen=True, slots=True)
class FusedLinearRegionImplementation:
    """Lower one linear_matmul inside a Linear module envelope."""

    descriptor: RegionImplementationDescriptor

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        del context
        if len(region.operation_ids) != 1:
            return "fused linear expects one operation"
        structural = graph.node(region.operation_ids[0])
        if structural.operation_family != "linear_matmul":
            return "not linear_matmul"
        if region.anchor.component_type != "Linear":
            return "not a Linear module invocation"
        return None

    def lower(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
        edge_map: dict,
        *,
        estimation: RegionEstimationContext,
        lowered_edges: dict,
    ) -> LoweredNode:
        del estimation
        structural = graph.node(region.operation_ids[0])
        return _lower_linear_matmul(
            structural,
            graph,
            context,
            edge_map,
            lowered_edges,
            region=region,
            implementation_id=self.descriptor.id,
        )


def _lower_linear_matmul(
    structural: Node,
    graph: Graph,
    context: InvocationContext,
    edge_map: dict,
    lowered_edges: dict,
    *,
    region: Region,
    implementation_id: str,
) -> LoweredNode:
    op = structural.declaration
    if op is None:
        raise ValueError(f"Node {structural.id} has no declaration")
    result = structural.result
    if result is None:
        raise ValueError(f"Node {structural.id} has no result")

    for port, edge_id in zip(structural.input_ports, structural.input_edges, strict=True):
        ensure_lowered_edge(
            edge_id,
            graph,
            context,
            lowered_edges,
            edge_map,
            node=structural,
            port_name=port.name,
            port_direction="input",
        )
    for port, edge_id in zip(structural.output_ports, structural.output_edges, strict=True):
        ensure_lowered_edge(
            edge_id,
            graph,
            context,
            lowered_edges,
            edge_map,
            node=structural,
            port_name=port.name,
            port_direction="output",
        )
    if structural.auxiliary_edges is not None:
        aux_by_name = {port.name: port for port in structural.auxiliary_ports}
        for port_name, edge_id in structural.auxiliary_edges.items():
            port = aux_by_name[port_name]
            ensure_lowered_edge(
                edge_id,
                graph,
                context,
                lowered_edges,
                edge_map,
                node=structural,
                port_name=port.name,
                port_direction="auxiliary",
            )

    estimation = build_estimation_context(structural, graph, context)
    events = op.resource_events(estimation, result)
    forward = op.forward_flops(estimation)
    backward = op.backward_flops(replace(estimation, phase="backward"))

    remapped = remap_events(events, structural, edge_map)
    with_saves = append_save_events(remapped, structural, edge_map)

    input_ids = tuple(edge_map[eid] for eid in structural.input_edges)
    output_ids = tuple(edge_map[eid] for eid in structural.output_edges)
    auxiliary_ids = ()
    if structural.auxiliary_edges is not None:
        auxiliary_ids = tuple(
            edge_map[eid] for eid in structural.auxiliary_edges.values()
        )

    return LoweredNode(
        id=f"region:{region.id}",
        node_id=structural.id,
        node_ids=region.operation_ids,
        implementation=implementation_id,
        input_edges=input_ids,
        output_edges=output_ids,
        auxiliary_edges=auxiliary_ids,
        resource_events=with_saves,
        forward_flops=forward,
        backward_flops=backward,
        module_path=region.anchor.module_path,
        component_type=region.anchor.component_type,
        region_id=region.id,
    )


LINEAR_PROVENANCE = ProvenanceMatchRule(
    id="prov-linear",
    kind="region/linear",
    priority=5,
    component_type="Linear",
)

FUSED_LINEAR_REGION = FusedLinearRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/linear",
        kind="region/linear",
        priority=5,
        provenance_rule=LINEAR_PROVENANCE,
    ),
)
