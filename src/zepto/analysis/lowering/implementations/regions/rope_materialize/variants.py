"""RoPEMaterialize region lowering."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import RegionEstimationContext, ensure_lowered_edge
from ....recipes.rope import (
    DEFAULT_ROPE_MATERIALIZE_RECIPE,
    RoPEMaterializeRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from .rules import ROPE_MATERIALIZE_PROVENANCE


@dataclass(frozen=True, slots=True)
class RoPEMaterializeRegionImplementation:
    """Replace decomposed freq→cos/sin ops with §8 materialize FLOPs."""

    descriptor: RegionImplementationDescriptor
    recipe: RoPEMaterializeRecipe = DEFAULT_ROPE_MATERIALIZE_RECIPE

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        del graph, context
        if region.anchor.component_type != "RoPEMaterialize":
            return "not a RoPEMaterialize module invocation"
        if len(region.operation_ids) < 1:
            return "RoPEMaterialize region expects ops"
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
        for edge_id in region.boundary_inputs:
            ensure_lowered_edge(
                edge_id, graph, context, lowered_edges, edge_map
            )
        for edge_id in region.boundary_outputs:
            ensure_lowered_edge(
                edge_id, graph, context, lowered_edges, edge_map
            )

        cos_tensor = graph.edge(region.boundary_outputs[0]).tensor
        seq_len, head_dim = cos_tensor.shape[0], cos_tensor.shape[1]

        output_edges = tuple(edge_map[eid] for eid in region.boundary_outputs)
        events = tuple(
            ResourceEvent(ResourceEventKind.ALLOCATE, edge_map[eid])
            for eid in region.boundary_outputs
        )

        forward_flops = self.recipe.forward_flops(
            seq_len=seq_len, head_dim=head_dim
        )
        backward_flops = self.recipe.backward_flops(requires_grad=False)

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=region.operation_ids[0],
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=tuple(
                edge_map[eid] for eid in region.boundary_inputs
            ),
            output_edges=output_edges,
            auxiliary_edges=(),
            resource_events=events,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
        )


def _descriptor(*, impl_id: str, priority: int) -> RegionImplementationDescriptor:
    return RegionImplementationDescriptor(
        id=impl_id,
        kind="region/rope_materialize",
        priority=priority,
        provenance_rule=ROPE_MATERIALIZE_PROVENANCE,
    )


ROPE_MATERIALIZE_REFERENCE = RoPEMaterializeRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/rope_materialize/reference", priority=5
    ),
)

ROPE_MATERIALIZE_REGIONS: tuple[RoPEMaterializeRegionImplementation, ...] = (
    ROPE_MATERIALIZE_REFERENCE,
)
