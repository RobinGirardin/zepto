"""Fused logit soft-cap region lowering (reference until kernel-full)."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.semantic.operations.helpers import numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import RegionEstimationContext, ensure_lowered_edge
from ....recipes.logit_soft_cap import (
    DEFAULT_LOGIT_SOFT_CAP_RECIPE,
    LogitSoftCapRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from .rules import LOGIT_SOFT_CAP_PATTERN, LOGIT_SOFT_CAP_PROVENANCE


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "logit soft-cap fusion requires requested_capabilities={'fused'}"
    return None


@dataclass(frozen=True, slots=True)
class FusedLogitSoftCapRegionImplementation:
    """Lower decomposed soft-cap ops as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: LogitSoftCapRecipe = DEFAULT_LOGIT_SOFT_CAP_RECIPE

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "LogitSoftCap":
            return "not a LogitSoftCap module invocation"
        if len(region.operation_ids) != 4:
            return "logit soft-cap fusion expects 4 ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        expected = self.descriptor.pattern_rule
        if expected is None:
            return "missing pattern rule"
        if families != expected.op_families:
            return "unexpected operation sequence"
        if reason := _requires_fused(context):
            return reason
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

        input_tensor = estimation.input_tensors.get("input")
        output_tensor = estimation.output_tensors.get("output")
        if input_tensor is None and region.boundary_inputs:
            input_tensor = graph.edge(region.boundary_inputs[0]).tensor
        if output_tensor is None and region.boundary_outputs:
            output_tensor = graph.edge(region.boundary_outputs[0]).tensor
        if input_tensor is None or output_tensor is None:
            raise ValueError("logit soft-cap region requires boundary tensors")

        n = numel(output_tensor)
        output_id = region.boundary_outputs[0]
        input_id = region.boundary_inputs[0]
        lowered_out = edge_map[output_id]
        lowered_in = edge_map[input_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]

        forward_flops = self.recipe.forward_flops(n)
        backward_flops = self.recipe.backward_flops(
            n, requires_grad=input_tensor.requires_grad
        )

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=region.operation_ids[0],
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=(lowered_in,),
            output_edges=(lowered_out,),
            auxiliary_edges=(),
            resource_events=tuple(events),
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
        )


def _descriptor(*, impl_id: str, priority: int) -> RegionImplementationDescriptor:
    return RegionImplementationDescriptor(
        id=impl_id,
        kind="region/logit_softcap",
        priority=priority,
        capabilities=frozenset({"fused"}),
        provenance_rule=LOGIT_SOFT_CAP_PROVENANCE,
        pattern_rule=LOGIT_SOFT_CAP_PATTERN,
    )


FUSED_LOGIT_SOFT_CAP_REFERENCE = FusedLogitSoftCapRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/logit_softcap/reference", priority=5
    ),
    recipe=DEFAULT_LOGIT_SOFT_CAP_RECIPE,
)

LOGIT_SOFT_CAP_REGIONS: tuple[FusedLogitSoftCapRegionImplementation, ...] = (
    FUSED_LOGIT_SOFT_CAP_REFERENCE,
)
