"""Fused L2-normalize region lowering: reference and FLA-aligned variants."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.helpers import numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    register_auxiliary_edge,
)
from ....recipes.l2_normalize import DEFAULT_L2_NORMALIZE_RECIPE, L2NormalizeRecipe
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import L2_NORMALIZE_PATTERN, L2_NORMALIZE_PROVENANCE


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "L2-normalize fusion requires requested_capabilities={'fused'}"
    return None


@dataclass(frozen=True, slots=True)
class FusedL2NormalizeRegionImplementation:
    """Lower a decomposed L2-normalize chain as one fused region leaf.

    Backward resource handling may append ``LOAD(rstd)`` when backward runs;
    forward-only graphs omit ``SAVE(rstd)`` when ``requires_grad`` is false.
    """

    descriptor: RegionImplementationDescriptor
    recipe: L2NormalizeRecipe = DEFAULT_L2_NORMALIZE_RECIPE

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "L2Normalize":
            return "not an L2Normalize module invocation"
        if len(region.operation_ids) != 7:
            return "L2-normalize fusion expects 7 ops"
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
            raise ValueError("L2-normalize region requires boundary tensors")

        n = numel(input_tensor)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []

        if self.recipe.materialize_rstd:
            saved_rstd = f"region:{region.id}:rstd"
            rstd_tensor = Tensor(
                shape=self.recipe.rstd_shape(input_tensor.shape),
                dtype=DType.FP32,
                semantic_type="rstd",
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                saved_rstd,
                rstd_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=saved_rstd,
                lowered_edges=lowered_edges,
            )
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, saved_rstd)
            )
            if self.recipe.save_rstd and input_tensor.requires_grad:
                events.append(
                    ResourceEvent(ResourceEventKind.SAVE, saved_rstd)
                )
            auxiliary_edges.append(saved_rstd)

        forward_flops = self.recipe.forward_flops(n)
        backward_flops = self.recipe.backward_flops(
            n, requires_grad=input_tensor.requires_grad
        )

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=region.operation_ids[0],
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=tuple(edge_map[eid] for eid in region.boundary_inputs),
            output_edges=(lowered_out,),
            auxiliary_edges=tuple(auxiliary_edges),
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
        kind="region/l2_normalize",
        priority=priority,
        capabilities=frozenset({"fused"}),
        provenance_rule=L2_NORMALIZE_PROVENANCE,
        pattern_rule=L2_NORMALIZE_PATTERN,
    )


FUSED_L2_NORMALIZE_REFERENCE = FusedL2NormalizeRegionImplementation(
    descriptor=_descriptor(impl_id="region/l2_normalize/reference", priority=5),
    recipe=DEFAULT_L2_NORMALIZE_RECIPE,
)

FUSED_L2_NORMALIZE_FLA = FusedL2NormalizeRegionImplementation(
    descriptor=_descriptor(impl_id="region/l2_normalize/fla", priority=8),
    recipe=DEFAULT_L2_NORMALIZE_RECIPE,
)

L2_NORMALIZE_REGIONS: tuple[FusedL2NormalizeRegionImplementation, ...] = (
    FUSED_L2_NORMALIZE_REFERENCE,
    FUSED_L2_NORMALIZE_FLA,
)
