"""Fused layer-norm region lowering via hybrid discovery."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.semantic.metadata import TensorRole
from zepto.semantic.operations.helpers import numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind
from ...context import InvocationContext
from ...helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    register_auxiliary_edge,
)
from ...region import (
    PatternConstraint,
    PatternMatchRule,
    ProvenanceMatchRule,
    Region,
)
from ...registry import RegionImplementationDescriptor
from ...role import RoleContext


LAYERNORM_PATTERN = PatternMatchRule(
    id="pat-layernorm-decomposed",
    kind="region/layernorm",
    priority=5,
    op_families=(
        "reduce_sum",
        "subtract",
        "multiply",
        "add",
        "square_root",
        "divide",
    ),
    edge_constraints=(
        (0, 1, "output_to_second_input"),
        (1, 2, "output_to_input"),
        (2, 3, "output_to_input"),
        (3, 4, "output_to_input"),
        (4, 5, "output_to_second_input"),
        (1, 5, "output_to_first_input"),
    ),
    constraints=(
        PatternConstraint(kind="parameter_count", max_parameters=2),
    ),
)

LAYERNORM_PROVENANCE = ProvenanceMatchRule(
    id="prov-layernorm",
    kind="region/layernorm",
    priority=5,
    component_type="LayerNorm",
)


@dataclass(frozen=True, slots=True)
class FusedLayerNormRegionImplementation:
    """Lower a decomposed layer-norm primitive chain as one fused region."""

    descriptor: RegionImplementationDescriptor

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        del context
        if region.anchor.component_type != "LayerNorm":
            return "not a LayerNorm module invocation"
        if len(region.operation_ids) < 2:
            return "layer norm fusion expects multiple ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        expected = self.descriptor.pattern_rule
        if expected is None:
            return "missing pattern rule"
        if families != expected.op_families:
            return "unexpected operation sequence"
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
            raise ValueError("LayerNorm region requires boundary tensors")

        n = numel(input_tensor)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]
        saved_mean = f"region:{region.id}:mean"
        saved_inv_std = f"region:{region.id}:inv_std"

        register_auxiliary_edge(
            saved_mean,
            output_tensor,
            role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
            context=context,
            storage_id=saved_mean,
            lowered_edges=lowered_edges,
        )
        register_auxiliary_edge(
            saved_inv_std,
            output_tensor,
            role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
            context=context,
            storage_id=saved_inv_std,
            lowered_edges=lowered_edges,
        )

        forward_flops = 5 * n
        backward_flops = 5 * n if input_tensor.requires_grad else 0

        events: tuple[ResourceEvent, ...] = (
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ResourceEvent(ResourceEventKind.ALLOCATE, saved_mean),
            ResourceEvent(ResourceEventKind.SAVE, saved_mean),
            ResourceEvent(ResourceEventKind.ALLOCATE, saved_inv_std),
            ResourceEvent(ResourceEventKind.SAVE, saved_inv_std),
        )

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=region.operation_ids[0],
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=tuple(edge_map[eid] for eid in region.boundary_inputs),
            output_edges=(lowered_out,),
            auxiliary_edges=(saved_mean, saved_inv_std),
            resource_events=events,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
        )


FUSED_LAYERNORM_REGION = FusedLayerNormRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/layernorm",
        kind="region/layernorm",
        priority=8,
        provenance_rule=LAYERNORM_PROVENANCE,
        pattern_rule=LAYERNORM_PATTERN,
    ),
)
