"""ReLU region lowering via pattern discovery."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.helpers import numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind
from ...context import InvocationContext
from ...helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    register_auxiliary_edge,
)
from ...region import PatternConstraint, PatternMatchRule, Region
from ...registry import RegionImplementationDescriptor
from ...role import RoleContext


@dataclass(frozen=True, slots=True)
class ReLURegionImplementation:
    """Lower maximum(left, 0) as one fused region with mask retention."""

    descriptor: RegionImplementationDescriptor

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        del context
        if len(region.operation_ids) != 1:
            return "ReLU region expects one operation"
        structural = graph.node(region.operation_ids[0])
        if structural.operation_family != "maximum":
            return "not maximum"
        if len(structural.input_edges) != 2:
            return "maximum expects two inputs"
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
        structural = graph.node(region.operation_ids[0])
        left_id, _right_id = structural.input_edges
        output_id = structural.output_edges[0]
        left_port, _right_port = structural.input_ports
        (output_port,) = structural.output_ports

        ensure_lowered_edge(
            left_id,
            graph,
            context,
            lowered_edges,
            edge_map,
            node=structural,
            port_name=left_port.name,
            port_direction="input",
        )
        ensure_lowered_edge(
            output_id,
            graph,
            context,
            lowered_edges,
            edge_map,
            node=structural,
            port_name=output_port.name,
            port_direction="output",
        )

        left_tensor = estimation.input_tensors.get("input")
        output_tensor = estimation.output_tensors.get("output")
        if left_tensor is None:
            left_tensor = graph.edge(left_id).tensor
        if output_tensor is None:
            output_tensor = graph.edge(output_id).tensor

        n = numel(output_tensor)
        lowered_out = edge_map[output_id]
        lowered_mask = f"region:{region.id}:mask"

        mask_tensor = Tensor(
            shape=output_tensor.shape,
            semantic_type="relu_mask",
            dtype=DType.BOOL,
            requires_grad=False,
            persistent=False,
        )
        register_auxiliary_edge(
            lowered_mask,
            mask_tensor,
            role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
            context=context,
            storage_id=lowered_mask,
            lowered_edges=lowered_edges,
        )

        forward_flops = n
        backward_flops = n if left_tensor.requires_grad else 0

        events: tuple[ResourceEvent, ...] = (
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_mask),
            ResourceEvent(ResourceEventKind.SAVE, lowered_mask),
        )
        if context.phase == "backward":
            events = (
                *events,
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    lowered_mask,
                    phase="backward",
                ),
            )

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=structural.id,
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=(edge_map[left_id],),
            output_edges=(lowered_out,),
            auxiliary_edges=(lowered_mask,),
            resource_events=events,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
        )


RELU_PATTERN = PatternMatchRule(
    id="pat-relu",
    kind="region/relu",
    priority=10,
    op_families=("maximum",),
    constraints=(PatternConstraint(kind="right_operand_is_zero"),),
)

RELU_REGION = ReLURegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/relu",
        kind="region/relu",
        priority=10,
        capabilities=frozenset({"relu", "mask_retention"}),
        pattern_rule=RELU_PATTERN,
    ),
)
