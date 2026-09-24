"""Fused ReLU region lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.helpers import includes_backward, numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    register_auxiliary_edge,
)
from ....recipes.relu import (
    DEFAULT_RELU_RECIPE,
    SAVED_INPUT_RELU_RECIPE,
    ReLURecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import RELU_PATTERN

SavedInputGate = Literal["never", "required"]


def _requires_saved_input(gate: SavedInputGate, context: InvocationContext) -> str | None:
    if gate == "required" and "saved_input" not in context.requested_capabilities:
        return "saved_input variant requires requested_capabilities={'saved_input'}"
    if gate == "never" and "saved_input" in context.requested_capabilities:
        return "default ReLU variant does not handle saved_input capability"
    return None


@dataclass(frozen=True, slots=True)
class ReLURegionImplementation:
    """Lower maximum(left, 0) as one fused region with mask or input retention."""

    descriptor: RegionImplementationDescriptor
    recipe: ReLURecipe = DEFAULT_RELU_RECIPE
    saved_input_gate: SavedInputGate = "never"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if len(region.operation_ids) != 1:
            return "ReLU region expects one operation"
        structural = graph.node(region.operation_ids[0])
        if structural.operation_family != "maximum":
            return "not maximum"
        if len(structural.input_edges) != 2:
            return "maximum expects two inputs"
        return _requires_saved_input(self.saved_input_gate, context)

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
        lowered_in = edge_map[left_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []
        release_edges: list[str] = []

        if self.recipe.save_relu_mask:
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
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, lowered_mask))
            events.append(ResourceEvent(ResourceEventKind.SAVE, lowered_mask))
            auxiliary_edges.append(lowered_mask)
            release_edges.append(lowered_mask)
        elif self.recipe.save_input and left_tensor.requires_grad:
            events.append(ResourceEvent(ResourceEventKind.SAVE, lowered_in))
            release_edges.append(lowered_in)

        forward_flops = self.recipe.forward_flops(n)
        backward_flops = self.recipe.backward_flops(
            n, requires_grad=left_tensor.requires_grad
        )

        if includes_backward(context.phase) and release_edges:
            events.append(
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    release_edges[0],
                    phase="backward",
                )
            )

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=structural.id,
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=(lowered_in,),
            output_edges=(lowered_out,),
            auxiliary_edges=tuple(auxiliary_edges),
            resource_events=tuple(events),
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
        )


RELU_REGION = ReLURegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/relu",
        kind="region/relu",
        priority=10,
        capabilities=frozenset({"relu", "mask_retention"}),
        pattern_rule=RELU_PATTERN,
    ),
    recipe=DEFAULT_RELU_RECIPE,
    saved_input_gate="never",
)

RELU_SAVED_INPUT = ReLURegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/relu/saved_input",
        kind="region/relu",
        priority=10,
        capabilities=frozenset({"relu", "saved_input"}),
        pattern_rule=RELU_PATTERN,
    ),
    recipe=SAVED_INPUT_RELU_RECIPE,
    saved_input_gate="required",
)

RELU_REGIONS: tuple[ReLURegionImplementation, ...] = (
    RELU_REGION,
    RELU_SAVED_INPUT,
)
