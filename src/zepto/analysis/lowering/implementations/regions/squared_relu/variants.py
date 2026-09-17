"""Fused ReLU² region lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.semantic.operations.helpers import numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import RegionEstimationContext, ensure_lowered_edge
from ....recipes.squared_relu import (
    DEFAULT_SQUARED_RELU_RECIPE,
    SquaredReLURecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from .rules import SQUARED_RELU_PATTERN

HardwareGate = Literal["any", "cuda_only"]


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


@dataclass(frozen=True, slots=True)
class FusedSquaredReLURegionImplementation:
    """Lower ``Multiply(ReLU(X), ReLU(X))`` as one fused ReLU² region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: SquaredReLURecipe = DEFAULT_SQUARED_RELU_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if len(region.operation_ids) != 2:
            return "ReLU² region expects two ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        expected = self.descriptor.pattern_rule
        if expected is None:
            return "missing pattern rule"
        if families != expected.op_families:
            return "unexpected operation sequence"
        return _check_hardware_gate(self.hardware_gate, context)

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
            raise ValueError("ReLU² region requires boundary tensors")

        n = numel(output_tensor)
        output_id = region.boundary_outputs[0]
        input_id = region.boundary_inputs[0]
        lowered_out = edge_map[output_id]
        lowered_in = edge_map[input_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        release_edges: list[str] = []

        if self.recipe.save_input and input_tensor.requires_grad:
            events.append(ResourceEvent(ResourceEventKind.SAVE, lowered_in))
            release_edges.append(lowered_in)

        forward_flops = self.recipe.forward_flops(n)
        backward_flops = self.recipe.backward_flops(
            n, requires_grad=input_tensor.requires_grad
        )

        if context.phase == "backward" and release_edges:
            events.append(
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    release_edges[0],
                    phase="backward",
                )
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


FUSED_SQUARED_RELU_REGION = FusedSquaredReLURegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/squared_relu",
        kind="region/squared_relu",
        priority=10,
        capabilities=frozenset({"squared_relu", "relu2", "saved_input"}),
        pattern_rule=SQUARED_RELU_PATTERN,
    ),
    recipe=DEFAULT_SQUARED_RELU_RECIPE,
    hardware_gate="any",
)

SQUARED_RELU_REGIONS: tuple[FusedSquaredReLURegionImplementation, ...] = (
    FUSED_SQUARED_RELU_REGION,
)
