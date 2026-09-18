"""Fused GELU region lowering (tanh and exact erf variants)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.semantic.operations.helpers import numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import RegionEstimationContext, ensure_lowered_edge
from ....recipes.gelu import GELU_ERF_RECIPE, GELU_TANH_RECIPE, GeluRecipe
from ....region import Region
from ....registry import RegionImplementationDescriptor
from .rules import (
    GELU_ERF_PATTERN,
    GELU_ERF_PROVENANCE,
    GELU_TANH_PATTERN,
    GELU_TANH_PROVENANCE,
)

HardwareGate = Literal["any", "cuda_only"]


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


@dataclass(frozen=True, slots=True)
class FusedGeluRegionImplementation:
    """Lower a decomposed GELU chain as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GeluRecipe = GELU_TANH_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        expected = self.descriptor.pattern_rule
        if expected is None:
            return "missing pattern rule"
        if len(region.operation_ids) != len(expected.op_families):
            return f"GELU region expects {len(expected.op_families)} ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
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
            raise ValueError("GELU region requires boundary tensors")

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


FUSED_GELU_TANH = FusedGeluRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/gelu",
        kind="region/gelu",
        priority=10,
        capabilities=frozenset({"gelu", "saved_input"}),
        provenance_rule=GELU_TANH_PROVENANCE,
        pattern_rule=GELU_TANH_PATTERN,
    ),
    recipe=GELU_TANH_RECIPE,
    hardware_gate="any",
)

FUSED_GELU_ERF = FusedGeluRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/gelu_erf",
        kind="region/gelu_erf",
        priority=10,
        capabilities=frozenset({"gelu", "saved_input"}),
        provenance_rule=GELU_ERF_PROVENANCE,
        pattern_rule=GELU_ERF_PATTERN,
    ),
    recipe=GELU_ERF_RECIPE,
    hardware_gate="any",
)

GELU_REGIONS: tuple[FusedGeluRegionImplementation, ...] = (
    FUSED_GELU_TANH,
    FUSED_GELU_ERF,
)
