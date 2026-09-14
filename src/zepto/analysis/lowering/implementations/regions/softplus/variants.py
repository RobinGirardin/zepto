"""Fused Softplus region lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.semantic.operations.helpers import numel
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import RegionEstimationContext, ensure_lowered_edge
from ....recipes.softplus import (
    DEFAULT_SOFTPLUS_RECIPE,
    XIELU_SCALAR_SOFTPLUS_RECIPE,
    SoftplusRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from .rules import SOFTPLUS_PATTERN, SOFTPLUS_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _rejects_scalar_capability(
    n: int,
    context: InvocationContext,
) -> str | None:
    if n == 1 and "scalar" in context.requested_capabilities:
        return "default softplus variant does not handle scalar capability"
    return None


def _requires_scalar_capability(
    n: int,
    context: InvocationContext,
) -> str | None:
    if "scalar" not in context.requested_capabilities:
        return "scalar variant requires requested_capabilities={'scalar'}"
    if n != 1:
        return "scalar softplus variant requires numel=1"
    return None


@dataclass(frozen=True, slots=True)
class FusedSoftplusRegionImplementation:
    """Lower ``Log(Add(1, Exp(X)))`` as one fused Softplus region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: SoftplusRecipe = DEFAULT_SOFTPLUS_RECIPE
    hardware_gate: HardwareGate = "any"
    scalar_gate: Literal["never", "required"] = "never"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if len(region.operation_ids) != 3:
            return "Softplus region expects three ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        expected = self.descriptor.pattern_rule
        if expected is None:
            return "missing pattern rule"
        if families != expected.op_families:
            return "unexpected operation sequence"

        output_tensor = None
        if region.boundary_outputs:
            output_tensor = graph.edge(region.boundary_outputs[0]).tensor
        if output_tensor is None:
            log_op = graph.node(region.operation_ids[2])
            if log_op.output_edges:
                output_tensor = graph.edge(log_op.output_edges[0]).tensor
        if output_tensor is None:
            return "missing output tensor"
        n = numel(output_tensor)

        if self.scalar_gate == "never":
            if reason := _rejects_scalar_capability(n, context):
                return reason
        elif self.scalar_gate == "required":
            if reason := _requires_scalar_capability(n, context):
                return reason

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
            raise ValueError("Softplus region requires boundary tensors")

        n = numel(output_tensor)
        output_id = region.boundary_outputs[0]
        input_id = region.boundary_inputs[0]
        lowered_out = edge_map[output_id]
        lowered_in = edge_map[input_id]

        events: list[ResourceEvent] = []
        release_edges: list[str] = []

        if self.recipe.save_input:
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out))
            if input_tensor.requires_grad:
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


FUSED_SOFTPLUS = FusedSoftplusRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/softplus",
        kind="region/softplus",
        priority=10,
        capabilities=frozenset({"softplus", "saved_input"}),
        provenance_rule=SOFTPLUS_PROVENANCE,
        pattern_rule=SOFTPLUS_PATTERN,
    ),
    recipe=DEFAULT_SOFTPLUS_RECIPE,
    hardware_gate="any",
    scalar_gate="never",
)

FUSED_SOFTPLUS_SCALAR = FusedSoftplusRegionImplementation(
    descriptor=RegionImplementationDescriptor(
        id="region/softplus/scalar",
        kind="region/softplus",
        priority=5,
        capabilities=frozenset({"softplus", "scalar"}),
        provenance_rule=SOFTPLUS_PROVENANCE,
        pattern_rule=SOFTPLUS_PATTERN,
    ),
    recipe=XIELU_SCALAR_SOFTPLUS_RECIPE,
    hardware_gate="any",
    scalar_gate="required",
)

SOFTPLUS_REGIONS: tuple[FusedSoftplusRegionImplementation, ...] = (
    FUSED_SOFTPLUS,
    FUSED_SOFTPLUS_SCALAR,
)
