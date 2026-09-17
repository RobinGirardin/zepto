"""Fused GatedRMSNorm region lowering: reference, default, hub-fla, and fla variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
from ....recipes.gated_rms_norm import (
    DEFAULT_GATED_RMS_NORM_RECIPE,
    GatedRMSNormRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import GATED_RMS_NORM_PATTERN, GATED_RMS_NORM_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "GatedRMSNorm fusion requires requested_capabilities={'fused'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _value_gate_tensors(
    estimation: RegionEstimationContext,
    graph: Graph,
    region: Region,
) -> tuple[Tensor, Tensor, Tensor]:
    value = estimation.input_tensors.get("value")
    gate = estimation.input_tensors.get("gate")
    output = estimation.output_tensors.get("output")
    if value is None and region.boundary_inputs:
        value = graph.edge(region.boundary_inputs[0]).tensor
    if gate is None and len(region.boundary_inputs) > 1:
        gate = graph.edge(region.boundary_inputs[1]).tensor
    if output is None and region.boundary_outputs:
        output = graph.edge(region.boundary_outputs[0]).tensor
    if value is None or gate is None or output is None:
        raise ValueError("GatedRMSNorm region requires value, gate, and output tensors")
    if value.shape != gate.shape:
        raise ValueError("GatedRMSNorm value and gate must share shape")
    return value, gate, output


@dataclass(frozen=True, slots=True)
class FusedGatedRMSNormRegionImplementation:
    """Lower a decomposed GatedRMSNorm chain as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GatedRMSNormRecipe = DEFAULT_GATED_RMS_NORM_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "GatedRMSNorm":
            return "not a GatedRMSNorm module invocation"
        if len(region.operation_ids) < 2:
            return "GatedRMSNorm fusion expects multiple ops"
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

        value, gate, output_tensor = _value_gate_tensors(
            estimation, graph, region
        )
        n = numel(value)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []

        if self.recipe.materialize_rstd:
            saved_rstd = f"region:{region.id}:rstd"
            rstd_tensor = Tensor(
                shape=self.recipe.rstd_shape(value.shape),
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
            if self.recipe.save_rstd:
                events.append(
                    ResourceEvent(ResourceEventKind.SAVE, saved_rstd)
                )
            auxiliary_edges.append(saved_rstd)

        requires_grad = value.requires_grad or gate.requires_grad
        forward_flops = self.recipe.forward_flops(n)
        backward_flops = self.recipe.backward_flops(
            n, requires_grad=requires_grad
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
        kind="region/gated_rms_norm",
        priority=priority,
        capabilities=frozenset({"fused"}),
        provenance_rule=GATED_RMS_NORM_PROVENANCE,
        pattern_rule=GATED_RMS_NORM_PATTERN,
    )


FUSED_GATED_RMS_NORM_REFERENCE = FusedGatedRMSNormRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_rms_norm/reference",
        priority=5,
    ),
    recipe=DEFAULT_GATED_RMS_NORM_RECIPE,
    hardware_gate="any",
)

FUSED_GATED_RMS_NORM_DEFAULT = FusedGatedRMSNormRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_rms_norm",
        priority=6,
    ),
    recipe=DEFAULT_GATED_RMS_NORM_RECIPE,
    hardware_gate="any",
)

FUSED_GATED_RMS_NORM_HUB_FLA = FusedGatedRMSNormRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_rms_norm/hub-fla",
        priority=7,
    ),
    recipe=DEFAULT_GATED_RMS_NORM_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_GATED_RMS_NORM_FLA = FusedGatedRMSNormRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_rms_norm/fla",
        priority=8,
    ),
    recipe=DEFAULT_GATED_RMS_NORM_RECIPE,
    hardware_gate="cuda_only",
)

GATED_RMS_NORM_REGIONS: tuple[FusedGatedRMSNormRegionImplementation, ...] = (
    FUSED_GATED_RMS_NORM_REFERENCE,
    FUSED_GATED_RMS_NORM_DEFAULT,
    FUSED_GATED_RMS_NORM_HUB_FLA,
    FUSED_GATED_RMS_NORM_FLA,
)
