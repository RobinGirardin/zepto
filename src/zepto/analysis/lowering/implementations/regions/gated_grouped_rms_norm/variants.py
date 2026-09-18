"""Fused GatedGroupedRMSNorm region lowering: reference, default, and eager-hf."""

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
from ....recipes.gated_grouped_rms_norm import (
    DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE,
    EAGER_HF_GATED_GROUPED_RMS_NORM_RECIPE,
    GatedGroupedRMSNormRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import (
    GATED_GROUPED_RMS_NORM_PATTERN,
    GATED_GROUPED_RMS_NORM_PROVENANCE,
)

HardwareGate = Literal["any", "cuda_only"]


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return (
            "GatedGroupedRMSNorm fusion requires requested_capabilities={'fused'}"
        )
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
        raise ValueError(
            "GatedGroupedRMSNorm region requires value, gate, and output tensors"
        )
    if value.shape != gate.shape:
        raise ValueError("GatedGroupedRMSNorm value and gate must share shape")
    return value, gate, output


def _num_groups_from_region(graph: Graph, region: Region) -> int:
    for op_id in region.operation_ids:
        op = graph.node(op_id)
        if op.operation_family != "add":
            continue
        for edge_id in op.output_edges:
            shape = graph.edge(edge_id).tensor.shape
            if len(shape) >= 2 and shape[-1] == 1:
                return int(shape[-2])
    raise ValueError("could not infer num_groups from grouped variance add")


def _num_row_groups(value: Tensor, num_groups: int) -> int:
    hidden = value.shape[-1]
    if hidden <= 0:
        return 0
    return numel(value) // hidden * num_groups


@dataclass(frozen=True, slots=True)
class FusedGatedGroupedRMSNormRegionImplementation:
    """Lower a decomposed GatedGroupedRMSNorm chain as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GatedGroupedRMSNormRecipe = DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "GatedGroupedRMSNorm":
            return "not a GatedGroupedRMSNorm module invocation"
        if len(region.operation_ids) < 2:
            return "GatedGroupedRMSNorm fusion expects multiple ops"
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

        value, gate, _output_tensor = _value_gate_tensors(
            estimation, graph, region
        )
        n = numel(value)
        num_groups = _num_groups_from_region(graph, region)
        num_row_groups = _num_row_groups(value, num_groups)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []

        requires_grad = value.requires_grad or gate.requires_grad
        if requires_grad and self.recipe.materialize_group_rstd:
            saved_group_rstd = f"region:{region.id}:group_rstd"
            group_rstd_tensor = Tensor(
                shape=self.recipe.group_rstd_shape(value.shape, num_groups),
                dtype=DType.FP32,
                semantic_type="group_rstd",
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                saved_group_rstd,
                group_rstd_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=saved_group_rstd,
                lowered_edges=lowered_edges,
            )
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, saved_group_rstd)
            )
            if self.recipe.save_group_rstd:
                events.append(
                    ResourceEvent(ResourceEventKind.SAVE, saved_group_rstd)
                )
            auxiliary_edges.append(saved_group_rstd)

        forward_flops = self.recipe.forward_flops(
            n, num_row_groups=num_row_groups
        )
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
        kind="region/gated_grouped_rms_norm",
        priority=priority,
        capabilities=frozenset({"fused"}),
        provenance_rule=GATED_GROUPED_RMS_NORM_PROVENANCE,
        pattern_rule=GATED_GROUPED_RMS_NORM_PATTERN,
    )


FUSED_GATED_GROUPED_RMS_NORM_REFERENCE = FusedGatedGroupedRMSNormRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_grouped_rms_norm/reference",
        priority=5,
    ),
    recipe=DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE,
    hardware_gate="any",
)

FUSED_GATED_GROUPED_RMS_NORM_DEFAULT = FusedGatedGroupedRMSNormRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_grouped_rms_norm",
        priority=6,
    ),
    recipe=DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE,
    hardware_gate="any",
)

FUSED_GATED_GROUPED_RMS_NORM_EAGER_HF = FusedGatedGroupedRMSNormRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_grouped_rms_norm/eager-hf",
        priority=5,
    ),
    recipe=EAGER_HF_GATED_GROUPED_RMS_NORM_RECIPE,
    hardware_gate="any",
)

GATED_GROUPED_RMS_NORM_REGIONS: tuple[
    FusedGatedGroupedRMSNormRegionImplementation, ...
] = (
    FUSED_GATED_GROUPED_RMS_NORM_REFERENCE,
    FUSED_GATED_GROUPED_RMS_NORM_DEFAULT,
    FUSED_GATED_GROUPED_RMS_NORM_EAGER_HF,
)
