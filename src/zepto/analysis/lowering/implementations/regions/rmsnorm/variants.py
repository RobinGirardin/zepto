"""Fused RMSNorm region lowering: reference, liger, hub-xpu, hub-mps."""

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
from ....recipes.rmsnorm import (
    DEFAULT_RMSNORM_RECIPE,
    HUB_MPS_INFERENCE_RECIPE,
    RMSNormRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import RMSNORM_PATTERN, RMSNORM_PROVENANCE

HardwareGate = Literal["any", "exclude_xpu_mps", "xpu_only", "mps_only"]


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    hw = context.hardware
    if gate == "xpu_only" and hw != "xpu":
        return "hub-xpu requires hardware='xpu'"
    if gate == "mps_only" and hw != "mps":
        return "hub-mps requires hardware='mps'"
    if gate == "exclude_xpu_mps" and hw in ("xpu", "mps"):
        return "liger hub is not routed on xpu/mps"
    return None


@dataclass(frozen=True, slots=True)
class FusedRMSNormRegionImplementation:
    """Lower a decomposed RMSNorm chain as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: RMSNormRecipe = DEFAULT_RMSNORM_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "RMSNorm":
            return "not an RMSNorm module invocation"
        if len(region.operation_ids) < 2:
            return "RMSNorm fusion expects multiple ops"
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
            raise ValueError("RMSNorm region requires boundary tensors")

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
            if self.recipe.save_rstd:
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
        kind="region/rmsnorm",
        priority=priority,
        capabilities=frozenset({"fused"}),
        provenance_rule=RMSNORM_PROVENANCE,
        pattern_rule=RMSNORM_PATTERN,
    )


FUSED_RMSNORM_REFERENCE = FusedRMSNormRegionImplementation(
    descriptor=_descriptor(impl_id="region/rmsnorm/reference", priority=5),
    recipe=DEFAULT_RMSNORM_RECIPE,
    hardware_gate="any",
)

FUSED_RMSNORM_HUB_XPU = FusedRMSNormRegionImplementation(
    descriptor=_descriptor(impl_id="region/rmsnorm/hub-xpu", priority=7),
    recipe=DEFAULT_RMSNORM_RECIPE,
    hardware_gate="xpu_only",
)

FUSED_RMSNORM_HUB_MPS = FusedRMSNormRegionImplementation(
    descriptor=_descriptor(impl_id="region/rmsnorm/hub-mps", priority=7),
    recipe=HUB_MPS_INFERENCE_RECIPE,
    hardware_gate="mps_only",
)

FUSED_RMSNORM_LIGER = FusedRMSNormRegionImplementation(
    descriptor=_descriptor(impl_id="region/rmsnorm/liger", priority=8),
    recipe=DEFAULT_RMSNORM_RECIPE,
    hardware_gate="exclude_xpu_mps",
)

RMSNORM_REGIONS: tuple[FusedRMSNormRegionImplementation, ...] = (
    FUSED_RMSNORM_REFERENCE,
    FUSED_RMSNORM_HUB_XPU,
    FUSED_RMSNORM_HUB_MPS,
    FUSED_RMSNORM_LIGER,
)
