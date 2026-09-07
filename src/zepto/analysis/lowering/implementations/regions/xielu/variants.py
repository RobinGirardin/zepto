"""Fused xIELU region lowering: reference and cuda variants."""

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
from ....recipes.xielu import DEFAULT_XIELU_RECIPE, XIELURecipe
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import XIELU_PATTERN, XIELU_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "xIELU fusion requires requested_capabilities={'fused'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


@dataclass(frozen=True, slots=True)
class FusedXIELURegionImplementation:
    """Lower a decomposed xIELU chain as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: XIELURecipe = DEFAULT_XIELU_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "XIELU":
            return "not an XIELU module invocation"
        if len(region.operation_ids) < 2:
            return "xIELU fusion expects multiple ops"
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

        input_tensor = estimation.input_tensors.get("input")
        output_tensor = estimation.output_tensors.get("output")
        if input_tensor is None and region.boundary_inputs:
            input_tensor = graph.edge(region.boundary_inputs[0]).tensor
        if output_tensor is None and region.boundary_outputs:
            output_tensor = graph.edge(region.boundary_outputs[0]).tensor
        if input_tensor is None or output_tensor is None:
            raise ValueError("xIELU region requires boundary tensors")

        n = numel(output_tensor)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []

        if self.recipe.save_sign_mask:
            saved_mask = f"region:{region.id}:sign_mask"
            mask_tensor = Tensor(
                shape=output_tensor.shape,
                semantic_type="xielu_sign_mask",
                dtype=DType.BOOL,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                saved_mask,
                mask_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=saved_mask,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, saved_mask))
            events.append(ResourceEvent(ResourceEventKind.SAVE, saved_mask))
            auxiliary_edges.append(saved_mask)

        forward_flops = self.recipe.forward_flops(n)
        backward_flops = self.recipe.backward_flops(
            n, requires_grad=input_tensor.requires_grad
        )

        if context.phase == "backward" and auxiliary_edges:
            events.append(
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    auxiliary_edges[0],
                    phase="backward",
                )
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
        kind="region/xielu",
        priority=priority,
        capabilities=frozenset({"fused", "xielu"}),
        provenance_rule=XIELU_PROVENANCE,
        pattern_rule=XIELU_PATTERN,
    )


FUSED_XIELU_REFERENCE = FusedXIELURegionImplementation(
    descriptor=_descriptor(impl_id="region/xielu/reference", priority=5),
    recipe=DEFAULT_XIELU_RECIPE,
    hardware_gate="any",
)

FUSED_XIELU_CUDA = FusedXIELURegionImplementation(
    descriptor=_descriptor(impl_id="region/xielu/cuda", priority=8),
    recipe=DEFAULT_XIELU_RECIPE,
    hardware_gate="cuda_only",
)

XIELU_REGIONS: tuple[FusedXIELURegionImplementation, ...] = (
    FUSED_XIELU_REFERENCE,
    FUSED_XIELU_CUDA,
)
