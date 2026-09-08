"""Fused GQA region lowering: flash2 and flash3 variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    register_auxiliary_edge,
)
from ....recipes.gqa import DEFAULT_GQA_RECIPE, GQARecipe
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import GQA_OP_SEQUENCES, GQA_PATTERN

HardwareGate = Literal["any", "cuda_only"]


def _requires_flash(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "gqa flash fusion requires requested_capabilities={'fused'}"
    if "flash" not in context.requested_capabilities:
        return "gqa flash fusion requires requested_capabilities={'flash'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _attention_dims(output_tensor: Tensor) -> tuple[int, int, int]:
    shape = output_tensor.shape
    if len(shape) != 3:
        raise ValueError(
            f"gqa flash region expects rank-3 context (h, S, d_h), got {shape}"
        )
    return int(shape[0]), int(shape[1]), int(shape[2])


@dataclass(frozen=True, slots=True)
class FusedGQARegionImplementation:
    """Lower the GQA attention core as one fused boundary-C region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GQARecipe = DEFAULT_GQA_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if len(region.operation_ids) != 10:
            return "gqa flash fusion expects exactly ten ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        if families not in GQA_OP_SEQUENCES:
            return "unexpected operation sequence"
        if reason := _requires_flash(context):
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
            raise ValueError("gqa region requires boundary tensors")

        num_heads, seq_len, head_dim = _attention_dims(output_tensor)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []

        if self.recipe.save_row_stats and input_tensor.requires_grad:
            saved_stats = f"region:{region.id}:row_stats"
            stats_tensor = Tensor(
                shape=(num_heads, seq_len, 2),
                semantic_type="flash_row_stats",
                dtype=DType.FP32,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                saved_stats,
                stats_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=saved_stats,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.SAVE, saved_stats))
            auxiliary_edges.append(saved_stats)

        forward_flops = self.recipe.forward_flops(
            num_heads=num_heads, seq_len=seq_len, head_dim=head_dim
        )
        backward_flops = self.recipe.backward_flops(
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            requires_grad=input_tensor.requires_grad,
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
        kind="region/gqa",
        priority=priority,
        capabilities=frozenset({"fused", "flash", "gqa"}),
        provenance_rule=None,
        pattern_rule=GQA_PATTERN,
    )


FUSED_GQA_FLASH2 = FusedGQARegionImplementation(
    descriptor=_descriptor(impl_id="region/gqa/flash2", priority=8),
    recipe=DEFAULT_GQA_RECIPE,
    hardware_gate="any",
)

FUSED_GQA_FLASH3 = FusedGQARegionImplementation(
    descriptor=_descriptor(impl_id="region/gqa/flash3", priority=10),
    recipe=DEFAULT_GQA_RECIPE,
    hardware_gate="cuda_only",
)

GQA_REGIONS: tuple[FusedGQARegionImplementation, ...] = (
    FUSED_GQA_FLASH2,
    FUSED_GQA_FLASH3,
)
