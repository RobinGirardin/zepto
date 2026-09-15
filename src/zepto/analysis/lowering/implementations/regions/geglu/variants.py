"""Fused GeGLU region lowering: decomposed and Liger variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.metadata import TensorRole
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    register_auxiliary_edge,
)
from ....recipes.geglu import (
    DEFAULT_GEGLU_RECIPE,
    GEGLU_ERF_RECIPE,
    LIGER_GEGLU_RECIPE,
    GeGLURecipe,
)
from ....region import PatternMatchRule, Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import (
    GEGLU_ERF_PATTERN,
    GEGLU_PATTERN,
    GEGLU_PROVENANCE,
)

HardwareGate = Literal["any", "cuda_only"]


def _requires_fused_post_gemm(context: InvocationContext) -> str | None:
    if "fused_post_gemm" not in context.requested_capabilities:
        return "liger variant requires requested_capabilities={'fused_post_gemm'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _extract_dims(
    region: Region,
    graph: Graph,
    input_tensor: Tensor,
) -> tuple[int, int, int]:
    if len(input_tensor.shape) != 2:
        raise ValueError("GeGLU region expects rank-2 input (S, d)")
    seq_len, hidden_size = int(input_tensor.shape[0]), int(input_tensor.shape[1])
    gate_op = graph.node(region.operation_ids[0])
    if not gate_op.output_edges:
        raise ValueError("gate_proj missing output edge")
    gate_tensor = graph.edge(gate_op.output_edges[0]).tensor
    if len(gate_tensor.shape) != 2:
        raise ValueError("gate_proj output must be rank-2")
    intermediate_size = int(gate_tensor.shape[1])
    return seq_len, hidden_size, intermediate_size


@dataclass(frozen=True, slots=True)
class FusedGeGLURegionImplementation:
    """Lower a GeGLU MLP stack as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GeGLURecipe = DEFAULT_GEGLU_RECIPE
    hardware_gate: HardwareGate = "any"
    elide_gelu_temp: bool = False
    capability_check: str | None = None

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        anchor_type = region.anchor.component_type
        if anchor_type is not None and anchor_type != "GeGLU":
            return "not a GeGLU module invocation"
        if len(region.operation_ids) != 5:
            return "GeGLU fusion expects five ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        expected = self.descriptor.pattern_rule
        if expected is None:
            return "missing pattern rule"
        if families != expected.op_families:
            return "unexpected operation sequence"
        if self.capability_check == "fused_post_gemm":
            if reason := _requires_fused_post_gemm(context):
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
        if input_tensor is None and region.boundary_inputs:
            input_tensor = graph.edge(region.boundary_inputs[0]).tensor
        if input_tensor is None:
            raise ValueError("GeGLU region requires boundary input tensor")

        seq_len, hidden_size, intermediate_size = _extract_dims(
            region, graph, input_tensor
        )
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []
        release_edges: list[str] = []

        if self.recipe.save_gate:
            gate_id = f"region:{region.id}:gate"
            register_auxiliary_edge(
                gate_id,
                Tensor(
                    shape=(seq_len, intermediate_size),
                    semantic_type="geglu_gate",
                    requires_grad=input_tensor.requires_grad,
                    persistent=False,
                ),
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=gate_id,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, gate_id))
            if input_tensor.requires_grad:
                events.append(ResourceEvent(ResourceEventKind.SAVE, gate_id))
                release_edges.append(gate_id)
            auxiliary_edges.append(gate_id)

        if self.recipe.save_up:
            up_id = f"region:{region.id}:up"
            register_auxiliary_edge(
                up_id,
                Tensor(
                    shape=(seq_len, intermediate_size),
                    semantic_type="geglu_up",
                    requires_grad=input_tensor.requires_grad,
                    persistent=False,
                ),
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=up_id,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, up_id))
            if input_tensor.requires_grad:
                events.append(ResourceEvent(ResourceEventKind.SAVE, up_id))
                release_edges.append(up_id)
            auxiliary_edges.append(up_id)

        if not self.elide_gelu_temp:
            gelu_g_id = f"region:{region.id}:gelu_g"
            register_auxiliary_edge(
                gelu_g_id,
                Tensor(
                    shape=(seq_len, intermediate_size),
                    semantic_type="geglu_gelu_g",
                    requires_grad=input_tensor.requires_grad,
                    persistent=False,
                ),
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=gelu_g_id,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, gelu_g_id))
            auxiliary_edges.append(gelu_g_id)

        h_id = f"region:{region.id}:h"
        register_auxiliary_edge(
            h_id,
            Tensor(
                shape=(seq_len, intermediate_size),
                semantic_type="geglu_hidden",
                requires_grad=input_tensor.requires_grad,
                persistent=False,
            ),
            role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
            context=context,
            storage_id=h_id,
            lowered_edges=lowered_edges,
        )
        events.append(ResourceEvent(ResourceEventKind.ALLOCATE, h_id))
        if input_tensor.requires_grad and self.recipe.save_down_input:
            events.append(ResourceEvent(ResourceEventKind.SAVE, h_id))
            release_edges.append(h_id)
        auxiliary_edges.append(h_id)

        forward_flops = self.recipe.forward_flops(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        backward_flops = self.recipe.backward_flops(
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            requires_grad=input_tensor.requires_grad,
        )

        if context.phase == "backward" and release_edges:
            for edge in release_edges:
                events.append(
                    ResourceEvent(
                        ResourceEventKind.RELEASE,
                        edge,
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


def _descriptor(
    *,
    impl_id: str,
    priority: int,
    capabilities: frozenset[str],
    pattern_rule: PatternMatchRule,
) -> RegionImplementationDescriptor:
    return RegionImplementationDescriptor(
        id=impl_id,
        kind="region/geglu",
        priority=priority,
        capabilities=capabilities,
        provenance_rule=GEGLU_PROVENANCE,
        pattern_rule=pattern_rule,
    )


FUSED_GEGLU_DECOMPOSED = FusedGeGLURegionImplementation(
    descriptor=_descriptor(
        impl_id="region/geglu/decomposed",
        priority=8,
        capabilities=frozenset({"geglu", "decomposed"}),
        pattern_rule=GEGLU_PATTERN,
    ),
    recipe=DEFAULT_GEGLU_RECIPE,
    hardware_gate="any",
    elide_gelu_temp=False,
)

FUSED_GEGLU_DECOMPOSED_ERF = FusedGeGLURegionImplementation(
    descriptor=_descriptor(
        impl_id="region/geglu/decomposed-erf",
        priority=8,
        capabilities=frozenset({"geglu", "decomposed", "gelu_erf"}),
        pattern_rule=GEGLU_ERF_PATTERN,
    ),
    recipe=GEGLU_ERF_RECIPE,
    hardware_gate="any",
    elide_gelu_temp=False,
)

FUSED_GEGLU_LIGER = FusedGeGLURegionImplementation(
    descriptor=_descriptor(
        impl_id="region/geglu/liger",
        priority=10,
        capabilities=frozenset({"geglu", "fused_post_gemm"}),
        pattern_rule=GEGLU_PATTERN,
    ),
    recipe=LIGER_GEGLU_RECIPE,
    hardware_gate="cuda_only",
    elide_gelu_temp=True,
    capability_check="fused_post_gemm",
)

GEGLU_REGIONS: tuple[FusedGeGLURegionImplementation, ...] = (
    FUSED_GEGLU_DECOMPOSED,
    FUSED_GEGLU_DECOMPOSED_ERF,
    FUSED_GEGLU_LIGER,
)
