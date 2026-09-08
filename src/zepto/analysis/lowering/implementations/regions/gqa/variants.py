"""Fused GQA region lowering: FlashAttention and SDPA variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

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
from ....recipes.gqa import (
    DEFAULT_GQA_RECIPE,
    DEFAULT_SDPA_MATH_RECIPE,
    GQARecipe,
    GQASDPAMathRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import GQA_OP_SEQUENCES, GQA_PATTERN

HardwareGate = Literal["any", "cuda_only"]
GQARecipeUnion = Union[GQARecipe, GQASDPAMathRecipe]


def _sdpa_mode(context: InvocationContext) -> str:
    for key, val in context.state:
        if key == "sdpa_mode":
            return str(val)
    return "flash"


def _requires_flash(context: InvocationContext) -> str | None:
    if context.attention_backend == "sdpa":
        return "flash variants incompatible with attention_backend='sdpa'"
    if "fused" not in context.requested_capabilities:
        return "gqa flash fusion requires requested_capabilities={'fused'}"
    if "flash" not in context.requested_capabilities:
        return "gqa flash fusion requires requested_capabilities={'flash'}"
    return None


def _requires_sdpa(context: InvocationContext, mode: str) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "sdpa fusion requires requested_capabilities={'fused'}"
    if "sdpa" not in context.requested_capabilities:
        return "sdpa fusion requires requested_capabilities={'sdpa'}"
    if context.attention_backend != "sdpa":
        return "sdpa variants require attention_backend='sdpa'"
    if _sdpa_mode(context) != mode:
        return f"sdpa_mode mismatch (want {mode})"
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


def _register_attention_aux(
    *,
    aux_id: str,
    shape: tuple[int, ...],
    semantic_type: str,
    context: InvocationContext,
    lowered_edges: dict,
) -> None:
    register_auxiliary_edge(
        aux_id,
        Tensor(
            shape=shape,
            semantic_type=semantic_type,
            dtype=DType.FP32,
            requires_grad=False,
            persistent=False,
        ),
        role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
        context=context,
        storage_id=aux_id,
        lowered_edges=lowered_edges,
    )


@dataclass(frozen=True, slots=True)
class FusedGQARegionImplementation:
    """Lower the GQA attention core as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GQARecipeUnion = DEFAULT_GQA_RECIPE
    hardware_gate: HardwareGate = "any"
    sdpa_mode: str | None = None

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if len(region.operation_ids) != 10:
            return "gqa fusion expects exactly ten ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        if families not in GQA_OP_SEQUENCES:
            return "unexpected operation sequence"
        if self.sdpa_mode is not None:
            if reason := _requires_sdpa(context, self.sdpa_mode):
                return reason
        elif reason := _requires_flash(context):
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

        events: list[ResourceEvent] = []
        auxiliary_edges: list[str] = []

        if isinstance(self.recipe, GQASDPAMathRecipe) and self.recipe.save_P:
            scores_id = f"region:{region.id}:scores"
            p_id = f"region:{region.id}:P"
            attn_shape = (num_heads, seq_len, seq_len)
            _register_attention_aux(
                aux_id=scores_id,
                shape=attn_shape,
                semantic_type="attention_scores",
                context=context,
                lowered_edges=lowered_edges,
            )
            _register_attention_aux(
                aux_id=p_id,
                shape=attn_shape,
                semantic_type="attention_weights",
                context=context,
                lowered_edges=lowered_edges,
            )
            events.extend(
                [
                    ResourceEvent(ResourceEventKind.ALLOCATE, scores_id),
                    ResourceEvent(ResourceEventKind.ALLOCATE, p_id),
                    ResourceEvent(ResourceEventKind.SAVE, p_id),
                    ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
                ]
            )
            auxiliary_edges.extend([scores_id, p_id])
        else:
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out))
            if (
                isinstance(self.recipe, GQARecipe)
                and self.recipe.save_row_stats
                and input_tensor.requires_grad
            ):
                saved_stats = f"region:{region.id}:row_stats"
                _register_attention_aux(
                    aux_id=saved_stats,
                    shape=(num_heads, seq_len, 2),
                    semantic_type="flash_row_stats",
                    context=context,
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


def _descriptor(
    *,
    impl_id: str,
    priority: int,
    capabilities: frozenset[str],
) -> RegionImplementationDescriptor:
    return RegionImplementationDescriptor(
        id=impl_id,
        kind="region/gqa",
        priority=priority,
        capabilities=capabilities,
        provenance_rule=None,
        pattern_rule=GQA_PATTERN,
    )


_FLASH_CAPS = frozenset({"fused", "flash", "gqa"})
_SDPA_CAPS = frozenset({"fused", "sdpa", "gqa"})

FUSED_GQA_FLASH2 = FusedGQARegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gqa/flash2", priority=8, capabilities=_FLASH_CAPS
    ),
    recipe=DEFAULT_GQA_RECIPE,
    hardware_gate="any",
)

FUSED_GQA_FLASH3 = FusedGQARegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gqa/flash3", priority=10, capabilities=_FLASH_CAPS
    ),
    recipe=DEFAULT_GQA_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_GQA_SDPA_MATH = FusedGQARegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gqa/sdpa-math",
        priority=7,
        capabilities=_SDPA_CAPS,
    ),
    recipe=DEFAULT_SDPA_MATH_RECIPE,
    hardware_gate="any",
    sdpa_mode="math",
)

FUSED_GQA_SDPA_FLASH = FusedGQARegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gqa/sdpa-flash",
        priority=7,
        capabilities=_SDPA_CAPS,
    ),
    recipe=DEFAULT_GQA_RECIPE,
    hardware_gate="any",
    sdpa_mode="flash",
)

FUSED_GQA_SDPA_MEM_EFFICIENT = FusedGQARegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gqa/sdpa-mem-efficient",
        priority=6,
        capabilities=_SDPA_CAPS,
    ),
    recipe=DEFAULT_GQA_RECIPE,
    hardware_gate="any",
    sdpa_mode="mem_efficient",
)

GQA_REGIONS: tuple[FusedGQARegionImplementation, ...] = (
    FUSED_GQA_FLASH2,
    FUSED_GQA_FLASH3,
    FUSED_GQA_SDPA_MATH,
    FUSED_GQA_SDPA_FLASH,
    FUSED_GQA_SDPA_MEM_EFFICIENT,
)
