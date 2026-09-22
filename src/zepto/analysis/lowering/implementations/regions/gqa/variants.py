"""Fused GQA region lowering: FlashAttention and SDPA variants."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    kv_state_port_events,
    register_auxiliary_edge,
    register_kv_cache_aux,
    resolve_kv_scenario,
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
from .rules import GQA_OP_SEQUENCES, GQA_PATTERN, GQA_PROVENANCE
from .shared import (
    attention_dims,
    attention_score_shape,
    flash_row_stats_shape,
    resolve_window_size,
)

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
        state_port_collector: list | None = None,
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

        batch, num_heads, seq_len, head_dim = attention_dims(output_tensor)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        window_size = resolve_window_size(region, graph)
        recipe = self.recipe
        if window_size is not None and isinstance(recipe, (GQARecipe, GQASDPAMathRecipe)):
            recipe = replace(recipe, window_size=window_size)

        events: list[ResourceEvent] = []
        auxiliary_edges: list[str] = []
        kv_scenario = resolve_kv_scenario(
            context=context,
            query_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            module_path=region.anchor.module_path,
            batch=batch,
        )
        is_decode = kv_scenario is not None and kv_scenario.is_decode
        flop_seq_len = (
            kv_scenario.cache_seq_len if is_decode else seq_len
        )

        if kv_scenario is not None and not is_decode:
            state_event, kv_alloc, kv_persist, kv_aux_id = kv_state_port_events(
                region_id=region.id,
                scenario=kv_scenario,
                seq_len=seq_len,
            )
            register_kv_cache_aux(
                aux_id=kv_aux_id,
                scenario=kv_scenario,
                seq_len=seq_len,
                context=context,
                lowered_edges=lowered_edges,
            )
            events.extend([kv_alloc, kv_persist])
            auxiliary_edges.append(kv_aux_id)
            if state_port_collector is not None:
                state_port_collector.append(state_event)

        if isinstance(recipe, GQASDPAMathRecipe) and recipe.save_P:
            scores_id = f"region:{region.id}:scores"
            p_id = f"region:{region.id}:P"
            attn_shape = attention_score_shape(batch, num_heads, seq_len)
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
                isinstance(recipe, GQARecipe)
                and recipe.save_row_stats
                and input_tensor.requires_grad
            ):
                saved_stats = f"region:{region.id}:row_stats"
                _register_attention_aux(
                    aux_id=saved_stats,
                    shape=flash_row_stats_shape(batch, num_heads, seq_len),
                    semantic_type="flash_row_stats",
                    context=context,
                    lowered_edges=lowered_edges,
                )
                events.append(ResourceEvent(ResourceEventKind.SAVE, saved_stats))
                auxiliary_edges.append(saved_stats)

        if is_decode and isinstance(recipe, GQARecipe):
            forward_flops = recipe.paged_forward_flops(
                num_heads=num_heads,
                cache_len=flop_seq_len,
                head_dim=head_dim,
            )
            backward_flops = 0
        else:
            forward_flops = recipe.forward_flops(
                num_heads=num_heads, seq_len=flop_seq_len, head_dim=head_dim
            )
            backward_flops = recipe.backward_flops(
                num_heads=num_heads,
                seq_len=flop_seq_len,
                head_dim=head_dim,
                requires_grad=input_tensor.requires_grad,
            )
        forward_flops *= batch
        backward_flops *= batch

        if context.phase == "backward":
            events.extend(_backward_release_saved_aux(auxiliary_edges))

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


def _backward_release_saved_aux(auxiliary_edges: list[str]) -> list[ResourceEvent]:
    """RELEASE saved-for-backward attention aux (row-stats / P), never kv_cache."""
    events: list[ResourceEvent] = []
    for aux_id in auxiliary_edges:
        if "kv_cache" in aux_id:
            continue
        if "row_stats" in aux_id or ":P" in aux_id:
            events.append(
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    aux_id,
                    phase="backward",
                )
            )
    return events


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
        provenance_rule=GQA_PROVENANCE,
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
