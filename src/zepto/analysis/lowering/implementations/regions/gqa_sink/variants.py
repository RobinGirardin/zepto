"""Fused GQA-sink region lowering: FlashAttention variants for GPT-OSS."""

from __future__ import annotations

from dataclasses import dataclass, replace

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    kv_state_port_events,
    register_kv_cache_aux,
    resolve_kv_scenario,
)
from ....recipes.gqa_sink import DEFAULT_GQA_SINK_RECIPE, GQASinkRecipe
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ..gqa.shared import (
    attention_dims,
    flash_row_stats_shape,
    resolve_window_size,
)
from ..gqa.variants import (
    HardwareGate,
    _check_hardware_gate,
    _register_attention_aux,
    _requires_flash,
)
from .rules import GQA_SINK_OP_SEQUENCES, GQA_SINK_PATTERN
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind


@dataclass(frozen=True, slots=True)
class FusedGQASinkRegionImplementation:
    """Lower the GQA sink-softmax attention core as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GQASinkRecipe = DEFAULT_GQA_SINK_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if len(region.operation_ids) != 7:
            return "gqa-sink fusion expects exactly seven ops"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        if families not in GQA_SINK_OP_SEQUENCES:
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
            raise ValueError("gqa-sink region requires boundary tensors")

        batch, num_heads, seq_len, head_dim = attention_dims(output_tensor)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        window_size = resolve_window_size(region, graph)
        recipe = self.recipe
        if window_size is not None:
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
        flop_seq_len = kv_scenario.cache_seq_len if is_decode else seq_len

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

        events.append(ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out))
        if recipe.save_row_stats and input_tensor.requires_grad:
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
        kind="region/gqa-sink",
        priority=priority,
        capabilities=capabilities,
        provenance_rule=None,
        pattern_rule=GQA_SINK_PATTERN,
    )


_FLASH_CAPS = frozenset({"fused", "flash", "gqa"})

FUSED_GQA_SINK_FLASH2 = FusedGQASinkRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gqa-sink/flash2", priority=8, capabilities=_FLASH_CAPS
    ),
    recipe=DEFAULT_GQA_SINK_RECIPE,
    hardware_gate="any",
)

FUSED_GQA_SINK_FLASH3 = FusedGQASinkRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gqa-sink/flash3", priority=10, capabilities=_FLASH_CAPS
    ),
    recipe=DEFAULT_GQA_SINK_RECIPE,
    hardware_gate="cuda_only",
)

GQA_SINK_REGIONS: tuple[FusedGQASinkRegionImplementation, ...] = (
    FUSED_GQA_SINK_FLASH2,
    FUSED_GQA_SINK_FLASH3,
)
