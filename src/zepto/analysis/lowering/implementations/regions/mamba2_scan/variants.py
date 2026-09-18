"""Fused Mamba-2 selective SSM scan: reference, SSD chunk prefill, and decode variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    recurrent_state_port_events,
    register_auxiliary_edge,
    resolve_recurrent_scenario,
)
from ....recipes.mamba2_scan import (
    DECODE_MAMBA2_SCAN_RECIPE,
    DEFAULT_MAMBA2_SCAN_RECIPE,
    Mamba2ScanRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import MAMBA2_SCAN_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]

# Elided on-chip temps (research §8): a_bar_t, b_bar_t, decayed_t, update_t,
# weighted_t, softplus_intermediate — fused leaf must not allocate per-step HBM.
_ALLOWED_OP_FAMILIES = frozenset(
    {
        "cast",
        "concat",
        "split",
        "reshape",
        "transpose",
        "exp",
        "log",
        "add",
        "multiply",
        "reduce_sum",
        "repeat_kv",
        "parameter_scale",
        "parameter_bias",
    }
)


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "mamba2 scan fusion requires requested_capabilities={'fused'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _input_tensors(
    estimation: RegionEstimationContext,
    graph: Graph,
    region: Region,
) -> dict[str, Tensor]:
    if estimation.input_tensors:
        return dict(estimation.input_tensors)
    tensors: dict[str, Tensor] = {}
    for index, edge_id in enumerate(region.boundary_inputs):
        key = "input" if index == 0 else f"input{index}"
        tensors[key] = graph.edge(edge_id).tensor
    return tensors


def _resolve_scan_geometry(
    input_tensors: Mapping[str, Tensor],
) -> tuple[int, int, int, int, int]:
    rank3 = [t for t in input_tensors.values() if len(t.shape) == 3]
    rank2 = [t for t in input_tensors.values() if len(t.shape) == 2]
    if not rank2 or not rank3:
        raise ValueError("mamba2 scan region requires delta (S, h) and x/B/C tensors")
    delta = max(rank2, key=lambda t: t.shape[1])
    seq_len, num_heads = delta.shape
    seq_rank3 = [t for t in rank3 if t.shape[0] == seq_len]
    x_candidates = [t for t in seq_rank3 if t.shape[1] == num_heads]
    if not x_candidates:
        raise ValueError("mamba2 scan region requires rank-3 x tensor (S, h, p)")
    x = x_candidates[0]
    head_dim = x.shape[2]
    bc = [t for t in seq_rank3 if t.shape[1] != num_heads]
    if not bc:
        bc = [t for t in seq_rank3 if t is not x]
    if not bc:
        raise ValueError("mamba2 scan region requires rank-3 b tensor (S, G, N)")
    b = bc[0]
    num_groups = b.shape[1]
    state_size = b.shape[2]
    if num_heads % num_groups != 0:
        raise ValueError("num_heads must be divisible by num_groups")
    return seq_len, num_heads, head_dim, state_size, num_groups


def _scan_dims(
    estimation: RegionEstimationContext,
    graph: Graph,
    region: Region,
) -> tuple[int, int, int, int, int]:
    return _resolve_scan_geometry(_input_tensors(estimation, graph, region))


def _has_scan_state_in(
    estimation: RegionEstimationContext,
    *,
    seq_len: int,
) -> bool:
    return any(
        len(t.shape) == 3 and t.shape[0] != seq_len
        for t in estimation.input_tensors.values()
    )


def _is_decode_graph(
    estimation: RegionEstimationContext,
    context: InvocationContext,
    module_path: tuple[str, ...],
    *,
    region: Region,
    graph: Graph,
) -> bool:
    try:
        seq_len, _, _, _, _ = _scan_dims(estimation, graph, region)
    except ValueError:
        return False
    if seq_len != 1:
        return False
    if _has_scan_state_in(estimation, seq_len=seq_len):
        return True
    if len(region.boundary_inputs) >= 5:
        return True
    scenario = resolve_recurrent_scenario(context, module_path)
    return scenario is not None and scenario.kind == "mamba2"


@dataclass(frozen=True, slots=True)
class FusedMamba2ScanRegionImplementation:
    """Lower a ``SelectiveSSMScan`` module graph as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: Mamba2ScanRecipe = DEFAULT_MAMBA2_SCAN_RECIPE
    hardware_gate: HardwareGate = "any"
    decode_only: bool = False

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "SelectiveSSMScan":
            return "not a SelectiveSSMScan module invocation"
        if reason := _requires_fused(context):
            return reason
        if reason := _check_hardware_gate(self.hardware_gate, context):
            return reason
        for op_id in region.operation_ids:
            family = graph.node(op_id).operation_family
            if family not in _ALLOWED_OP_FAMILIES:
                return f"unexpected operation family {family!r}"
        input_tensors: dict[str, Tensor] = {}
        for index, edge_id in enumerate(region.boundary_inputs):
            key = "input" if index == 0 else f"input{index}"
            input_tensors[key] = graph.edge(edge_id).tensor
        estimation = RegionEstimationContext(
            region=region,
            phase=context.phase,
            input_tensors=input_tensors,
            output_tensors={},
            parameter_tensors={},
            precision=context.precision,
            state=context.state,
        )
        if self.decode_only:
            if not _is_decode_graph(
                estimation,
                context,
                region.anchor.module_path,
                region=region,
                graph=graph,
            ):
                return "decode variant requires S=1 with scan state port"
        elif _is_decode_graph(
            estimation,
            context,
            region.anchor.module_path,
            region=region,
            graph=graph,
        ):
            return "prefill variant incompatible with decode graph"
        return None

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

        seq_len, num_heads, head_dim, state_size, _num_groups = _scan_dims(
            estimation, graph, region
        )
        value_requires_grad = any(
            graph.edge(edge_id).tensor.requires_grad
            for edge_id in region.boundary_inputs
        ) or any(
            graph.edge(edge_id).tensor.requires_grad
            for edge_id in region.boundary_outputs
        )

        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]
        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []

        decode = self.decode_only or _is_decode_graph(
            estimation,
            context,
            region.anchor.module_path,
            region=region,
            graph=graph,
        )

        if not decode:
            workspace_id = f"region:{region.id}:ssd_chunk_scan_workspace"
            workspace_tensor = Tensor(
                shape=(0,),
                semantic_type="mamba2_ssd_chunk_workspace",
                dtype=DType.FP32,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                workspace_id,
                workspace_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=workspace_id,
                lowered_edges=lowered_edges,
            )
            events.append(
                ResourceEvent(ResourceEventKind.WORKSPACE, workspace_id)
            )
            auxiliary_edges.append(workspace_id)

        if (
            self.recipe.save_state_all_timesteps
            and value_requires_grad
            and not decode
        ):
            checkpoint_id = f"region:{region.id}:state_checkpoint"
            checkpoint_tensor = Tensor(
                shape=(seq_len, num_heads, head_dim, state_size),
                semantic_type="mamba2_state_checkpoint",
                dtype=DType.FP32,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                checkpoint_id,
                checkpoint_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=checkpoint_id,
                lowered_edges=lowered_edges,
            )
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, checkpoint_id)
            )
            events.append(ResourceEvent(ResourceEventKind.SAVE, checkpoint_id))
            auxiliary_edges.append(checkpoint_id)

        if len(region.boundary_outputs) > 1 and self.recipe.save_scan_state_out:
            state_out_id = region.boundary_outputs[1]
            ensure_lowered_edge(
                state_out_id, graph, context, lowered_edges, edge_map
            )
            lowered_state = edge_map[state_out_id]
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, lowered_state),
            )
            events.append(
                ResourceEvent(ResourceEventKind.PERSIST, lowered_state),
            )

        if decode:
            scenario = resolve_recurrent_scenario(
                context, region.anchor.module_path
            )
            if scenario is not None and scenario.kind == "mamba2":
                state_event, allocate, persist, aux_id = recurrent_state_port_events(
                    region_id=region.id,
                    scenario=scenario,
                )
                port_tensor = Tensor(
                    shape=(
                        scenario.num_heads,
                        scenario.head_dim,
                        scenario.state_dim,
                    ),
                    semantic_type="scan_state",
                    dtype=DType.FP32,
                    requires_grad=False,
                    persistent=True,
                )
                register_auxiliary_edge(
                    aux_id,
                    port_tensor,
                    role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                    context=context,
                    storage_id=aux_id,
                    lowered_edges=lowered_edges,
                )
                if state_port_collector is not None:
                    state_port_collector.append(state_event)
                events.extend([allocate, persist])
                auxiliary_edges.append(aux_id)

        forward_flops = self.recipe.forward_flops(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
        )
        backward_flops = self.recipe.backward_flops(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=state_size,
            requires_grad=value_requires_grad,
        )

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=region.operation_ids[0],
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=tuple(edge_map[eid] for eid in region.boundary_inputs),
            output_edges=tuple(
                edge_map[eid] for eid in region.boundary_outputs
            ),
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
        kind="region/mamba2_scan",
        priority=priority,
        capabilities=frozenset({"fused", "recurrent_state_port"}),
        provenance_rule=MAMBA2_SCAN_PROVENANCE,
        pattern_rule=None,
    )


FUSED_MAMBA2_SCAN_REFERENCE = FusedMamba2ScanRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/mamba2_scan/reference",
        priority=5,
    ),
    recipe=DEFAULT_MAMBA2_SCAN_RECIPE,
    hardware_gate="any",
)

FUSED_MAMBA2_SCAN_SSD_CHUNK = FusedMamba2ScanRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/mamba2_scan",
        priority=8,
    ),
    recipe=DEFAULT_MAMBA2_SCAN_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_MAMBA2_SCAN_DECODE = FusedMamba2ScanRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/mamba2_scan/decode",
        priority=9,
    ),
    recipe=DECODE_MAMBA2_SCAN_RECIPE,
    hardware_gate="cuda_only",
    decode_only=True,
)

MAMBA2_SCAN_REGIONS: tuple[FusedMamba2ScanRegionImplementation, ...] = (
    FUSED_MAMBA2_SCAN_REFERENCE,
    FUSED_MAMBA2_SCAN_SSD_CHUNK,
    FUSED_MAMBA2_SCAN_DECODE,
)
