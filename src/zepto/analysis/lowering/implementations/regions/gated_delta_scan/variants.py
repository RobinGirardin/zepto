"""Fused gated delta scan region lowering: reference, FLA chunk, and decode variants."""

from __future__ import annotations

from collections import Counter
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
    with_batch_prefix,
)
from ....recipes.gated_delta_scan import (
    DECODE_GATED_DELTA_SCAN_RECIPE,
    DEFAULT_GATED_DELTA_SCAN_RECIPE,
    GatedDeltaScanRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import GATED_DELTA_SCAN_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]

_ALLOWED_OP_FAMILIES = frozenset(
    {
        "cast",
        "concat",
        "split",
        "reshape",
        "transpose",
        "exp",
        "multiply",
        "matmul",
        "subtract",
        "add",
    }
)


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "gated delta scan fusion requires requested_capabilities={'fused'}"
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
    rank4 = [t for t in input_tensors.values() if len(t.shape) == 4]
    rank3 = [t for t in input_tensors.values() if len(t.shape) == 3]
    rank2 = [t for t in input_tensors.values() if len(t.shape) == 2]

    if rank4:
        rank3_prefix = [t for t in rank3]
        if rank3_prefix:
            beta = max(rank3_prefix, key=lambda t: t.shape[1] * t.shape[-1])
            batch, seq_len, num_heads = (int(dim) for dim in beta.shape)
            qkv = [
                t for t in rank4 if t.shape[:3] == (batch, seq_len, num_heads)
            ]
            if qkv:
                key_dim = int(qkv[0].shape[3])
                value_dim = max(int(t.shape[3]) for t in qkv)
                return batch, seq_len, num_heads, key_dim, value_dim
        query = max(rank4, key=lambda t: t.shape[1])
        batch, seq_len, num_heads, key_dim = (int(dim) for dim in query.shape)
        value_dim = max(int(t.shape[-1]) for t in rank4)
        return batch, seq_len, num_heads, key_dim, value_dim

    if not rank3:
        raise ValueError("gated delta scan region requires rank-3 Q/K/V tensors")
    counts: Counter[int] = Counter()
    for tensor in rank3:
        counts[tensor.shape[0]] += 1
    for tensor in rank2:
        counts[tensor.shape[0]] += 1
    seq_len = max(counts, key=lambda dim: counts[dim])
    seq_rank3 = [t for t in rank3 if t.shape[0] == seq_len]
    num_heads = int(seq_rank3[0].shape[1])
    key_dim = int(seq_rank3[0].shape[2])
    value_dim = max(int(t.shape[2]) for t in seq_rank3)
    return 1, seq_len, num_heads, key_dim, value_dim


def _scan_dims(
    estimation: RegionEstimationContext,
    graph: Graph,
    region: Region,
) -> tuple[int, int, int, int, int]:
    return _resolve_scan_geometry(
        _input_tensors(estimation, graph, region)
    )


def _has_scan_state_in(estimation: RegionEstimationContext) -> bool:
    return estimation.input_tensors.get("input5") is not None


def _is_decode_graph(
    estimation: RegionEstimationContext,
    context: InvocationContext,
    module_path: tuple[str, ...],
    *,
    region: Region,
    graph: Graph,
) -> bool:
    try:
        _batch, seq_len, _, _, _ = _scan_dims(estimation, graph, region)
    except ValueError:
        return False
    if seq_len != 1:
        return False
    if _has_scan_state_in(estimation):
        return True
    if len(region.boundary_inputs) >= 6:
        return True
    scenario = resolve_recurrent_scenario(context, module_path)
    return scenario is not None and scenario.kind == "gated_delta"


@dataclass(frozen=True, slots=True)
class FusedGatedDeltaScanRegionImplementation:
    """Lower a ``GatedDeltaScan`` module graph as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: GatedDeltaScanRecipe = DEFAULT_GATED_DELTA_SCAN_RECIPE
    hardware_gate: HardwareGate = "any"
    decode_only: bool = False

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "GatedDeltaScan":
            return "not a GatedDeltaScan module invocation"
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

        batch, seq_len, num_heads, key_dim, value_dim = _scan_dims(
            estimation, graph, region
        )
        num_tokens = batch * seq_len
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
            workspace_id = f"region:{region.id}:chunk_scan_workspace"
            workspace_tensor = Tensor(
                shape=(0,),
                semantic_type="gated_delta_scan_workspace",
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
                shape=with_batch_prefix(
                    batch, (seq_len, num_heads, key_dim, value_dim)
                ),
                semantic_type="gated_delta_state_checkpoint",
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

        if len(region.boundary_outputs) > 1:
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
                context, region.anchor.module_path, batch=batch
            )
            if scenario is not None and scenario.kind == "gated_delta":
                state_event, allocate, persist, aux_id = recurrent_state_port_events(
                    region_id=region.id,
                    scenario=scenario,
                )
                port_tensor = Tensor(
                    shape=with_batch_prefix(
                        scenario.batch,
                        (
                            scenario.num_heads,
                            scenario.head_dim,
                            scenario.state_dim,
                        ),
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
            seq_len=num_tokens,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
        )
        backward_flops = self.recipe.backward_flops(
            seq_len=num_tokens,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
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
        kind="region/gated_delta_scan",
        priority=priority,
        capabilities=frozenset({"fused", "recurrent_state_port"}),
        provenance_rule=GATED_DELTA_SCAN_PROVENANCE,
        pattern_rule=None,
    )


FUSED_GATED_DELTA_SCAN_REFERENCE = FusedGatedDeltaScanRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_delta_scan/reference",
        priority=5,
    ),
    recipe=DEFAULT_GATED_DELTA_SCAN_RECIPE,
    hardware_gate="any",
)

FUSED_GATED_DELTA_SCAN_FLA_CHUNK = FusedGatedDeltaScanRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_delta_scan",
        priority=8,
    ),
    recipe=DEFAULT_GATED_DELTA_SCAN_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_GATED_DELTA_SCAN_DECODE = FusedGatedDeltaScanRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_delta_scan/decode",
        priority=9,
    ),
    recipe=DECODE_GATED_DELTA_SCAN_RECIPE,
    hardware_gate="cuda_only",
    decode_only=True,
)

GATED_DELTA_SCAN_REGIONS: tuple[
    FusedGatedDeltaScanRegionImplementation, ...
] = (
    FUSED_GATED_DELTA_SCAN_REFERENCE,
    FUSED_GATED_DELTA_SCAN_FLA_CHUNK,
    FUSED_GATED_DELTA_SCAN_DECODE,
)
