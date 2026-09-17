"""Fused GatedDeltaNet block region: composed Tier-A, FLA parity, and decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.modules.mixer_config import GatedDeltaNetConfig
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    conv_state_port_events,
    ensure_lowered_edge,
    recurrent_state_port_events,
    register_auxiliary_edge,
    resolve_conv_scenario,
    resolve_recurrent_scenario,
)
from ....recipes.gated_delta_net import (
    DECODE_GATED_DELTA_NET_RECIPE,
    DEFAULT_GATED_DELTA_NET_RECIPE,
    FLA_LAYER_PARITY_GATED_DELTA_NET_RECIPE,
    GatedDeltaNetRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import GATED_DELTA_NET_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "gated delta net fusion requires requested_capabilities={'fused'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _hidden_tensor(
    estimation: RegionEstimationContext,
    graph: Graph,
    region: Region,
) -> Tensor | None:
    if estimation.input_tensors.get("input") is not None:
        return estimation.input_tensors["input"]
    if region.boundary_inputs:
        return graph.edge(region.boundary_inputs[0]).tensor
    return None


def _resolve_config(
    region: Region,
    graph: Graph,
    hidden: Tensor,
) -> GatedDeltaNetConfig | str:
    if len(hidden.shape) != 2:
        return "GatedDeltaNet expects rank-2 hidden_states"
    hidden_size = hidden.shape[1]
    h_v: int | None = None
    head_dim: int | None = None
    kernel_size = 4
    conv_bias = False
    conv_channels: int | None = None
    one_d_weights: list[int] = []
    for parameter_id in region.parameter_ids:
        tensor = graph.parameter(parameter_id).as_tensor()
        if tensor.semantic_type == "bias" and len(tensor.shape) == 1:
            if h_v is None or tensor.shape[0] <= h_v:
                h_v = tensor.shape[0]
            if tensor.shape[0] == h_v:
                conv_bias = conv_bias or tensor.shape[0] != h_v
        if tensor.semantic_type == "weight" and len(tensor.shape) == 1:
            one_d_weights.append(tensor.shape[0])
        if (
            tensor.semantic_type == "weight"
            and len(tensor.shape) == 2
            and tensor.shape[1] in (2, 3, 4)
            and tensor.shape[0] > hidden_size
        ):
            conv_channels = tensor.shape[0]
            kernel_size = tensor.shape[1]
    dt_bias_len = [
        graph.parameter(parameter_id).as_tensor().shape[0]
        for parameter_id in region.parameter_ids
        if graph.parameter(parameter_id).as_tensor().semantic_type == "bias"
        and len(graph.parameter(parameter_id).as_tensor().shape) == 1
    ]
    if dt_bias_len:
        h_v = dt_bias_len[0]
    if h_v is None:
        return "could not resolve num_v_heads"
    small_weights = sorted({dim for dim in one_d_weights if dim < hidden_size})
    if not small_weights:
        return "could not resolve head_dim"
    head_dim = min(dim for dim in small_weights if dim != h_v)
    if conv_channels is None:
        channel_candidates = [
            dim for dim in one_d_weights if dim not in (h_v, head_dim)
        ]
        if not channel_candidates:
            return "could not resolve conv channel count"
        conv_channels = max(channel_candidates)
    for parameter_id in region.parameter_ids:
        tensor = graph.parameter(parameter_id).as_tensor()
        if tensor.semantic_type == "bias" and len(tensor.shape) == 1:
            if tensor.shape[0] == conv_channels:
                conv_bias = True
    d_v_proj = h_v * head_dim
    if (conv_channels - d_v_proj) % 2 != 0:
        return "invalid conv channel geometry"
    d_q = (conv_channels - d_v_proj) // 2
    if d_q % head_dim != 0:
        return "invalid qk projection geometry"
    h_k = d_q // head_dim
    if h_v % h_k != 0:
        return "num_v_heads must be divisible by num_qk_heads"
    return GatedDeltaNetConfig(
        hidden_size=hidden_size,
        num_qk_heads=h_k,
        num_v_heads=h_v,
        head_dim=head_dim,
        conv_kernel_size=kernel_size,
        conv_bias=conv_bias,
    )


def _has_state_inputs(estimation: RegionEstimationContext, region: Region) -> bool:
    if estimation.input_tensors.get("input1") is not None:
        return True
    if estimation.input_tensors.get("input2") is not None:
        return True
    return len(region.boundary_inputs) >= 3


def _is_decode_graph(
    estimation: RegionEstimationContext,
    context: InvocationContext,
    region: Region,
    hidden: Tensor,
) -> bool:
    if hidden.shape[0] != 1:
        return False
    if _has_state_inputs(estimation, region):
        return True
    if context.phase == "decode":
        return True
    scenario = resolve_recurrent_scenario(context, region.anchor.module_path)
    if scenario is not None and scenario.kind == "gated_delta":
        return True
    conv = resolve_conv_scenario(context, region.anchor.module_path)
    return conv is not None


@dataclass(frozen=True, slots=True)
class FusedGatedDeltaNetRegionImplementation:
    """Lower a full ``GatedDeltaNet`` graph as one fused block envelope."""

    descriptor: RegionImplementationDescriptor
    recipe: GatedDeltaNetRecipe = DEFAULT_GATED_DELTA_NET_RECIPE
    hardware_gate: HardwareGate = "any"
    decode_only: bool = False

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "GatedDeltaNet":
            return "not a GatedDeltaNet module invocation"
        if reason := _requires_fused(context):
            return reason
        if reason := _check_hardware_gate(self.hardware_gate, context):
            return reason
        hidden = graph.edge(region.boundary_inputs[0]).tensor if region.boundary_inputs else None
        if hidden is None:
            return "missing hidden_states boundary"
        cfg_or_err = _resolve_config(region, graph, hidden)
        if isinstance(cfg_or_err, str):
            return cfg_or_err
        estimation = RegionEstimationContext(
            region=region,
            phase=context.phase,
            input_tensors={"input": hidden},
            output_tensors={},
            parameter_tensors={},
            precision=context.precision,
            state=context.state,
        )
        decode = _is_decode_graph(estimation, context, region, hidden)
        if self.decode_only:
            if not decode:
                return "decode variant requires S=1 with conv/scan state ports"
        elif decode:
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

        hidden = _hidden_tensor(estimation, graph, region)
        if hidden is None:
            raise ValueError("GatedDeltaNet region requires hidden_states")
        cfg_or_err = _resolve_config(region, graph, hidden)
        if isinstance(cfg_or_err, str):
            raise ValueError(cfg_or_err)
        cfg = cfg_or_err
        seq_len = hidden.shape[0]
        requires_grad = hidden.requires_grad or any(
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
            estimation, context, region, hidden
        )

        g = self.recipe._geometry(cfg, seq_len)
        h_k, h_v, d_k, d_v = g["h_k"], g["h_v"], g["d_k"], g["d_v"]
        channels = g["C"]

        if requires_grad and self.recipe.save_l2_rstd and not self.recipe.use_qk_l2norm_in_kernel:
            for tag in ("rstd_q", "rstd_k"):
                aux_id = f"region:{region.id}:{tag}"
                rstd_tensor = Tensor(
                    shape=(seq_len, h_k, 1),
                    semantic_type="l2_rstd",
                    dtype=DType.FP32,
                    requires_grad=False,
                    persistent=False,
                )
                register_auxiliary_edge(
                    aux_id,
                    rstd_tensor,
                    role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                    context=context,
                    storage_id=aux_id,
                    lowered_edges=lowered_edges,
                )
                events.append(ResourceEvent(ResourceEventKind.SAVE, aux_id))
                auxiliary_edges.append(aux_id)

        if (
            requires_grad
            and self.recipe.save_conv_pre_activation
            and self.recipe.conv_activation == "silu"
        ):
            aux_id = f"region:{region.id}:conv_pre_activation"
            pre_act = Tensor(
                shape=(seq_len, channels),
                semantic_type="conv_pre_activation",
                dtype=DType.BF16,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                aux_id,
                pre_act,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=aux_id,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.SAVE, aux_id))
            auxiliary_edges.append(aux_id)

        if (
            requires_grad
            and self.recipe.save_scan_state_checkpoint
            and not decode
        ):
            aux_id = f"region:{region.id}:scan_state_checkpoint"
            checkpoint = Tensor(
                shape=(seq_len, h_v, d_k, d_v),
                semantic_type="gated_delta_state_checkpoint",
                dtype=DType.FP32,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                aux_id,
                checkpoint,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=aux_id,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, aux_id))
            events.append(ResourceEvent(ResourceEventKind.SAVE, aux_id))
            auxiliary_edges.append(aux_id)

        if (
            requires_grad
            and self.recipe.save_gated_rms_rstd
            and not self.recipe.use_gate_in_kernel
        ):
            aux_id = f"region:{region.id}:gated_rms_rstd"
            rstd = Tensor(
                shape=(seq_len, h_v, 1),
                semantic_type="gated_rms_rstd",
                dtype=DType.FP32,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                aux_id,
                rstd,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=aux_id,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.SAVE, aux_id))
            auxiliary_edges.append(aux_id)

        module_path = region.anchor.module_path
        if len(region.boundary_outputs) > 1:
            conv_state_id = region.boundary_outputs[1]
            ensure_lowered_edge(
                conv_state_id, graph, context, lowered_edges, edge_map
            )
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, edge_map[conv_state_id]),
            )
            events.append(
                ResourceEvent(ResourceEventKind.PERSIST, edge_map[conv_state_id]),
            )
        elif decode:
            conv_scenario = resolve_conv_scenario(context, module_path)
            if conv_scenario is not None:
                _state_event, allocate, persist, aux_id = conv_state_port_events(
                    region_id=region.id,
                    scenario=conv_scenario,
                )
                port_tensor = Tensor(
                    shape=(conv_scenario.channels, conv_scenario.kernel_size),
                    semantic_type="conv_state",
                    dtype=DType.BF16,
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
                    state_port_collector.append(_state_event)
                events.extend([allocate, persist])
                auxiliary_edges.append(aux_id)

        if len(region.boundary_outputs) > 2:
            scan_state_id = region.boundary_outputs[2]
            ensure_lowered_edge(
                scan_state_id, graph, context, lowered_edges, edge_map
            )
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, edge_map[scan_state_id]),
            )
            events.append(
                ResourceEvent(ResourceEventKind.PERSIST, edge_map[scan_state_id]),
            )
        else:
            scan_scenario = resolve_recurrent_scenario(context, module_path)
            if scan_scenario is not None and scan_scenario.kind == "gated_delta":
                state_event, allocate, persist, aux_id = recurrent_state_port_events(
                    region_id=region.id,
                    scenario=scan_scenario,
                )
                port_tensor = Tensor(
                    shape=(
                        scan_scenario.num_heads,
                        scan_scenario.head_dim,
                        scan_scenario.state_dim,
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

        if decode:
            forward_flops = self.recipe.decode_forward_flops(cfg)
            backward_flops = self.recipe.decode_backward_flops(
                cfg, requires_grad=requires_grad
            )
        else:
            forward_flops = self.recipe.forward_flops(cfg, seq_len=seq_len)
            backward_flops = self.recipe.backward_flops(
                cfg, seq_len=seq_len, requires_grad=requires_grad
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
        kind="region/gated_delta_net",
        priority=priority,
        capabilities=frozenset(
            {"fused", "conv_state_port", "recurrent_state_port"}
        ),
        provenance_rule=GATED_DELTA_NET_PROVENANCE,
        pattern_rule=None,
    )


FUSED_GATED_DELTA_NET_COMPOSED_TIER_A = FusedGatedDeltaNetRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_delta_net/composed_tier_a",
        priority=10,
    ),
    recipe=DEFAULT_GATED_DELTA_NET_RECIPE,
    hardware_gate="any",
)

FUSED_GATED_DELTA_NET_FLA_LAYER_PARITY = FusedGatedDeltaNetRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_delta_net/fla_layer_parity",
        priority=9,
    ),
    recipe=FLA_LAYER_PARITY_GATED_DELTA_NET_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_GATED_DELTA_NET_DECODE = FusedGatedDeltaNetRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/gated_delta_net/decode",
        priority=10,
    ),
    recipe=DECODE_GATED_DELTA_NET_RECIPE,
    hardware_gate="any",
    decode_only=True,
)

GATED_DELTA_NET_REGIONS: tuple[FusedGatedDeltaNetRegionImplementation, ...] = (
    FUSED_GATED_DELTA_NET_FLA_LAYER_PARITY,
    FUSED_GATED_DELTA_NET_COMPOSED_TIER_A,
    FUSED_GATED_DELTA_NET_DECODE,
)
