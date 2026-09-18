"""Fused Mamba2Mixer block region: composed Tier-A, hub mega, and decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.modules.mixers.mixer_config import Mamba2MixerConfig
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
from ....recipes.gated_grouped_rms_norm import DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE
from ....recipes.mamba2_mixer import (
    DECODE_MAMBA2_MIXER_RECIPE,
    DEFAULT_MAMBA2_MIXER_RECIPE,
    HUB_MEGA_MAMBA2_MIXER_RECIPE,
    Mamba2MixerRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import MAMBA2_MIXER_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "mamba2 mixer fusion requires requested_capabilities={'fused'}"
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
) -> Mamba2MixerConfig | str:
    if len(hidden.shape) != 2:
        return "Mamba2Mixer expects rank-2 hidden_states"
    hidden_size = hidden.shape[1]
    in_proj_size: int | None = None
    intermediate_size: int | None = None
    conv_channels: int | None = None
    conv_kernel_size = 4
    conv_bias = False
    scan_head_counts: list[int] = []

    in_proj_candidates: list[int] = []
    for parameter_id in region.parameter_ids:
        tensor = graph.parameter(parameter_id).as_tensor()
        if tensor.semantic_type == "weight" and len(tensor.shape) == 2:
            if tensor.shape[1] == hidden_size:
                intermediate_size = tensor.shape[0]
            if tensor.shape[0] == hidden_size:
                in_proj_candidates.append(tensor.shape[1])
    if in_proj_candidates:
        in_proj_size = max(in_proj_candidates)
        if tensor.semantic_type in ("weight", "bias") and len(tensor.shape) == 1:
            dim = tensor.shape[0]
            if dim < hidden_size and dim <= (intermediate_size or hidden_size):
                scan_head_counts.append(dim)

    if in_proj_size is None or intermediate_size is None:
        return "could not resolve Mamba2Mixer projection geometry"

    conv_lags = [
        graph.parameter(parameter_id).as_tensor().shape[0]
        for parameter_id in region.parameter_ids
        if graph.parameter(parameter_id).as_tensor().semantic_type == "weight"
        and len(graph.parameter(parameter_id).as_tensor().shape) == 1
        and graph.parameter(parameter_id).as_tensor().shape[0] > intermediate_size
    ]
    if conv_lags:
        conv_channels = conv_lags[0]
        conv_kernel_size = len(conv_lags)
    else:
        conv_channels = in_proj_size - intermediate_size - min(scan_head_counts or [1])

    for parameter_id in region.parameter_ids:
        tensor = graph.parameter(parameter_id).as_tensor()
        if (
            tensor.semantic_type == "bias"
            and len(tensor.shape) == 1
            and tensor.shape[0] == conv_channels
        ):
            conv_bias = True

    num_heads = in_proj_size - intermediate_size - conv_channels
    if num_heads <= 0 or intermediate_size % num_heads != 0:
        return "invalid Mamba2Mixer head geometry"
    head_dim = intermediate_size // num_heads

    bc_width = conv_channels - intermediate_size
    if bc_width <= 0 or bc_width % 2 != 0:
        return "invalid conv channel geometry"
    gn_product = bc_width // 2
    num_groups: int | None = None
    state_size: int | None = None
    for groups in range(num_heads, 0, -1):
        if num_heads % groups != 0:
            continue
        if gn_product % groups != 0:
            continue
        n = gn_product // groups
        if n > 0:
            num_groups = groups
            state_size = n
            break
    if num_groups is None or state_size is None:
        return "could not resolve num_groups and state_size"

    return Mamba2MixerConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        state_size=state_size,
        num_groups=num_groups,
        conv_kernel_size=conv_kernel_size,
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
    if scenario is not None and scenario.kind == "mamba2":
        return True
    conv = resolve_conv_scenario(context, region.anchor.module_path)
    return conv is not None


@dataclass(frozen=True, slots=True)
class FusedMamba2MixerRegionImplementation:
    """Lower a full ``Mamba2Mixer`` graph as one fused block envelope."""

    descriptor: RegionImplementationDescriptor
    recipe: Mamba2MixerRecipe = DEFAULT_MAMBA2_MIXER_RECIPE
    hardware_gate: HardwareGate = "any"
    decode_only: bool = False

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "Mamba2Mixer":
            return "not a Mamba2Mixer module invocation"
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
            raise ValueError("Mamba2Mixer region requires hidden_states")
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

        c_c = cfg.conv_channels
        h, p, n, g = cfg.num_heads, cfg.head_dim, cfg.state_size, cfg.num_groups

        if (
            requires_grad
            and self.recipe.save_conv_pre_activation
            and self.recipe.conv_activation == "silu"
        ):
            aux_id = f"region:{region.id}:conv_pre_activation"
            pre_act = Tensor(
                shape=(seq_len, c_c),
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
                shape=(seq_len, h, p, n),
                semantic_type="mamba2_state_checkpoint",
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

        if requires_grad and self.recipe.save_group_rstd:
            aux_id = f"region:{region.id}:group_rstd"
            rstd_shape = DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE.group_rstd_shape(
                (seq_len, cfg.intermediate_size), g
            )
            rstd = Tensor(
                shape=rstd_shape,
                semantic_type="group_rstd",
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
            if scan_scenario is not None and scan_scenario.kind == "mamba2":
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
        kind="region/mamba2_mixer",
        priority=priority,
        capabilities=frozenset(
            {"fused", "conv_state_port", "recurrent_state_port"}
        ),
        provenance_rule=MAMBA2_MIXER_PROVENANCE,
        pattern_rule=None,
    )


FUSED_MAMBA2_MIXER_COMPOSED_TIER_A = FusedMamba2MixerRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/mamba2_mixer/composed_tier_a",
        priority=11,
    ),
    recipe=DEFAULT_MAMBA2_MIXER_RECIPE,
    hardware_gate="any",
)

FUSED_MAMBA2_MIXER_HUB_MEGA = FusedMamba2MixerRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/mamba2_mixer/hub_mega",
        priority=12,
    ),
    recipe=HUB_MEGA_MAMBA2_MIXER_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_MAMBA2_MIXER_DECODE = FusedMamba2MixerRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/mamba2_mixer/decode",
        priority=11,
    ),
    recipe=DECODE_MAMBA2_MIXER_RECIPE,
    hardware_gate="any",
    decode_only=True,
)

MAMBA2_MIXER_REGIONS: tuple[FusedMamba2MixerRegionImplementation, ...] = (
    FUSED_MAMBA2_MIXER_HUB_MEGA,
    FUSED_MAMBA2_MIXER_COMPOSED_TIER_A,
    FUSED_MAMBA2_MIXER_DECODE,
)
