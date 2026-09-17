"""Fused depthwise causal conv1d region lowering: decomposed, cuda, and hub variants."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    conv_state_port_events,
    ensure_lowered_edge,
    register_auxiliary_edge,
    resolve_conv_scenario,
)
from ....recipes.depthwise_causal_conv1d import (
    DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE,
    DepthwiseCausalConv1dRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import DEPTHWISE_CAUSAL_CONV1D_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]

_ALLOWED_OP_FAMILIES = frozenset(
    {
        "concat",
        "split",
        "reshape",
        "transpose",
        "parameter_scale",
        "parameter_bias",
        "add",
        "sigmoid",
        "multiply",
    }
)

_CUDA_KERNEL_WIDTHS = frozenset({2, 3, 4})


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "depthwise causal conv1d fusion requires requested_capabilities={'fused'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _recipe_for_region(
    region: Region,
    graph: Graph,
    base: DepthwiseCausalConv1dRecipe,
) -> DepthwiseCausalConv1dRecipe:
    weight_count = 0
    use_bias = False
    for parameter_id in region.parameter_ids:
        param = graph.parameter(parameter_id)
        tensor = param.as_tensor()
        if tensor.semantic_type == "weight":
            weight_count += 1
        elif tensor.semantic_type == "bias":
            use_bias = True
    kernel_size = weight_count if weight_count > 0 else base.kernel_size
    families = {
        graph.node(op_id).operation_family for op_id in region.operation_ids
    }
    activation: Literal["none", "silu"] = (
        "silu" if "sigmoid" in families and "multiply" in families else "none"
    )
    return replace(
        base,
        kernel_size=kernel_size,
        use_bias=use_bias,
        activation=activation,
    )


def _is_decode(
    estimation: RegionEstimationContext,
    value_tensor: Tensor,
) -> bool:
    if estimation.input_tensors.get("input1") is not None:
        return True
    return value_tensor.shape[0] == 1 and len(value_tensor.shape) == 2


@dataclass(frozen=True, slots=True)
class FusedDepthwiseCausalConv1dRegionImplementation:
    """Lower a ``DepthwiseCausalConv1d`` module graph as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: DepthwiseCausalConv1dRecipe = DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE
    hardware_gate: HardwareGate = "any"
    reject_unsupported_cuda_width: bool = False

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "DepthwiseCausalConv1d":
            return "not a DepthwiseCausalConv1d module invocation"
        if reason := _requires_fused(context):
            return reason
        if reason := _check_hardware_gate(self.hardware_gate, context):
            return reason
        for op_id in region.operation_ids:
            family = graph.node(op_id).operation_family
            if family not in _ALLOWED_OP_FAMILIES:
                return f"unexpected operation family {family!r}"
        tuned = _recipe_for_region(region, graph, self.recipe)
        if (
            self.reject_unsupported_cuda_width
            and tuned.kernel_size not in _CUDA_KERNEL_WIDTHS
        ):
            return f"cuda causal-conv1d supports K in {sorted(_CUDA_KERNEL_WIDTHS)}"
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

        value_tensor = estimation.input_tensors.get("input")
        output_tensor = estimation.output_tensors.get("output")
        if value_tensor is None and region.boundary_inputs:
            value_tensor = graph.edge(region.boundary_inputs[0]).tensor
        if output_tensor is None and region.boundary_outputs:
            output_tensor = graph.edge(region.boundary_outputs[0]).tensor
        if value_tensor is None or output_tensor is None:
            raise ValueError(
                "depthwise causal conv1d region requires value and output tensors"
            )

        recipe = _recipe_for_region(region, graph, self.recipe)
        n = numel(output_tensor)
        value_requires_grad = any(
            graph.edge(edge_id).tensor.requires_grad
            for edge_id in region.boundary_inputs
        ) or output_tensor.requires_grad
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        auxiliary_edges: list[str] = []

        decode = _is_decode(estimation, value_tensor)

        if decode:
            scenario = resolve_conv_scenario(context, region.anchor.module_path)
            if scenario is not None:
                state_event, allocate, persist, aux_id = conv_state_port_events(
                    region_id=region.id,
                    scenario=scenario,
                )
                if state_port_collector is not None:
                    state_port_collector.append(state_event)
                events.extend([allocate, persist])
                auxiliary_edges.append(aux_id)
        elif len(region.boundary_outputs) > 1:
            state_out_id = region.boundary_outputs[1]
            ensure_lowered_edge(
                state_out_id, graph, context, lowered_edges, edge_map
            )
            lowered_state = edge_map[state_out_id]
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, lowered_state),
            )

        if (
            recipe.save_pre_activation
            and recipe.activation == "silu"
            and value_requires_grad
        ):
            saved_pre = f"region:{region.id}:pre_activation"
            pre_tensor = Tensor(
                shape=output_tensor.shape,
                semantic_type="conv_pre_activation",
                dtype=output_tensor.dtype or DType.BF16,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                saved_pre,
                pre_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=saved_pre,
                lowered_edges=lowered_edges,
            )
            events.append(ResourceEvent(ResourceEventKind.ALLOCATE, saved_pre))
            events.append(ResourceEvent(ResourceEventKind.SAVE, saved_pre))
            auxiliary_edges.append(saved_pre)

        forward_flops = recipe.forward_flops(n)
        backward_flops = recipe.backward_flops(
            n, requires_grad=value_requires_grad
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
        kind="region/depthwise_causal_conv1d",
        priority=priority,
        capabilities=frozenset({"fused"}),
        provenance_rule=DEPTHWISE_CAUSAL_CONV1D_PROVENANCE,
        pattern_rule=None,
    )


FUSED_DEPTHWISE_CAUSAL_CONV1D_DECOMPOSED = FusedDepthwiseCausalConv1dRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/depthwise_causal_conv1d/decomposed",
        priority=5,
    ),
    recipe=DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE,
    hardware_gate="any",
)

FUSED_DEPTHWISE_CAUSAL_CONV1D_HUB = FusedDepthwiseCausalConv1dRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/depthwise_causal_conv1d/hub",
        priority=7,
    ),
    recipe=DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_DEPTHWISE_CAUSAL_CONV1D_CUDA = FusedDepthwiseCausalConv1dRegionImplementation(
    descriptor=_descriptor(
        impl_id="region/depthwise_causal_conv1d/cuda",
        priority=8,
    ),
    recipe=DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE,
    hardware_gate="cuda_only",
    reject_unsupported_cuda_width=True,
)

DEPTHWISE_CAUSAL_CONV1D_REGIONS: tuple[
    FusedDepthwiseCausalConv1dRegionImplementation, ...
] = (
    FUSED_DEPTHWISE_CAUSAL_CONV1D_DECOMPOSED,
    FUSED_DEPTHWISE_CAUSAL_CONV1D_HUB,
    FUSED_DEPTHWISE_CAUSAL_CONV1D_CUDA,
)
