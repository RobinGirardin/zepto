"""Fused attention-sink softmax region lowering: reference and megatron variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind
from zepto.semantic.operations.helpers import includes_backward

from ....context import InvocationContext
from ....helpers import RegionEstimationContext, ensure_lowered_edge
from ....recipes.softmax_one import (
    DEFAULT_SOFTMAX_ONE_RECIPE,
    MEGATRON_SOFTMAX_ONE_RECIPE,
    SoftmaxOneRecipe,
)
from ....region import Region
from ....registry import RegionImplementationDescriptor
from .rules import SOFTMAX_ONE_OP_SEQUENCES, SOFTMAX_ONE_PROVENANCE

HardwareGate = Literal["any", "cuda_only"]


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "softmax-one fusion requires requested_capabilities={'fused'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "cuda variant requires hardware='cuda'"
    return None


def _sink_softmax_dims(tensor: Tensor) -> tuple[int, int, int]:
    """Return ``(batch, num_heads, seq_len)`` from rank-3 or rank-4 attention scores."""
    shape = tensor.shape
    if len(shape) == 3:
        heads, seq_len, _ = shape
        return 1, int(heads), int(seq_len)
    if len(shape) == 4:
        batch, heads, seq_len, _ = shape
        return int(batch), int(heads), int(seq_len)
    raise ValueError(
        f"softmax-one region expects rank-3 (h, S, S) or rank-4 (B, h, S, S), "
        f"got {shape}"
    )


@dataclass(frozen=True, slots=True)
class FusedSoftmaxOneRegionImplementation:
    """Lower ``AttentionSoftmaxWithSink`` as one fused boundary-A region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: SoftmaxOneRecipe = DEFAULT_SOFTMAX_ONE_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "AttentionSoftmaxWithSink":
            return "not an AttentionSoftmaxWithSink module invocation"
        if len(region.operation_ids) != 1:
            return "softmax-one fusion expects a single fused op"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        if families not in SOFTMAX_ONE_OP_SEQUENCES:
            return "unexpected operation sequence"
        if reason := _requires_fused(context):
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

        input_tensor = estimation.input_tensors.get("scores")
        output_tensor = estimation.output_tensors.get("output")
        if input_tensor is None and region.boundary_inputs:
            input_tensor = graph.edge(region.boundary_inputs[0]).tensor
        if output_tensor is None and region.boundary_outputs:
            output_tensor = graph.edge(region.boundary_outputs[0]).tensor
        if input_tensor is None or output_tensor is None:
            raise ValueError("softmax-one region requires boundary tensors")

        batch, num_heads, seq_len = _sink_softmax_dims(output_tensor)
        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]

        events: list[ResourceEvent] = [
            ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
        ]
        if self.recipe.save_P and input_tensor.requires_grad:
            events.append(ResourceEvent(ResourceEventKind.SAVE, lowered_out))

        forward_flops = self.recipe.forward_flops(
            num_heads=num_heads, seq_len=seq_len
        )
        backward_flops = self.recipe.backward_flops(
            num_heads=num_heads,
            seq_len=seq_len,
            requires_grad=input_tensor.requires_grad,
        )
        forward_flops *= batch
        backward_flops *= batch

        if (
            includes_backward(context.phase)
            and self.recipe.save_P
            and input_tensor.requires_grad
        ):
            events.append(
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    lowered_out,
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
            auxiliary_edges=(),
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
        kind="region/softmax-one",
        priority=priority,
        capabilities=frozenset({"fused", "attention_sink"}),
        provenance_rule=SOFTMAX_ONE_PROVENANCE,
        pattern_rule=None,
    )


FUSED_SOFTMAX_ONE_REFERENCE = FusedSoftmaxOneRegionImplementation(
    descriptor=_descriptor(impl_id="region/softmax-one/reference", priority=5),
    recipe=DEFAULT_SOFTMAX_ONE_RECIPE,
    hardware_gate="any",
)

FUSED_SOFTMAX_ONE_MEGATRON = FusedSoftmaxOneRegionImplementation(
    descriptor=_descriptor(impl_id="region/softmax-one/megatron", priority=9),
    recipe=MEGATRON_SOFTMAX_ONE_RECIPE,
    hardware_gate="any",
)

SOFTMAX_ONE_REGIONS: tuple[FusedSoftmaxOneRegionImplementation, ...] = (
    FUSED_SOFTMAX_ONE_REFERENCE,
    FUSED_SOFTMAX_ONE_MEGATRON,
)
