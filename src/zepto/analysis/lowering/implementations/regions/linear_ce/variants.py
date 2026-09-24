"""Fused linear + CE region lowering: reference, liger, hub-trl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zepto.analysis.lowered import LoweredNode
from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.semantic.metadata import DType, TensorRole
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind
from zepto.semantic.operations.helpers import includes_backward

from ....context import InvocationContext
from ....helpers import (
    RegionEstimationContext,
    ensure_lowered_edge,
    register_auxiliary_edge,
    unpack_hidden,
)
from ....recipes.linear_ce import DEFAULT_LINEAR_CE_RECIPE, LinearCERecipe
from ....region import Region
from ....registry import RegionImplementationDescriptor
from ....role import RoleContext
from .rules import LINEAR_CE_OP_SEQUENCES, LINEAR_CE_PATTERN, LINEAR_CE_PROVENANCE

HardwareGate = Literal["any", "cuda_only", "exclude_xpu_mps"]


def _requires_fused(context: InvocationContext) -> str | None:
    if "fused" not in context.requested_capabilities:
        return "linear_ce fusion requires requested_capabilities={'fused'}"
    return None


def _check_hardware_gate(gate: HardwareGate, context: InvocationContext) -> str | None:
    if gate == "cuda_only" and context.hardware != "cuda":
        return "hub-trl requires hardware='cuda'"
    if gate == "exclude_xpu_mps" and context.hardware in ("xpu", "mps"):
        return "liger is not routed on xpu/mps"
    return None


def _elem_bytes(tensor: Tensor) -> int:
    if tensor.dtype is None:
        return 2
    return tensor.dtype.itemsize


@dataclass(frozen=True, slots=True)
class FusedLinearCERegionImplementation:
    """Lower decomposed LM-head linear + CE as one fused region leaf."""

    descriptor: RegionImplementationDescriptor
    recipe: LinearCERecipe = DEFAULT_LINEAR_CE_RECIPE
    hardware_gate: HardwareGate = "any"

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        if region.anchor.component_type != "FusedLinearCrossEntropy":
            return "not a FusedLinearCrossEntropy module invocation"
        families = tuple(
            graph.node(op_id).operation_family for op_id in region.operation_ids
        )
        if families not in LINEAR_CE_OP_SEQUENCES:
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

        hidden_tensor = graph.edge(region.boundary_inputs[0]).tensor
        output_tensor = graph.edge(region.boundary_outputs[0]).tensor
        _batch, _seq_len, d = unpack_hidden(hidden_tensor)
        s = _batch * _seq_len
        v = self.weight_vocab_size(region, graph)
        elem = _elem_bytes(hidden_tensor)

        output_id = region.boundary_outputs[0]
        hidden_id = region.boundary_inputs[0]
        lowered_out = edge_map[output_id]
        lowered_hidden = edge_map[hidden_id]

        events: list[ResourceEvent] = []
        auxiliary_edges: list[str] = []

        if not self.recipe.materialize_full_logits:
            chunk_s = self.recipe.chunk_size(s, d, v)
            logits_chunk = f"region:{region.id}:logits_chunk"
            chunk_tensor = Tensor(
                shape=(chunk_s, v),
                semantic_type="logits_chunk",
                dtype=hidden_tensor.dtype or DType.BF16,
                requires_grad=False,
                persistent=False,
            )
            register_auxiliary_edge(
                logits_chunk,
                chunk_tensor,
                role_ctx=RoleContext(explicit_role=TensorRole.AUXILIARY),
                context=context,
                storage_id=logits_chunk,
                lowered_edges=lowered_edges,
            )
            events.append(
                ResourceEvent(ResourceEventKind.ALLOCATE, logits_chunk)
            )
            auxiliary_edges.append(logits_chunk)

        events.append(ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out))

        if self.recipe.save_hidden and hidden_tensor.requires_grad:
            events.append(ResourceEvent(ResourceEventKind.SAVE, lowered_hidden))

        forward_flops = self.recipe.forward_flops(s, d, v)
        backward_flops = self.recipe.backward_flops(
            s, d, v, requires_grad=hidden_tensor.requires_grad
        )

        if includes_backward(context.phase) and self.recipe.save_hidden:
            events.append(
                ResourceEvent(
                    ResourceEventKind.RELEASE,
                    lowered_hidden,
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

    @staticmethod
    def weight_vocab_size(region: Region, graph: Graph) -> int:
        """Read vocab size from the linear_matmul weight parameter."""
        first_op = graph.node(region.operation_ids[0])
        for param_id in first_op.parameter_ids:
            param = graph.parameter(param_id)
            if param.shape and len(param.shape) == 2:
                return param.shape[1]
        raise ValueError("linear_ce region requires weight parameter on linear_matmul")


def _descriptor(*, impl_id: str, priority: int) -> RegionImplementationDescriptor:
    return RegionImplementationDescriptor(
        id=impl_id,
        kind="region/linear_ce",
        priority=priority,
        capabilities=frozenset({"fused"}),
        provenance_rule=LINEAR_CE_PROVENANCE,
        pattern_rule=LINEAR_CE_PATTERN,
    )


FUSED_LINEAR_CE_REFERENCE = FusedLinearCERegionImplementation(
    descriptor=_descriptor(impl_id="region/linear_ce/reference", priority=5),
    recipe=DEFAULT_LINEAR_CE_RECIPE,
    hardware_gate="any",
)

FUSED_LINEAR_CE_HUB_TRL = FusedLinearCERegionImplementation(
    descriptor=_descriptor(impl_id="region/linear_ce/hub-trl", priority=7),
    recipe=DEFAULT_LINEAR_CE_RECIPE,
    hardware_gate="cuda_only",
)

FUSED_LINEAR_CE_LIGER = FusedLinearCERegionImplementation(
    descriptor=_descriptor(impl_id="region/linear_ce/liger", priority=8),
    recipe=DEFAULT_LINEAR_CE_RECIPE,
    hardware_gate="exclude_xpu_mps",
)

LINEAR_CE_REGIONS: tuple[FusedLinearCERegionImplementation, ...] = (
    FUSED_LINEAR_CE_REFERENCE,
    FUSED_LINEAR_CE_HUB_TRL,
    FUSED_LINEAR_CE_LIGER,
)
