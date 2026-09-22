"""RoPEApply region lowering."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.lowered import LoweredNode
from zepto.graph.graph import Graph
from zepto.semantic.operations.records import ResourceEvent, ResourceEventKind

from ....context import InvocationContext
from ....helpers import RegionEstimationContext, ensure_lowered_edge
from ....recipes.rope import DEFAULT_ROPE_APPLY_RECIPE, RoPEApplyRecipe
from ....region import Region
from ....registry import RegionImplementationDescriptor
from .rules import ROPE_APPLY_PROVENANCE


def _rotary_dim_from_boundaries(region: Region, graph: Graph) -> int:
    for edge_id in region.boundary_inputs:
        tensor = graph.edge(edge_id).tensor
        if len(tensor.shape) == 2:
            return tensor.shape[1]
    raise ValueError("RoPEApply region missing cos/sin cache inputs")


def _headed_outputs(
    region: Region, graph: Graph
) -> list[tuple[int, int, int, int, bool]]:
    """Per rotated output: (batch, num_heads, seq_len, rotary_dim, requires_grad).

    Rank-3 ``(h, S, d)`` is the unbatched alias (implicit ``batch=1``). Rank-4
    ``(B, h, S, d)`` keeps the leading batch so apply FLOPs scale with ``B``.
    """
    rotary_dim = _rotary_dim_from_boundaries(region, graph)
    headed: list[tuple[int, int, int, int, bool]] = []
    for edge_id in region.boundary_outputs:
        tensor = graph.edge(edge_id).tensor
        shape = tensor.shape
        if len(shape) == 3:
            num_heads, seq_len, _ = shape
            batch = 1
        elif len(shape) == 4:
            batch, num_heads, seq_len, _ = shape
        else:
            continue
        headed.append(
            (int(batch), int(num_heads), int(seq_len), rotary_dim, tensor.requires_grad)
        )
    if not headed:
        raise ValueError("RoPEApply region has no headed boundary outputs")
    return headed


@dataclass(frozen=True, slots=True)
class RoPEApplyRegionImplementation:
    """Replace decomposed rotate-half/mul/add with §8 apply FLOPs."""

    descriptor: RegionImplementationDescriptor
    recipe: RoPEApplyRecipe = DEFAULT_ROPE_APPLY_RECIPE

    def compatible(
        self,
        region: Region,
        graph: Graph,
        context: InvocationContext,
    ) -> str | None:
        del context
        if region.anchor.component_type != "RoPEApply":
            return "not a RoPEApply module invocation"
        if len(region.boundary_inputs) < 3:
            return "RoPEApply region expects value, cos, sin inputs"
        try:
            _headed_outputs(region, graph)
        except ValueError as exc:
            return str(exc)
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
    ) -> LoweredNode:
        for edge_id in region.boundary_inputs:
            ensure_lowered_edge(
                edge_id, graph, context, lowered_edges, edge_map
            )
        for edge_id in region.boundary_outputs:
            ensure_lowered_edge(
                edge_id, graph, context, lowered_edges, edge_map
            )

        headed = _headed_outputs(region, graph)
        forward_flops = 0
        backward_flops = 0
        for batch, num_heads, seq_len, rotary_dim, requires_grad in headed:
            fwd = self.recipe.forward_flops(
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=rotary_dim,
            )
            bwd = self.recipe.backward_flops(
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=rotary_dim,
                requires_grad=requires_grad,
            )
            forward_flops += batch * fwd
            backward_flops += batch * bwd

        output_id = region.boundary_outputs[0]
        lowered_out = edge_map[output_id]
        output_edges = tuple(edge_map[eid] for eid in region.boundary_outputs)

        return LoweredNode(
            id=f"region:{region.id}",
            node_id=region.operation_ids[0],
            node_ids=region.operation_ids,
            implementation=self.descriptor.id,
            input_edges=tuple(
                edge_map[eid] for eid in region.boundary_inputs
            ),
            output_edges=output_edges,
            auxiliary_edges=(),
            resource_events=(
                ResourceEvent(ResourceEventKind.ALLOCATE, lowered_out),
            ),
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            module_path=region.anchor.module_path,
            component_type=region.anchor.component_type,
            region_id=region.id,
        )


def _descriptor(*, impl_id: str, priority: int) -> RegionImplementationDescriptor:
    return RegionImplementationDescriptor(
        id=impl_id,
        kind="region/rope_apply",
        priority=priority,
        provenance_rule=ROPE_APPLY_PROVENANCE,
    )


ROPE_APPLY_REFERENCE = RoPEApplyRegionImplementation(
    descriptor=_descriptor(impl_id="region/rope_apply/reference", priority=5),
)

ROPE_APPLY_REGIONS: tuple[RoPEApplyRegionImplementation, ...] = (
    ROPE_APPLY_REFERENCE,
)
