"""Shared helpers for GQA and GQA-sink region lowering."""

from __future__ import annotations

from zepto.compose.values import Tensor
from zepto.graph.graph import Graph
from zepto.graph.ids import NodeId
from zepto.semantic.operations import (
    MaterializedCausalMask,
    MaterializedSlidingWindowCausalMask,
)

from ....region import Region

_MASK_FAMILIES = frozenset(
    {
        "materialized_causal_mask",
        "materialized_sliding_window_causal_mask",
    }
)


def attention_dims(output_tensor: Tensor) -> tuple[int, int, int, int]:
    """Return ``(batch, num_heads, seq_len, head_dim)``.

    ``batch=1`` for rank-3 ``(h, S, d_h)`` legacy tensors.
    """
    shape = output_tensor.shape
    if len(shape) == 3:
        h, s, dh = shape
        return 1, int(h), int(s), int(dh)
    if len(shape) == 4:
        b, h, s, dh = shape
        return int(b), int(h), int(s), int(dh)
    raise ValueError(
        f"gqa flash region expects rank-3 (h, S, d_h) or rank-4 "
        f"(B, h, S, d_h), got {shape}"
    )


def resolve_window_size(region: Region, graph: Graph) -> int | None:
    """Resolve sliding-window width from the mask edge feeding the fused ``add`` op."""
    if len(region.operation_ids) < 5:
        return None
    add_op_id = region.operation_ids[4]
    add_node = graph.node(add_op_id)
    if add_node.operation_family != "add":
        return None

    for edge_id in add_node.input_edges:
        edge = graph.edge(edge_id)
        if edge.producer is None:
            continue
        producer = graph.node(edge.producer.node_id)
        if producer.operation_family not in _MASK_FAMILIES:
            continue
        declaration = producer.declaration
        if isinstance(declaration, MaterializedSlidingWindowCausalMask):
            return declaration.window_size
        if isinstance(declaration, MaterializedCausalMask):
            return None
    return None


def mask_producer_for_region(region: Region, graph: Graph) -> NodeId | None:
    """Return the mask op node id consumed by the fused region's ``add`` op, if any."""
    if len(region.operation_ids) < 5:
        return None
    add_op_id = region.operation_ids[4]
    add_node = graph.node(add_op_id)
    for edge_id in add_node.input_edges:
        edge = graph.edge(edge_id)
        if edge.producer is None:
            continue
        producer_id = edge.producer.node_id
        producer = graph.node(producer_id)
        if producer.operation_family in _MASK_FAMILIES:
            if len(edge.consumers) == 1:
                return producer_id
    return None
