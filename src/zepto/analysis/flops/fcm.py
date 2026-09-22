"""FCM-comparable FLOP totals from the structural graph.

Walks ``Graph.nodes`` in ``node_order`` and keeps Zepto's existing
``forward_flops`` / ``backward_flops`` only for allowlisted families
(``matmul``, ``linear_matmul``, ``conv2d``, ``conv3d``). Fused-region
recipes and optimizer policy are not consulted.
"""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis.flops.policy import is_fcm_operation_family
from zepto.analysis.lowering.context import InvocationContext
from zepto.analysis.lowering.helpers import build_estimation_context
from zepto.graph.graph import Graph


def account_fcm_flops(graph: Graph, context: InvocationContext) -> tuple[int, int]:
    """Return ``(forward, backward)`` using Zepto formulas on allowlisted families only."""
    forward = 0
    backward = 0
    for node_id in graph.node_order:
        node = graph.node(node_id)
        if not is_fcm_operation_family(node.operation_family):
            continue
        op = node.declaration
        if op is None:
            continue
        estimation = build_estimation_context(node, graph, context)
        fwd = op.forward_flops(estimation)
        bwd = op.backward_flops(replace(estimation, phase="backward"))
        if not isinstance(fwd, int) or fwd < 0:
            raise ValueError(f"{op.family}: forward_flops must be non-negative int")
        if not isinstance(bwd, int) or bwd < 0:
            raise ValueError(f"{op.family}: backward_flops must be non-negative int")
        forward += fwd
        backward += bwd
    return (forward, backward)
