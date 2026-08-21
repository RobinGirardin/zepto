"""Composition helpers for lowering and future cost estimation."""

from __future__ import annotations

from .graph import StructuralGraph
from .lowering import InvocationContext, lower
from .lowered import LoweredGraph

__all__ = ["LoweredGraph", "lower", "InvocationContext"]


def estimate(
    graph: StructuralGraph,
    context: InvocationContext,
    *,
    memory_policy: object | None = None,
) -> None:
    """Compose lowering with memory and FLOP accounting.

    Full ``CostReport`` support is deferred until memory and FLOP modules land.
    """
    del memory_policy
    lower(graph, context)
    raise NotImplementedError(
        "estimate() requires account_memory() and account_flops()"
    )
