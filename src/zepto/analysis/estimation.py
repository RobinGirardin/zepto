"""Composition helpers for lowering and cost estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zepto.graph.graph import Graph

from .lowering import InvocationContext, lower
from .reports import CostReport

if TYPE_CHECKING:
    from .lowered import LoweredGraph

__all__ = [
    "CostReport",
    "InvocationContext",
    "estimate",
    "lower",
]


def estimate(
    graph: Graph,
    context: InvocationContext,
    *,
    return_lowered: bool = False,
) -> CostReport | tuple[CostReport, LoweredGraph]:
    """Compose lowering with memory and FLOP accounting."""
    from .flops import account_flops
    from .memory import account_memory

    lowered = lower(graph, context)
    mem = account_memory(lowered)
    flops = account_flops(lowered)
    report = CostReport(memory=mem, flops=flops, context=context)
    return (report, lowered) if return_lowered else report
