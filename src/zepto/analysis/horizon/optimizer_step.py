"""State-only optimizer horizon steps (no compose / lower)."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from ..lowered import LoweredGraph
from ..lowering.context import InvocationContext
from .records import InvocationRecord
from .spec import StepKind
from .state import parameter_bytes


def reference_lowered_for_optimizer(
    timeline: Sequence[InvocationRecord],
) -> LoweredGraph:
    """Return the most recent lowered graph before an optimizer boundary."""
    for record in reversed(timeline):
        if record.step.kind != StepKind.OPTIMIZER:
            return record.lowered
    raise ValueError(
        "optimizer step requires a prior invocation with a lowered graph"
    )


def make_optimizer_stub_lowered(
    reference: LoweredGraph,
    ctx: InvocationContext,
) -> LoweredGraph:
    """Build a parameter-only lowered graph for optimizer state accounting."""
    stub = LoweredGraph(
        edges=MappingProxyType({}),
        parameters=reference.parameters,
        parameter_map=reference.parameter_map,
        nodes=(),
        context=ctx,
        edge_map=MappingProxyType({}),
        node_map=MappingProxyType({}),
        output_edge_ids=(),
        selections=(),
        fusion_map=MappingProxyType({}),
        region_map=MappingProxyType({}),
        region_selections=(),
        state_port_events=(),
        discovered_regions=(),
        winning_regions=(),
        lowering_plan=None,
        source_graph=None,
    )
    if parameter_bytes(stub) != parameter_bytes(reference):
        raise ValueError("optimizer stub parameter bytes mismatch")
    return stub
