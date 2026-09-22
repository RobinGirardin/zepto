"""Lowering schedule construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from zepto.graph.graph import Graph
from zepto.graph.ids import NodeId
from .context import InvocationContext
from .region import Region
from .registry import LoweringRegistry, region_has_compatible_implementation


@dataclass(frozen=True, slots=True)
class LoweringStep:
    """One step in the lowering schedule."""

    kind: Literal["region", "operation"]
    region: Region | None = None
    operation_id: NodeId | None = None


@dataclass(frozen=True, slots=True)
class LoweringPlan:
    """Non-overlapping lowering schedule."""

    steps: tuple[LoweringStep, ...]
    consumed: frozenset[NodeId]
    region_by_op: Mapping[NodeId, str]


def resolve_overlaps(
    graph: Graph,
    regions: tuple[Region, ...],
    context: InvocationContext,
    registry: LoweringRegistry,
) -> tuple[Region, ...]:
    """Select disjoint winning regions from overlapping candidates."""
    viable = [
        region
        for region in regions
        if region_has_compatible_implementation(
            region, registry, context, graph=graph
        )
    ]

    def sort_key(region: Region) -> tuple:
        pinned = 0
        if region.kind in context.region_implementation_pins:
            pinned = 2
        elif (
            region.anchor.component_type
            and region.anchor.component_type in context.module_implementation_pins
        ):
            pinned = 1
        descriptor = registry.region_descriptor(region.kind)
        return (
            -pinned,
            -descriptor.priority,
            -len(region.operation_ids),
            -_matcher_priority(region),
            region.id,
        )

    sorted_regions = sorted(viable, key=sort_key)
    claimed: set[NodeId] = set()
    winners: list[Region] = []

    for region in sorted_regions:
        op_set = set(region.operation_ids)
        if op_set & claimed:
            continue
        claimed.update(op_set)
        winners.append(region)

    return tuple(winners)


def _matcher_priority(region: Region) -> int:
    if region.matcher_id.startswith("hybrid:"):
        return 1
    if region.matcher_id.startswith("pat-"):
        return 0
    return 0


def build_lowering_plan_from_winners(
    graph: Graph,
    winners: tuple[Region, ...],
) -> LoweringPlan:
    """Build an ordered schedule from already-resolved winning regions."""
    region_starts: dict[NodeId, Region] = {}
    region_by_op: dict[NodeId, str] = {}
    consumed: set[NodeId] = set()

    for region in winners:
        first_op = region.operation_ids[0]
        region_starts[first_op] = region
        for op_id in region.operation_ids:
            region_by_op[op_id] = region.id

    steps: list[LoweringStep] = []
    for operation_id in graph.node_order:
        if operation_id in consumed:
            continue
        if operation_id in region_starts:
            region = region_starts[operation_id]
            steps.append(LoweringStep(kind="region", region=region))
            consumed.update(region.operation_ids)
            continue
        steps.append(LoweringStep(kind="operation", operation_id=operation_id))
        consumed.add(operation_id)

    return LoweringPlan(
        steps=tuple(steps),
        consumed=frozenset(consumed),
        region_by_op=region_by_op,
    )


def build_lowering_plan(
    graph: Graph,
    regions: tuple[Region, ...],
    context: InvocationContext,
    registry: LoweringRegistry,
) -> LoweringPlan:
    """Build an ordered non-overlapping lowering schedule."""
    winners = resolve_overlaps(graph, regions, context, registry)
    return build_lowering_plan_from_winners(graph, winners)
