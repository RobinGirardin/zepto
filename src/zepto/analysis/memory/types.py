"""Shared memory simulation record types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from zepto.analysis.reports.attribution import AttributionSlice
from zepto.analysis.reports.memory import MemoryBreakdown
from zepto.semantic.operations.records import ResourceEventKind


@dataclass
class StorageSlot:
    """Live storage tracked by the resource event simulator."""

    storage_id: str
    bytes: int
    live: bool
    refcount: int
    kind: Literal[
        "parameter",
        "activation",
        "saved",
        "workspace",
        "gradient_wavefront",
        "weight_grad",
        "state",
        "persistent_input",
    ]
    pinned: bool
    producer_node: int | None
    edge_id: str | None


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """One ordered point on the memory timeline."""

    node_index: int
    kind: ResourceEventKind | Literal["scheduled"]
    edge_id: str | None
    storage_id: str
    delta_live: int
    phase: str
    scheduled: bool = False
    cumulative_live: int = 0


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Output of one resource event simulation run."""

    sum_all_bytes: int
    peak_live_bytes: int
    peak_live_bytes_naive: int
    breakdown: MemoryBreakdown
    timeline: tuple[TimelineEvent, ...]
    by_module: tuple[AttributionSlice, ...] = ()
    by_region: tuple[AttributionSlice, ...] = ()
    by_implementation: tuple[AttributionSlice, ...] = ()
    live_timeline: tuple[tuple[int, int], ...] | None = None


@dataclass
class _AttributionAccumulator:
    """Mutable rollup state keyed by attribution label."""

    forward_flops: int = 0
    backward_flops: int = 0
    allocated_bytes: int = 0
    peak_bytes: int = 0
    persistent_bytes: int = 0
    workspace_bytes: int = 0
    peak_live_at_node: dict[int, int] = field(default_factory=dict)
