"""Memory report types."""

from __future__ import annotations

from dataclasses import dataclass

from .attribution import AttributionSlice


@dataclass(frozen=True, slots=True)
class MemoryBreakdown:
    """Sum-all byte inventory by category."""

    activations: int = 0
    parameters: int = 0
    workspace: int = 0
    saved_for_backward: int = 0
    persistent_inputs: int = 0
    gradients: int = 0
    weight_grads: int = 0
    state: int = 0


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """Memory accounting result for one invocation."""

    sum_all_bytes: int
    peak_live_bytes: int
    breakdown: MemoryBreakdown
    by_module: tuple[AttributionSlice, ...] = ()
    by_region: tuple[AttributionSlice, ...] = ()
    by_implementation: tuple[AttributionSlice, ...] = ()
    live_timeline: tuple[tuple[int, int], ...] | None = None
