"""FLOP report types."""

from __future__ import annotations

from dataclasses import dataclass

from .attribution import AttributionSlice


@dataclass(frozen=True, slots=True)
class FlopReport:
    """FLOP accounting result for one invocation."""

    forward_flops: int
    backward_flops: int
    total_flops: int
    by_module: tuple[AttributionSlice, ...] = ()
    by_region: tuple[AttributionSlice, ...] = ()
    by_implementation: tuple[AttributionSlice, ...] = ()
