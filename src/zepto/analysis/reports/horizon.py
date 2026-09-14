"""Horizon-level memory and FLOP report types."""

from __future__ import annotations

from dataclasses import dataclass

from .flops import FlopReport
from .memory import MemoryBreakdown, MemoryReport


@dataclass(frozen=True, slots=True)
class HorizonMemoryReport:
    """Memory accounting reduced across a horizon timeline."""

    peak_live_bytes: int
    sum_all_bytes: int
    persistent_carry_bytes: int
    per_step: tuple[MemoryReport, ...]
    breakdown: MemoryBreakdown


@dataclass(frozen=True, slots=True)
class HorizonFlopReport:
    """FLOP accounting summed across a horizon timeline."""

    total_forward_flops: int
    total_backward_flops: int
    total_flops: int
    per_step: tuple[FlopReport, ...]
