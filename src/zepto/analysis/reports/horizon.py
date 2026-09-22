"""Horizon-level memory and FLOP report types."""

from __future__ import annotations

from dataclasses import dataclass

from .flops import FlopReport
from .memory import MemoryBreakdown, MemoryReport


@dataclass(frozen=True, slots=True)
class HorizonMemoryReport:
    """Memory accounting reduced across a horizon timeline.

    ``breakdown.runtime_workspace`` is the **max** per-step cuBLAS pool
    (peak-relevant). Handle pools are ephemeral; summing them would count
    1+2+0 handles on a G=1 train instead of the 2 at peak. Graph-local
    ``workspace`` is still summed. ``sum_all_bytes`` already includes
    per-step runtime.
    """

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
    zepto_forward_flops: int = 0
    zepto_backward_flops: int = 0
    zepto_total_flops: int = 0
    fcm_forward_flops: int = 0
    fcm_backward_flops: int = 0
    fcm_total_flops: int = 0
