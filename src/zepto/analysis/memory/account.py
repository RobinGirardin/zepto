"""Memory accounting entry point."""

from __future__ import annotations

from zepto.analysis.lowered import LoweredGraph
from zepto.analysis.reports.memory import MemoryReport

from .simulator import ResourceEventSimulator


def account_memory(lowered: LoweredGraph) -> MemoryReport:
    """Run resource event simulation and return a memory report."""
    result = ResourceEventSimulator(lowered).run()
    return MemoryReport(
        sum_all_bytes=result.sum_all_bytes,
        peak_live_bytes=result.peak_live_bytes,
        breakdown=result.breakdown,
        by_module=result.by_module,
        by_region=result.by_region,
        by_implementation=result.by_implementation,
        live_timeline=result.live_timeline,
    )
