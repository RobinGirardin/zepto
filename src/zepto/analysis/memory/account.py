"""Memory accounting entry point."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis.horizon.records import HorizonSimulation
from zepto.analysis.lowered import LoweredGraph
from zepto.analysis.reports.horizon import HorizonMemoryReport
from zepto.analysis.reports.memory import MemoryReport
from zepto.analysis.runtime import runtime_workspace_bytes

from .simulator import ResourceEventSimulator


def account_memory(
    target: LoweredGraph | HorizonSimulation,
) -> MemoryReport | HorizonMemoryReport:
    """Run resource event simulation for one invocation or a horizon."""
    if isinstance(target, LoweredGraph):
        return _account_single_memory(target)
    from zepto.analysis.horizon.account import account_horizon_memory

    return account_horizon_memory(target)


def _account_single_memory(lowered: LoweredGraph) -> MemoryReport:
    result = ResourceEventSimulator(lowered).run()
    runtime_ws = runtime_workspace_bytes(lowered.context)
    breakdown = replace(result.breakdown, runtime_workspace=runtime_ws)
    peak = result.peak_live_bytes + runtime_ws
    return MemoryReport(
        sum_all_bytes=result.sum_all_bytes + runtime_ws,
        peak_live_bytes=peak,
        breakdown=breakdown,
        by_module=result.by_module,
        by_region=result.by_region,
        by_implementation=result.by_implementation,
        live_timeline=result.live_timeline,
    )
