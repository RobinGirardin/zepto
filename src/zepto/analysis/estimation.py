"""Composition helpers for lowering and cost estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zepto.graph.graph import Graph

from .lowering import InvocationContext, lower
from .reports import CostReport, HorizonCostReport

if TYPE_CHECKING:
    from .horizon import HorizonSimulation, HorizonSpec
    from .horizon.simulate import InputsFn, ModuleFn
    from .lowered import LoweredGraph

__all__ = [
    "CostReport",
    "HorizonCostReport",
    "InvocationContext",
    "estimate",
    "estimate_horizon",
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


def estimate_horizon(
    horizon: HorizonSpec,
    module_fn: ModuleFn,
    inputs_fn: InputsFn,
    context: InvocationContext,
    *,
    return_simulation: bool = False,
) -> HorizonCostReport | tuple[HorizonCostReport, HorizonSimulation]:
    """Simulate a horizon and return combined memory and FLOP accounting."""
    from .flops import account_flops
    from .horizon import simulate_horizon
    from .memory import account_memory

    sim = simulate_horizon(horizon, module_fn, inputs_fn, context)
    hmem = account_memory(sim)
    hflops = account_flops(sim)
    per_step = tuple(
        CostReport(
            memory=step_mem,
            flops=step_flop,
            context=record.lowered.context,
        )
        for record, step_mem, step_flop in zip(
            sim.timeline,
            hmem.per_step,
            hflops.per_step,
            strict=True,
        )
    )
    report = HorizonCostReport(
        peak_vram=hmem.peak_live_bytes,
        total_flops=hflops.total_flops,
        per_step=per_step,
        state_final=sim.state_final,
        memory=hmem,
        flops=hflops,
    )
    return (report, sim) if return_simulation else report
