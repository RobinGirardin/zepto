"""Zepto-side FLOP and VRAM measurements only."""

from __future__ import annotations

from dataclasses import dataclass, replace

from zepto.analysis import AdamW, HorizonSpec, estimate, estimate_horizon
from zepto.analysis.horizon.spec import HorizonStep, StepKind
from zepto.analysis.reports.cost import HorizonCostReport
from zepto.analysis.reports.memory import MemoryBreakdown
from zepto.compose import compose_graph

from zepto.empirical.models.base import ModelFamily

ZEPTO_BATCH_REP = "parallel_batch"


@dataclass(frozen=True, slots=True)
class ZeptoMeasureResult:
    y_flop: int
    y_vram: int
    zepto_batch_representation: str
    y_runtime_workspace: int
    y_activations: int
    y_flop_no_opt: int


def _y_runtime_workspace_from_breakdown(breakdown: MemoryBreakdown) -> int:
    """Smoke parity: runtime cuBLAS policy plus graph lowering workspace."""
    return breakdown.runtime_workspace + breakdown.workspace


def _y_runtime_workspace_from_horizon(report: HorizonCostReport) -> int:
    """Workspace of the peak-relevant step (max of per-step runtime+workspace).

    Horizon ``breakdown.runtime_workspace`` is already the max cuBLAS pool;
    this still adds graph-local ``workspace`` on that same step.
    """
    return max(
        _y_runtime_workspace_from_breakdown(step.memory.breakdown)
        for step in report.per_step
    )


def measure_infer_zepto(
    family: ModelFamily,
    opts: dict[str, int],
    *,
    seq_len: int,
    batch_size: int,
    ctx,
) -> ZeptoMeasureResult:
    """One parallel (B, S) forward. Does not use HorizonSpec.repeat."""
    module_factory = family.build_zepto_infer_module_factory(opts, seq_len=seq_len)
    step = HorizonStep(
        kind=StepKind.GENERIC, seq_len=seq_len, batch=batch_size, phase="forward"
    )
    forward_ctx = replace(ctx, phase="forward")
    inputs = family.zepto_infer_inputs(step, forward_ctx, None)
    graph = compose_graph(module_factory, inputs)
    report = estimate(graph, forward_ctx)
    bd = report.memory.breakdown
    y_flop = int(report.flops.forward_flops)
    return ZeptoMeasureResult(
        y_flop=y_flop,
        y_vram=int(report.memory.peak_live_bytes),
        zepto_batch_representation=ZEPTO_BATCH_REP,
        y_runtime_workspace=_y_runtime_workspace_from_breakdown(bd),
        y_activations=bd.activations,
        y_flop_no_opt=y_flop,
    )


def measure_train_zepto(
    family: ModelFamily,
    opts: dict[str, int],
    *,
    seq_len: int,
    batch_size: int,
    ctx,
) -> ZeptoMeasureResult:
    spec = HorizonSpec.training(
        seq_len=seq_len,
        batch=batch_size,
        micro_batches=1,
        optimizer=AdamW,
    )
    module_fn = family.build_zepto_train_module_factory(opts, seq_len=seq_len)
    report = estimate_horizon(spec, module_fn, family.zepto_train_inputs, ctx)
    bd = report.memory.breakdown
    y_flop = int(report.total_flops)

    spec_no_opt = HorizonSpec.training(
        seq_len=seq_len,
        batch=batch_size,
        micro_batches=1,
        optimizer=None,
    )
    report_no_opt = estimate_horizon(
        spec_no_opt, module_fn, family.zepto_train_inputs, ctx
    )
    y_flop_no_opt = int(report_no_opt.total_flops)

    return ZeptoMeasureResult(
        y_flop=y_flop,
        y_vram=int(report.peak_vram),
        zepto_batch_representation=ZEPTO_BATCH_REP,
        y_runtime_workspace=_y_runtime_workspace_from_horizon(report),
        y_activations=bd.activations,
        y_flop_no_opt=y_flop_no_opt,
    )
