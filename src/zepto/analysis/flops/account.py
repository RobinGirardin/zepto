"""FLOP accounting entry point."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.analysis.horizon.records import HorizonSimulation
from zepto.analysis.lowered import LoweredGraph, LoweredNode
from zepto.analysis.reports.attribution import AttributionSlice
from zepto.analysis.reports.flops import FlopReport
from zepto.analysis.reports.horizon import HorizonFlopReport


@dataclass
class _FlopAccumulator:
    forward_flops: int = 0
    backward_flops: int = 0


def account_flops(
    target: LoweredGraph | HorizonSimulation,
) -> FlopReport | HorizonFlopReport:
    """Sum lowered node FLOPs for one invocation or a horizon."""
    if isinstance(target, LoweredGraph):
        return _account_single_flops(target)
    from zepto.analysis.horizon.account import account_horizon_flops

    return account_horizon_flops(target)


def _phase_total(forward: int, backward: int, phase: str) -> int:
    if phase == "forward":
        return forward
    if phase == "backward":
        return backward
    return forward + backward


def _assemble_flop_report(
    *,
    zepto_forward: int,
    zepto_backward: int,
    fcm_forward: int,
    fcm_backward: int,
    phase: str,
    policy: str,
    by_module: tuple = (),
    by_region: tuple = (),
    by_implementation: tuple = (),
) -> FlopReport:
    zepto_total = _phase_total(zepto_forward, zepto_backward, phase)
    fcm_total = _phase_total(fcm_forward, fcm_backward, phase)
    if policy == "flop_counter_mode":
        fwd, bwd, total = fcm_forward, fcm_backward, fcm_total
    else:
        fwd, bwd, total = zepto_forward, zepto_backward, zepto_total
    return FlopReport(
        forward_flops=fwd,
        backward_flops=bwd,
        total_flops=total,
        by_module=by_module,
        by_region=by_region,
        by_implementation=by_implementation,
        flop_policy=policy,
        zepto_forward_flops=zepto_forward,
        zepto_backward_flops=zepto_backward,
        zepto_total_flops=zepto_total,
        fcm_forward_flops=fcm_forward,
        fcm_backward_flops=fcm_backward,
        fcm_total_flops=fcm_total,
    )


def _account_single_flops(lowered: LoweredGraph) -> FlopReport:
    forward = sum(node.forward_flops for node in lowered.nodes)
    backward = sum(node.backward_flops for node in lowered.nodes)
    return _assemble_flop_report(
        zepto_forward=forward,
        zepto_backward=backward,
        fcm_forward=0,
        fcm_backward=0,
        phase=lowered.context.phase,
        policy=lowered.context.flop_policy,
        by_module=_rollup_flops(lowered.nodes, _module_key),
        by_region=_rollup_flops(lowered.nodes, _region_key),
        by_implementation=_rollup_flops(lowered.nodes, _implementation_key),
    )


def _module_key(node: LoweredNode) -> str | None:
    key = "/".join(node.module_path) or "root"
    return key


def _region_key(node: LoweredNode) -> str | None:
    return node.region_id


def _implementation_key(node: LoweredNode) -> str | None:
    return node.implementation


def _rollup_flops(
    nodes: tuple[LoweredNode, ...],
    key_fn,
) -> tuple[AttributionSlice, ...]:
    rollups: dict[str, _FlopAccumulator] = {}
    for node in nodes:
        key = key_fn(node)
        if not key:
            continue
        acc = rollups.setdefault(key, _FlopAccumulator())
        acc.forward_flops += node.forward_flops
        acc.backward_flops += node.backward_flops
    return tuple(
        AttributionSlice(
            key=key,
            forward_flops=acc.forward_flops,
            backward_flops=acc.backward_flops,
        )
        for key, acc in sorted(rollups.items())
    )
