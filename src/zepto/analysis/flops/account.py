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


def _account_single_flops(lowered: LoweredGraph) -> FlopReport:
    forward = sum(node.forward_flops for node in lowered.nodes)
    backward = sum(node.backward_flops for node in lowered.nodes)
    total = _select_total(forward, backward, lowered.context.phase)
    return FlopReport(
        forward_flops=forward,
        backward_flops=backward,
        total_flops=total,
        by_module=_rollup_flops(lowered.nodes, _module_key),
        by_region=_rollup_flops(lowered.nodes, _region_key),
        by_implementation=_rollup_flops(lowered.nodes, _implementation_key),
    )


def _select_total(forward: int, backward: int, phase: str) -> int:
    if phase == "forward":
        return forward
    if phase == "backward":
        return backward
    return forward + backward


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
