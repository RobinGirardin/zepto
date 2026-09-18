"""Discovery and lowering tests for region/logit_softcap."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.output.logit_soft_cap import LogitSoftCap, LogitSoftCapConfig


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def test_compose_logit_soft_cap_discovers_region() -> None:
    graph = compose_graph(
        lambda _ctx: LogitSoftCap(LogitSoftCapConfig(cap=20.0)),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    soft_cap = [r for r in regions if r.kind == "region/logit_softcap"]
    assert len(soft_cap) == 1
    assert soft_cap[0].anchor.component_type == "LogitSoftCap"
    assert len(soft_cap[0].operation_ids) == 4


def test_logit_soft_cap_fused_lowering_single_node() -> None:
    graph = compose_graph(
        lambda _ctx: LogitSoftCap(LogitSoftCapConfig(cap=30.0)),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, _fused_context())
    region_nodes = [n for n in lowered.nodes if n.implementation.startswith("region/")]
    assert len(region_nodes) == 1
    assert region_nodes[0].forward_flops == 7 * 4 * 8


def test_logit_soft_cap_unfused_without_fused_capability() -> None:
    graph = compose_graph(
        lambda _ctx: LogitSoftCap(LogitSoftCapConfig(cap=20.0)),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 4
