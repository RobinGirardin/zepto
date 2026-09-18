"""Discovery, lowering, and variant tests for region/linear_ce_softcap."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.linear_ce import DEFAULT_LINEAR_CE_RECIPE
from zepto.analysis.lowering.recipes.linear_ce_softcap import (
    DEFAULT_LINEAR_CE_SOFTCAP_RECIPE,
)
from zepto.compose import Tensor, compose_graph
from zepto.modules.output.capped_fused_linear_cross_entropy import (
    CappedFusedLinearCrossEntropy,
)
from zepto.modules.output.logit_soft_cap import LogitSoftCapConfig
from zepto.semantic import ResourceEventKind

_S, _D, _V = 4, 8, 16


def _fused_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused"}),
        "hardware": "cuda",
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _compose_linear_ce_softcap():
    return compose_graph(
        lambda ctx: CappedFusedLinearCrossEntropy(
            hidden_size=_D,
            vocab_size=_V,
            soft_cap=LogitSoftCapConfig(cap=20.0),
        ),
        (
            Tensor(shape=(_S, _D), requires_grad=True),
            Tensor(shape=(_S,), semantic_type="labels", requires_grad=False),
        ),
    )


def test_compose_linear_ce_softcap_discovers_region() -> None:
    graph = _compose_linear_ce_softcap()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    softcap = [r for r in regions if r.kind == "region/linear_ce_softcap"]
    assert len(softcap) == 1
    assert len(softcap[0].operation_ids) == 12
    assert softcap[0].anchor.component_type == "CappedFusedLinearCrossEntropy"


def test_linear_ce_softcap_unfused_without_fused_capability() -> None:
    graph = _compose_linear_ce_softcap()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 12


def test_linear_ce_softcap_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_linear_ce_softcap()
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/linear_ce_softcap/liger"
    expected_fwd = DEFAULT_LINEAR_CE_SOFTCAP_RECIPE.forward_flops(_S, _D, _V)
    expected_bwd = DEFAULT_LINEAR_CE_SOFTCAP_RECIPE.backward_flops(
        _S, _D, _V, requires_grad=True
    )
    assert op.forward_flops == expected_fwd
    assert op.backward_flops == expected_bwd
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 2
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any("logits_chunk" in aux for aux in op.auxiliary_edges)


def test_linear_ce_softcap_fusion_map_absorbs_all_ops() -> None:
    graph = _compose_linear_ce_softcap()
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == 12
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_linear_ce_softcap_liger_wins_on_cuda() -> None:
    graph = _compose_linear_ce_softcap()
    lowered = lower(graph, _fused_context(hardware="cuda"))
    assert lowered.nodes[0].implementation == "region/linear_ce_softcap/liger"


def test_linear_ce_softcap_reference_on_xpu() -> None:
    graph = _compose_linear_ce_softcap()
    lowered = lower(
        graph,
        reference_invocation(
            requested_capabilities=frozenset({"fused"}),
            hardware="xpu",
        ),
    )
    assert lowered.nodes[0].implementation == "region/linear_ce_softcap/reference"


def test_linear_ce_softcap_peak_logits_chunk_formula() -> None:
    recipe = DEFAULT_LINEAR_CE_SOFTCAP_RECIPE
    s, d, v = 8192, 4096, 131072
    peak = recipe.peak_logits_bytes(s, d, v, elem_bytes=2)
    assert peak == 2**30
    assert recipe.chunk_size(s, d, v) == 4096


def test_linear_ce_softcap_fwd_delta_vs_linear_ce() -> None:
    softcap_recipe = DEFAULT_LINEAR_CE_SOFTCAP_RECIPE
    ce_recipe = DEFAULT_LINEAR_CE_RECIPE
    delta = softcap_recipe.forward_flops(_S, _D, _V) - ce_recipe.forward_flops(
        _S, _D, _V
    )
    assert delta == 7 * _S * _V
