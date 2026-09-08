"""Discovery, lowering, and variant tests for region/linear_ce."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.linear_ce import DEFAULT_LINEAR_CE_RECIPE
from zepto.compose import Tensor, compose_graph
from zepto.modules.fused_linear_cross_entropy import FusedLinearCrossEntropy
from zepto.semantic import ResourceEventKind

_S, _D, _V = 4, 8, 16


def _fused_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused"}),
        "hardware": "cuda",
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _compose_linear_ce():
    return compose_graph(
        lambda ctx: FusedLinearCrossEntropy(hidden_size=_D, vocab_size=_V),
        (
            Tensor(shape=(_S, _D), requires_grad=True),
            Tensor(shape=(_S,), semantic_type="labels", requires_grad=False),
        ),
    )


def test_compose_linear_ce_discovers_region() -> None:
    graph = _compose_linear_ce()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, _fused_context(), registry)
    linear_ce = [r for r in regions if r.kind == "region/linear_ce"]
    assert len(linear_ce) == 1
    assert len(linear_ce[0].operation_ids) == 8
    assert linear_ce[0].anchor.component_type == "FusedLinearCrossEntropy"


def test_linear_ce_unfused_without_fused_capability() -> None:
    graph = _compose_linear_ce()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 8


def test_linear_ce_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_linear_ce()
    lowered = lower(graph, _fused_context())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/linear_ce/liger"
    expected_fwd = DEFAULT_LINEAR_CE_RECIPE.forward_flops(_S, _D, _V)
    expected_bwd = DEFAULT_LINEAR_CE_RECIPE.backward_flops(
        _S, _D, _V, requires_grad=True
    )
    assert op.forward_flops == expected_fwd
    assert op.backward_flops == expected_bwd
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    assert len(allocs) == 2  # logits_chunk + loss scalar
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1  # hidden
    assert any("logits_chunk" in aux for aux in op.auxiliary_edges)


def test_linear_ce_fusion_map_absorbs_all_ops() -> None:
    graph = _compose_linear_ce()
    lowered = lower(graph, _fused_context())
    assert len(lowered.fusion_map) == 8
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_linear_ce_liger_wins_on_cuda() -> None:
    graph = _compose_linear_ce()
    lowered = lower(graph, _fused_context(hardware="cuda"))
    assert lowered.nodes[0].implementation == "region/linear_ce/liger"


def test_linear_ce_reference_on_xpu() -> None:
    graph = _compose_linear_ce()
    lowered = lower(
        graph,
        reference_invocation(
            requested_capabilities=frozenset({"fused"}),
            hardware="xpu",
        ),
    )
    assert lowered.nodes[0].implementation == "region/linear_ce/reference"


def test_linear_ce_peak_logits_chunk_formula() -> None:
    recipe = DEFAULT_LINEAR_CE_RECIPE
    s, d, v = 8192, 4096, 131072
    peak = recipe.peak_logits_bytes(s, d, v, elem_bytes=2)
    assert peak == 2**30  # 1.0 GiB at Apertus-8B with C=16
    assert recipe.chunk_size(s, d, v) == 4096
