"""Discovery, lowering, and variant tests for region/geglu."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.analysis.lowering.recipes.geglu import DEFAULT_GEGLU_RECIPE
from zepto.compose import Tensor, compose_graph
from zepto.modules.ffn.ffn import FFN
from zepto.modules.ffn.geglu import GeGLU
from zepto.modules.ffn.swiglu import SwiGLU
from zepto.semantic import ResourceEventKind


def _geglu_graph(
    *,
    seq_len: int = 8192,
    hidden_size: int = 3072,
    intermediate_size: int = 24576,
    requires_grad: bool = True,
    gelu_approx: str = "tanh",
):
    return compose_graph(
        lambda ctx: GeGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gelu_approx=gelu_approx,
        ),
        (Tensor(shape=(seq_len, hidden_size), requires_grad=requires_grad),),
    )


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_compose_geglu_discovers_region() -> None:
    graph = _geglu_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    geglu = [r for r in regions if r.kind == "region/geglu"]
    assert len(geglu) == 1
    assert geglu[0].anchor.component_type == "GeGLU"
    assert len(geglu[0].operation_ids) == 5


def test_geglu_decomposed_lowering_flops_and_allocs() -> None:
    seq_len, hidden_size, intermediate_size = 8192, 3072, 24576
    graph = _geglu_graph(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/geglu/decomposed"
    n = seq_len * intermediate_size
    assert op.forward_flops == 6 * seq_len * hidden_size * intermediate_size + 13 * n
    assert op.backward_flops == 12 * seq_len * hidden_size * intermediate_size + 22 * n
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    save_targets = {ev.value for ev in saves}
    assert any("gate" in target for target in save_targets)
    assert any("up" in target for target in save_targets)
    assert any("h" in target for target in save_targets)


def test_geglu_decomposed_erf_variant() -> None:
    seq_len, hidden_size, intermediate_size = 4, 8, 16
    graph = _geglu_graph(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gelu_approx="none",
    )
    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]
    assert op.implementation == "region/geglu/decomposed-erf"
    n = seq_len * intermediate_size
    assert op.forward_flops == 6 * seq_len * hidden_size * intermediate_size + 9 * n
    assert op.backward_flops == 12 * seq_len * hidden_size * intermediate_size + 19 * n


def test_geglu_fusion_map_absorbs_all_ops() -> None:
    graph = _geglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    lowered = lower(graph, reference_invocation())
    assert len(lowered.fusion_map) == 5
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_geglu_unfused_without_capability() -> None:
    graph = _geglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(
        graph, regions, ctx, _identity_registry()
    )
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 5


def test_geglu_liger_wins_on_cuda_with_fused_post_gemm() -> None:
    graph = _geglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    ctx = reference_invocation(
        hardware="cuda",
        requested_capabilities=frozenset({"fused_post_gemm"}),
    )
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/geglu/liger"
    impl_ids = {node.implementation for node in lowered.nodes}
    assert "region/gelu" not in impl_ids


def test_geglu_no_backward_without_grad() -> None:
    graph = _geglu_graph(
        seq_len=4,
        hidden_size=8,
        intermediate_size=16,
        requires_grad=False,
    )
    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]
    assert op.backward_flops == 0
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert saves == []


def test_geglu_paper_flops_helper() -> None:
    seq_len, hidden_size, intermediate_size = 8192, 3072, 24576
    recipe = DEFAULT_GEGLU_RECIPE
    assert recipe.paper_forward_flops(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ) == 6 * seq_len * hidden_size * intermediate_size


def test_geglu_does_not_match_ungated_ffn() -> None:
    graph = compose_graph(
        lambda ctx: FFN(hidden_size=8, intermediate_size=16),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    assert not any(r.kind == "region/geglu" for r in regions)


def test_geglu_does_not_match_swiglu() -> None:
    graph = compose_graph(
        lambda ctx: SwiGLU(hidden_size=8, intermediate_size=16),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    assert not any(r.kind == "region/geglu" for r in regions)
