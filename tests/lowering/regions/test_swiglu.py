"""Discovery, lowering, and variant tests for region/swiglu."""

from __future__ import annotations

from types import MappingProxyType

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults, register_identity_defaults
from zepto.analysis.lowering.recipes.swiglu import DEFAULT_SWIGLU_RECIPE
from zepto.compose import Tensor, compose_graph
from zepto.modules.ffn.ffn import FFN
from zepto.modules.ffn.swiglu import SwiGLU
from zepto.semantic import ResourceEventKind


def _swiglu_graph(
    *,
    seq_len: int = 8192,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    requires_grad: bool = True,
):
    return compose_graph(
        lambda ctx: SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size),
        (Tensor(shape=(seq_len, hidden_size), requires_grad=requires_grad),),
    )


def _identity_registry() -> LoweringRegistry:
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    return registry


def test_compose_swiglu_discovers_region() -> None:
    graph = _swiglu_graph()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    swiglu = [r for r in regions if r.kind == "region/swiglu"]
    assert len(swiglu) == 1
    assert swiglu[0].anchor.component_type == "SwiGLU"
    assert len(swiglu[0].operation_ids) == 6


def test_swiglu_decomposed_lowering_flops_and_allocs() -> None:
    seq_len, hidden_size, intermediate_size = 8192, 4096, 14336
    graph = _swiglu_graph(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    lowered = lower(graph, reference_invocation())
    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "region/swiglu/decomposed"
    n = seq_len * intermediate_size
    assert op.forward_flops == 6 * seq_len * hidden_size * intermediate_size + 6 * n
    assert op.backward_flops == 12 * seq_len * hidden_size * intermediate_size + 10 * n
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    save_targets = {ev.value for ev in saves}
    assert any("gate" in target for target in save_targets)
    assert any("up" in target for target in save_targets)
    assert any("h" in target for target in save_targets)


def test_swiglu_fusion_map_absorbs_all_ops() -> None:
    graph = _swiglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    lowered = lower(graph, reference_invocation())
    assert len(lowered.fusion_map) == 6
    region_op_id = lowered.nodes[0].id
    assert all(v == region_op_id for v in lowered.fusion_map.values())


def test_swiglu_unfused_without_capability() -> None:
    graph = _swiglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation()
    regions = discover_regions(graph, ctx, registry)
    plan = build_lowering_plan(
        graph, regions, ctx, _identity_registry()
    )
    assert all(step.kind == "operation" for step in plan.steps)
    assert len(plan.steps) == 6


def test_swiglu_decomposed_wins_with_fused_only() -> None:
    graph = _swiglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    ctx = reference_invocation(
        hardware="cuda",
        requested_capabilities=frozenset({"fused"}),
    )
    lowered = lower(graph, ctx)
    assert lowered.nodes[0].implementation == "region/swiglu/decomposed"


def test_swiglu_liger_wins_on_cuda_with_fused_post_gemm() -> None:
    graph = _swiglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    ctx = reference_invocation(
        hardware="cuda",
        requested_capabilities=frozenset({"fused_post_gemm"}),
    )
    lowered = lower(graph, ctx)
    assert lowered.region_selections[0].chosen.id == "region/swiglu/liger"
    impl_ids = {node.implementation for node in lowered.nodes}
    assert "region/silu" not in impl_ids


def test_swiglu_liger_fused_gate_up_save_policy() -> None:
    seq_len, intermediate_size = 4, 16
    graph = _swiglu_graph(
        seq_len=seq_len,
        hidden_size=8,
        intermediate_size=intermediate_size,
    )
    ctx = reference_invocation(
        hardware="cuda",
        requested_capabilities=frozenset({"fused_gate_up"}),
        region_implementation_pins=MappingProxyType(
            {"region/swiglu": "region/swiglu/liger-fused-gate-up"}
        ),
    )
    lowered = lower(graph, ctx)
    op = lowered.nodes[0]
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 2  # fused gate_up + down input h
    gate_up_aux = [aux for aux in op.auxiliary_edges if "gate_up" in aux]
    assert len(gate_up_aux) == 1
    gate_up_tensor = None
    for aux in op.auxiliary_edges:
        if "gate_up" in aux:
            gate_up_tensor = aux
    assert gate_up_tensor is not None
    assert not any("gate" in aux and "gate_up" not in aux for aux in op.auxiliary_edges)
    assert not any(aux.endswith(":up") for aux in op.auxiliary_edges)


def test_swiglu_persists_projection_grads_on_full_phase() -> None:
    from zepto.analysis.memory.simulator import ResourceEventSimulator

    hidden, intermediate = 8, 16
    graph = _swiglu_graph(
        seq_len=4, hidden_size=hidden, intermediate_size=intermediate
    )
    lowered = lower(graph, reference_invocation(phase="full"))
    op = lowered.nodes[0]
    persist = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.PERSIST]
    assert {ev.value.split(":")[-1] for ev in persist} == {
        "grad_gate",
        "grad_up",
        "grad_down",
    }
    result = ResourceEventSimulator(lowered).run()
    assert result.breakdown.weight_grads == 3 * hidden * intermediate * 4


def test_swiglu_forward_skips_projection_persist_grads() -> None:
    graph = _swiglu_graph(seq_len=4, hidden_size=8, intermediate_size=16)
    lowered = lower(graph, reference_invocation())
    persist = [
        ev
        for ev in lowered.nodes[0].resource_events
        if ev.kind is ResourceEventKind.PERSIST
    ]
    assert persist == []


def test_swiglu_no_backward_without_grad() -> None:
    graph = _swiglu_graph(
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


def test_swiglu_paper_flops_helper() -> None:
    seq_len, hidden_size, intermediate_size = 8192, 4096, 14336
    recipe = DEFAULT_SWIGLU_RECIPE
    assert recipe.paper_forward_flops(
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ) == 6 * seq_len * hidden_size * intermediate_size


def test_swiglu_does_not_match_ungated_ffn() -> None:
    graph = compose_graph(
        lambda ctx: FFN(hidden_size=8, intermediate_size=16),
        (Tensor(shape=(4, 8), requires_grad=True),),
    )
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, reference_invocation(), registry)
    assert not any(r.kind == "region/swiglu" for r in regions)


def test_swiglu_batched_lower_flops_and_aux_scale() -> None:
    from tests.lowering.regions._batch_helpers import (
        assert_aux_numel_scales,
        assert_flops_scales,
        aux_tensor,
    )

    batch, seq_len, hidden, intermediate = 2, 4, 8, 16
    single = lower(
        _swiglu_graph(seq_len=seq_len, hidden_size=hidden, intermediate_size=intermediate),
        reference_invocation(),
    )
    batched = lower(
        compose_graph(
            lambda ctx: SwiGLU(hidden_size=hidden, intermediate_size=intermediate),
            (Tensor(shape=(batch, seq_len, hidden), requires_grad=True),),
        ),
        reference_invocation(),
    )
    assert len(batched.nodes) == 1
    assert batched.nodes[0].implementation == "region/swiglu/decomposed"
    assert_flops_scales(
        batched.nodes[0].forward_flops, single.nodes[0].forward_flops, batch
    )
    assert aux_tensor(single, single.nodes[0], ":gate").shape == (seq_len, intermediate)
    assert aux_tensor(batched, batched.nodes[0], ":gate").shape == (
        batch,
        seq_len,
        intermediate,
    )
    assert_aux_numel_scales(batched, batched.nodes[0], single, single.nodes[0], batch)
