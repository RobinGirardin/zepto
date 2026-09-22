"""Discovery, lowering, and variant tests for region/gqa (FlashAttention)."""

from __future__ import annotations

from types import MappingProxyType

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.attention.attention import FlexibleAttention
from zepto.modules.attention.attention_config import llama_gqa
from zepto.modules.attention.gqa import GroupedQueryAttention
from zepto.semantic import ResourceEventKind

_SEQ = 8
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_HIDDEN = _HEADS * _HEAD_DIM

_ATTENTION_CORE = (
    "repeat_kv",
    "repeat_kv",
    "transpose",
    "matmul",
    "add",
    "multiply",
    "exp",
    "reduce_sum",
    "divide",
    "matmul",
)


def _flash_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _compose_gqa():
    return compose_graph(
        lambda _ctx: GroupedQueryAttention(
            _HEADS * _HEAD_DIM, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM
        ),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )


def _expected_forward_flops() -> int:
    h, s, dh = _HEADS, _SEQ, _HEAD_DIM
    return 4 * h * s * s * dh + 5 * h * s * s


def _gqa_regions(graph, ctx, registry):
    return [r for r in discover_regions(graph, ctx, registry) if r.kind == "region/gqa"]


def _gqa_node(lowered):
    nodes = [n for n in lowered.nodes if n.implementation.startswith("region/gqa")]
    assert len(nodes) == 1
    return nodes[0]


def _compose_flexible_llama():
    return compose_graph(
        lambda _ctx: FlexibleAttention(
            llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)
        ),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )


def test_flexible_llama_discovers_flash_region() -> None:
    graph = _compose_flexible_llama()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = _gqa_regions(graph, _flash_context(), registry)
    assert len(regions) >= 1
    region = regions[0]
    assert len(region.operation_ids) == 10
    families = tuple(
        graph.node(op_id).operation_family for op_id in region.operation_ids
    )
    assert families == _ATTENTION_CORE


def test_flexible_llama_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_flexible_llama()
    lowered = lower(graph, _flash_context())
    op = _gqa_node(lowered)
    assert op.implementation == "region/gqa/flash2"
    expected = _expected_forward_flops()
    assert op.forward_flops == expected
    assert op.backward_flops == 2 * expected
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(allocs) == 1
    assert len(saves) == 1
    assert any("row_stats" in aux for aux in op.auxiliary_edges)


def test_compose_gqa_discovers_flash_region() -> None:
    graph = _compose_gqa()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = _gqa_regions(graph, _flash_context(), registry)
    assert len(regions) >= 1
    region = regions[0]
    assert len(region.operation_ids) == 10
    families = tuple(
        graph.node(op_id).operation_family for op_id in region.operation_ids
    )
    assert families == _ATTENTION_CORE


def test_gqa_unfused_without_flash_capability() -> None:
    graph = _compose_gqa()
    lowered = lower(
        graph, reference_invocation(requested_capabilities=frozenset({"fused"}))
    )
    gqa_nodes = [
        n for n in lowered.nodes if n.implementation.startswith("region/gqa")
    ]
    assert len(gqa_nodes) == 0
    masked = [
        n
        for n in lowered.nodes
        if n.implementation.startswith("region/masked_softmax")
    ]
    assert len(masked) == 1


def test_gqa_fused_lowering_flops_and_allocs() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _flash_context())
    op = _gqa_node(lowered)
    assert op.implementation == "region/gqa/flash2"
    expected = _expected_forward_flops()
    assert op.forward_flops == expected
    assert op.backward_flops == 2 * expected
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(allocs) == 1
    assert len(saves) == 1
    assert any("row_stats" in aux for aux in op.auxiliary_edges)


def test_gqa_fusion_map_absorbs_attention_core() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _flash_context())
    region = _gqa_node(lowered)
    fused_ops = [
        op_id
        for op_id, region_node_id in lowered.fusion_map.items()
        if region_node_id == region.id
    ]
    assert len(fused_ops) == 10


def test_gqa_pin_flash2() -> None:
    graph = _compose_gqa()
    ctx = _flash_context(
        hardware="cuda",
        region_implementation_pins=MappingProxyType(
            {"region/gqa": "region/gqa/flash2"}
        ),
    )
    lowered = lower(graph, ctx)
    assert _gqa_node(lowered).implementation == "region/gqa/flash2"


def test_gqa_flash3_wins_on_cuda_hardware() -> None:
    graph = _compose_gqa()
    ctx = _flash_context(hardware="cuda")
    lowered = lower(graph, ctx)
    assert _gqa_node(lowered).implementation == "region/gqa/flash3"
    assert lowered.region_selections[0].chosen.id == "region/gqa/flash3"


def test_gqa_flash2_wins_on_non_cuda_hardware() -> None:
    graph = _compose_gqa()
    ctx = _flash_context(hardware="mps")
    lowered = lower(graph, ctx)
    assert _gqa_node(lowered).implementation == "region/gqa/flash2"


def test_gqa_prefill_kv_template_allocates_state() -> None:
    from zepto.analysis.horizon.state import StatePortRegistry
    from zepto.semantic.metadata import DType

    graph = _compose_gqa()
    registry = StatePortRegistry.empty().configure_kv(
        num_layers=1,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        dtype=DType.FP32,
    )
    lowered = lower(graph, _flash_context(), state_ports=registry)
    gqa = _gqa_node(lowered)

    kv_allocs = [
        ev
        for ev in gqa.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
        and any("kv_cache" in aux for aux in gqa.auxiliary_edges)
    ]
    assert len(lowered.state_port_events) == 1
    assert len(kv_allocs) >= 1
    assert "kv_cache" in gqa.auxiliary_edges[0]


def test_gqa_backward_does_not_release_kv_cache() -> None:
    from zepto.analysis.horizon.state import StatePortRegistry
    from zepto.semantic.metadata import DType

    graph = _compose_gqa()
    registry = StatePortRegistry.empty().configure_kv(
        num_layers=1,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        dtype=DType.FP32,
    )
    lowered = lower(
        graph, _flash_context(phase="backward"), state_ports=registry
    )
    gqa = _gqa_node(lowered)
    assert any("kv_cache" in aux for aux in gqa.auxiliary_edges)
    released = [
        ev.value
        for ev in gqa.resource_events
        if ev.kind is ResourceEventKind.RELEASE
    ]
    assert released
    assert not any("kv_cache" in aux_id for aux_id in released)
    assert any("row_stats" in aux_id for aux_id in released)


def test_gqa_fused_vs_unfused_vram_delta() -> None:
    graph = _compose_gqa()
    unfused = lower(graph, reference_invocation())
    fused = lower(graph, _flash_context())

    unfused_core_allocs = sum(
        1
        for node in unfused.nodes
        if node.node_id
        and graph.node(node.node_id).operation_family in _ATTENTION_CORE
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    fused_core_allocs = sum(
        1
        for node in fused.nodes
        if node.implementation.startswith("region/gqa")
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    assert unfused_core_allocs >= 8
    assert fused_core_allocs == 1
    assert unfused_core_allocs - fused_core_allocs >= 5
