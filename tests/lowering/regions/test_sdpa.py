"""Discovery, lowering, and variant tests for region/gqa SDPA sub-backends."""

from __future__ import annotations

from types import MappingProxyType

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from modules.attention.attention import FlexibleAttention
from modules.attention.attention_config import llama_gqa
from modules.attention.gqa import GroupedQueryAttention
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


def _sdpa_context(*, sdpa_mode: str = "flash", **kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "sdpa"}),
        "attention_backend": "sdpa",
        "state": (("sdpa_mode", sdpa_mode),),
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


def _expected_flash_forward_flops() -> int:
    h, s, dh = _HEADS, _SEQ, _HEAD_DIM
    return 4 * h * s * s * dh + 5 * h * s * s


def _expected_math_forward_flops() -> int:
    h, s, dh = _HEADS, _SEQ, _HEAD_DIM
    return 4 * h * s * s * dh + 6 * h * s * s


def _gqa_regions(graph, ctx, registry):
    return [r for r in discover_regions(graph, ctx, registry) if r.kind == "region/gqa"]


def _gqa_node(lowered):
    nodes = [n for n in lowered.nodes if n.implementation.startswith("region/gqa")]
    assert len(nodes) == 1
    return nodes[0]


def test_flexible_llama_sdpa_flash_flops() -> None:
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(
            llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)
        ),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    lowered = lower(graph, _sdpa_context(sdpa_mode="flash"))
    op = _gqa_node(lowered)
    assert op.implementation == "region/gqa/sdpa-flash"
    expected = _expected_flash_forward_flops()
    assert op.forward_flops == expected


def test_compose_gqa_discovers_sdpa_region() -> None:
    graph = _compose_gqa()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = _gqa_regions(graph, _sdpa_context(), registry)
    assert len(regions) >= 1
    region = regions[0]
    assert len(region.operation_ids) == 10
    families = tuple(
        graph.node(op_id).operation_family for op_id in region.operation_ids
    )
    assert families == _ATTENTION_CORE


def test_sdpa_unfused_without_sdpa_capability() -> None:
    graph = _compose_gqa()
    lowered = lower(
        graph,
        reference_invocation(
            requested_capabilities=frozenset({"fused"}),
            attention_backend="sdpa",
        ),
    )
    sdpa_nodes = [
        n for n in lowered.nodes if n.implementation.startswith("region/gqa/sdpa")
    ]
    assert len(sdpa_nodes) == 0


def test_sdpa_flash_flops_and_row_stats() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _sdpa_context(sdpa_mode="flash"))
    op = _gqa_node(lowered)
    assert op.implementation == "region/gqa/sdpa-flash"
    expected = _expected_flash_forward_flops()
    assert op.forward_flops == expected
    assert op.backward_flops == 2 * expected
    allocs = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.ALLOCATE]
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(allocs) == 1
    assert len(saves) == 1
    assert any("row_stats" in aux for aux in op.auxiliary_edges)


def test_sdpa_math_flops_and_save_p() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _sdpa_context(sdpa_mode="math"))
    op = _gqa_node(lowered)
    assert op.implementation == "region/gqa/sdpa-math"
    expected = _expected_math_forward_flops()
    assert op.forward_flops == expected
    saves = [ev for ev in op.resource_events if ev.kind is ResourceEventKind.SAVE]
    assert len(saves) == 1
    assert any(":P" in aux for aux in op.auxiliary_edges)


def test_sdpa_mem_efficient_selected_by_mode() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _sdpa_context(sdpa_mode="mem_efficient"))
    assert _gqa_node(lowered).implementation == "region/gqa/sdpa-mem-efficient"


def test_sdpa_default_mode_is_flash() -> None:
    graph = _compose_gqa()
    ctx = reference_invocation(
        requested_capabilities=frozenset({"fused", "sdpa"}),
        attention_backend="sdpa",
    )
    lowered = lower(graph, ctx)
    assert _gqa_node(lowered).implementation == "region/gqa/sdpa-flash"


def test_sdpa_pin_math() -> None:
    graph = _compose_gqa()
    ctx = _sdpa_context(
        sdpa_mode="flash",
        region_implementation_pins=MappingProxyType(
            {"region/gqa": "region/gqa/sdpa-math"}
        ),
        state=(("sdpa_mode", "math"),),
    )
    lowered = lower(graph, ctx)
    assert _gqa_node(lowered).implementation == "region/gqa/sdpa-math"


def test_sdpa_flash2_not_selected_when_sdpa_backend() -> None:
    graph = _compose_gqa()
    ctx = _sdpa_context(hardware="cuda")
    lowered = lower(graph, ctx)
    impl = _gqa_node(lowered).implementation
    assert impl.startswith("region/gqa/sdpa")
    assert impl not in ("region/gqa/flash2", "region/gqa/flash3")


def test_sdpa_fusion_map_absorbs_attention_core() -> None:
    graph = _compose_gqa()
    lowered = lower(graph, _sdpa_context())
    region = _gqa_node(lowered)
    fused_ops = [
        op_id
        for op_id, region_node_id in lowered.fusion_map.items()
        if region_node_id == region.id
    ]
    assert len(fused_ops) == 10


def test_sdpa_flash_fused_vs_unfused_vram_delta() -> None:
    graph = _compose_gqa()
    unfused = lower(graph, reference_invocation())
    fused = lower(graph, _sdpa_context(sdpa_mode="flash"))

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
        if node.implementation.startswith("region/gqa/sdpa")
        for ev in node.resource_events
        if ev.kind is ResourceEventKind.ALLOCATE
    )
    assert unfused_core_allocs >= 8
    assert fused_core_allocs == 1
    assert unfused_core_allocs - fused_core_allocs >= 5
