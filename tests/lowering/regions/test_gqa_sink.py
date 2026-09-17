"""Discovery, lowering, and mask elision tests for region/gqa-sink."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.gqa_sink import GQASinkRecipe
from zepto.compose import Module, Tensor, compose_graph
from modules.attention.attention import FlexibleAttention
from modules.attention.attention_config import AttentionConfig, gpt_oss_attention_layer
from modules.attention.materialized_causal_mask import MaterializedCausalMask
from modules.attention.sliding_window_causal_mask import SlidingWindowCausalMask
from zepto.semantic import ResourceEventKind

_SEQ = 8
_WINDOW = 4
_HIDDEN = 32
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8

_SINK_CORE = (
    "repeat_kv",
    "repeat_kv",
    "transpose",
    "matmul",
    "add",
    "attention_softmax_with_sink",
    "matmul",
)


def _flash_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _test_sink_config(*, mask: str = "sliding") -> AttentionConfig:
    return AttentionConfig(
        hidden_size=_HIDDEN,
        num_q_heads=_HEADS,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        mask=mask,  # type: ignore[arg-type]
        window_size=_WINDOW if mask == "sliding" else None,
        softmax="sink",
        qkv_bias=True,
    )


class _SinkSlidingAttention(Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FlexibleAttention(_test_sink_config(mask="sliding"))
        self.mask = SlidingWindowCausalMask(_SEQ, _WINDOW)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.attn(hidden_states, self.mask())  # type: ignore[return-value]


class _SinkFullAttention(Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FlexibleAttention(_test_sink_config(mask="full"))
        self.mask = MaterializedCausalMask(_SEQ)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.attn(hidden_states, self.mask())  # type: ignore[return-value]


def _compose_sliding_sink():
    return compose_graph(
        lambda _ctx: _SinkSlidingAttention(),
        (Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),),
    )


def _compose_full_sink():
    return compose_graph(
        lambda _ctx: _SinkFullAttention(),
        (Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),),
    )


def _sink_regions(graph, ctx, registry):
    return [
        r for r in discover_regions(graph, ctx, registry) if r.kind == "region/gqa-sink"
    ]


def _sink_node(lowered):
    nodes = [
        n for n in lowered.nodes if n.implementation.startswith("region/gqa-sink/")
    ]
    assert len(nodes) == 1
    return nodes[0]


def test_sink_discovers_region() -> None:
    graph = _compose_sliding_sink()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = _sink_regions(graph, _flash_context(), registry)
    assert len(regions) >= 1
    region = regions[0]
    assert len(region.operation_ids) == 7
    families = tuple(
        graph.node(op_id).operation_family for op_id in region.operation_ids
    )
    assert families == _SINK_CORE


def test_sink_flash_flops() -> None:
    graph = _compose_sliding_sink()
    lowered = lower(graph, _flash_context())
    op = _sink_node(lowered)
    recipe = GQASinkRecipe(window_size=_WINDOW)
    expected = recipe.forward_flops(
        num_heads=_HEADS, seq_len=_SEQ, head_dim=_HEAD_DIM
    )
    assert op.forward_flops == expected


def test_sink_no_standard_gqa_match() -> None:
    graph = _compose_sliding_sink()
    lowered = lower(graph, _flash_context())
    gqa_nodes = [
        n for n in lowered.nodes if n.implementation.startswith("region/gqa/")
        and not n.implementation.startswith("region/gqa-sink/")
    ]
    assert len(gqa_nodes) == 0


def test_sink_fusion_map() -> None:
    graph = _compose_sliding_sink()
    lowered = lower(graph, _flash_context())
    region = _sink_node(lowered)
    fused_ops = [
        op_id
        for op_id, region_node_id in lowered.fusion_map.items()
        if region_node_id == region.id
    ]
    assert len(fused_ops) == 7


def test_sink_flash_elides_mask() -> None:
    graph = _compose_sliding_sink()
    lowered = lower(graph, _flash_context())
    mask_nodes = [
        n
        for n in lowered.nodes
        if n.implementation == "materialized_sliding_window_causal_mask/identity"
    ]
    assert len(mask_nodes) == 1
    persists = [
        ev
        for n in mask_nodes
        for ev in n.resource_events
        if ev.kind is ResourceEventKind.PERSIST
    ]
    assert len(persists) == 0


def test_gpt_oss_even_layer_preset() -> None:
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(gpt_oss_attention_layer(layer_index=0)),
        (
            Tensor(shape=(_SEQ, 2880), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    lowered = lower(graph, _flash_context())
    assert _sink_node(lowered).implementation == "region/gqa-sink/flash2"


def test_gpt_oss_odd_layer_full_mask() -> None:
    graph = _compose_full_sink()
    lowered = lower(graph, _flash_context())
    assert _sink_node(lowered).implementation == "region/gqa-sink/flash2"
