"""Sliding-window GQA region discovery, FLOPs, and mask elision tests."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.gqa import effective_attention_pairs, gqa_forward_flops
from zepto.compose import Module, Tensor, compose_graph
from modules.attention.attention import FlexibleAttention
from modules.attention.attention_config import llama_gqa
from modules.attention.materialized_causal_mask import MaterializedCausalMask
from modules.attention.sliding_window_causal_mask import SlidingWindowCausalMask
from zepto.semantic import ResourceEventKind

_SEQ = 8
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_HIDDEN = _HEADS * _HEAD_DIM
_WINDOW = 4

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


class _SlidingWindowLlamaAttention(Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FlexibleAttention(
            llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)
        )
        self.mask = SlidingWindowCausalMask(_SEQ, _WINDOW)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.attn(hidden_states, self.mask())  # type: ignore[return-value]


class _CausalMaskLlamaAttention(Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FlexibleAttention(
            llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)
        )
        self.mask = MaterializedCausalMask(_SEQ)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.attn(hidden_states, self.mask())  # type: ignore[return-value]


def _compose_sliding():
    return compose_graph(
        lambda _ctx: _SlidingWindowLlamaAttention(),
        (Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),),
    )


def _compose_full_llama():
    return compose_graph(
        lambda _ctx: FlexibleAttention(
            llama_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM)
        ),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )


def _compose_causal_mask_module():
    return compose_graph(
        lambda _ctx: _CausalMaskLlamaAttention(),
        (Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),),
    )


def _gqa_regions(graph, ctx, registry):
    return [r for r in discover_regions(graph, ctx, registry) if r.kind == "region/gqa"]


def _gqa_node(lowered):
    nodes = [n for n in lowered.nodes if n.implementation.startswith("region/gqa/")]
    assert len(nodes) == 1
    return nodes[0]


def _mask_nodes(lowered, family: str):
    return [
        n
        for n in lowered.nodes
        if n.implementation == f"{family}/identity"
    ]


def test_sliding_window_discovers_gqa_region() -> None:
    graph = _compose_sliding()
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


def test_sliding_window_flash_flops() -> None:
    graph = _compose_sliding()
    lowered = lower(graph, _flash_context())
    op = _gqa_node(lowered)
    expected = gqa_forward_flops(
        num_heads=_HEADS,
        seq_len=_SEQ,
        head_dim=_HEAD_DIM,
        window_size=_WINDOW,
    )
    assert op.forward_flops == expected


def test_sliding_window_full_seq_equiv() -> None:
    pairs_window = effective_attention_pairs(_SEQ, _WINDOW)
    pairs_full = effective_attention_pairs(_SEQ, _SEQ)
    assert pairs_window < pairs_full
    pairs_w_eq_s = effective_attention_pairs(_SEQ, _SEQ)
    pairs_none = effective_attention_pairs(_SEQ, None)
    assert pairs_w_eq_s == pairs_none == _SEQ * _SEQ


def test_sliding_window_regression_llama_full() -> None:
    graph = _compose_full_llama()
    lowered = lower(graph, _flash_context())
    op = _gqa_node(lowered)
    h, s, dh = _HEADS, _SEQ, _HEAD_DIM
    expected = 4 * h * s * s * dh + 5 * h * s * s
    assert op.forward_flops == expected


def test_flash_elides_sliding_mask_vram() -> None:
    graph = _compose_sliding()
    lowered = lower(graph, _flash_context())
    mask_nodes = _mask_nodes(
        lowered, "materialized_sliding_window_causal_mask"
    )
    assert len(mask_nodes) == 1
    persists = [
        ev
        for n in mask_nodes
        for ev in n.resource_events
        if ev.kind is ResourceEventKind.PERSIST
    ]
    assert len(persists) == 0


def test_eager_retains_sliding_mask_vram() -> None:
    graph = _compose_sliding()
    lowered = lower(graph, reference_invocation())
    mask_nodes = _mask_nodes(
        lowered, "materialized_sliding_window_causal_mask"
    )
    assert len(mask_nodes) == 1
    persists = [
        ev
        for n in mask_nodes
        for ev in n.resource_events
        if ev.kind is ResourceEventKind.PERSIST
    ]
    assert len(persists) == 1


def test_flash_elides_full_causal_mask_vram() -> None:
    graph = _compose_causal_mask_module()
    lowered = lower(graph, _flash_context())
    mask_nodes = _mask_nodes(lowered, "materialized_causal_mask")
    assert len(mask_nodes) == 1
    persists = [
        ev
        for n in mask_nodes
        for ev in n.resource_events
        if ev.kind is ResourceEventKind.PERSIST
    ]
    assert len(persists) == 0


def test_sdpa_math_retains_mask_vram() -> None:
    graph = _compose_sliding()
    ctx = reference_invocation(
        requested_capabilities=frozenset({"fused", "sdpa"}),
        attention_backend="sdpa",
        state=(("sdpa_mode", "math"),),
    )
    lowered = lower(graph, ctx)
    mask_nodes = _mask_nodes(
        lowered, "materialized_sliding_window_causal_mask"
    )
    assert len(mask_nodes) == 1
    persists = [
        ev
        for n in mask_nodes
        for ev in n.resource_events
        if ev.kind is ResourceEventKind.PERSIST
    ]
    assert len(persists) == 1
