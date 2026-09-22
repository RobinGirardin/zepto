"""Discovery and lowering tests for region/masked_softmax/sink (boundary B)."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.plan import resolve_overlaps
from zepto.analysis.lowering.registry import region_has_compatible_implementation
from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.attention.attention_softmax_with_sink import AttentionSoftmaxWithSink
from zepto.semantic import Add, Multiply

_HEADS = 4
_KV_HEADS = 2
_SEQ = 8
_HEAD_DIM = 8
_HIDDEN = _HEADS * _HEAD_DIM
_NUMEL = _HEADS * _SEQ * _SEQ

_SINK_SCORE_PATH = ("add", "multiply", "attention_softmax_with_sink")


def _fused_context(**kwargs):
    defaults: dict = {"requested_capabilities": frozenset({"fused"})}
    defaults.update(kwargs)
    return reference_invocation(**defaults)


class _MaskedSoftmaxSinkScore(Module):
    """Stub score path: scale → mask add → sink softmax."""

    module_kind = "MaskedSoftmaxSinkScore"

    def __init__(self) -> None:
        super().__init__()
        self._scale = Tensor(
            shape=(1,), semantic_type="softmax_scale", requires_grad=False
        )
        self.softmax = AttentionSoftmaxWithSink(_HEADS)

    def forward(self, scores: Tensor, mask: Tensor) -> Tensor:
        scaled = Multiply()(scores, self._scale)
        masked = Add()(scaled, mask)
        return self.softmax(masked)  # type: ignore[return-value]


def _compose_masked_softmax_sink():
    return compose_graph(
        lambda _ctx: _MaskedSoftmaxSinkScore(),
        (
            Tensor(shape=(_HEADS, _SEQ, _SEQ), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )


def _compose_gqa_sink():
    from tests.lowering.regions.test_gqa_sink import _compose_sliding_sink

    return _compose_sliding_sink()


def _sink_regions(graph, ctx, registry):
    return [
        r
        for r in discover_regions(graph, ctx, registry)
        if r.kind == "region/masked_softmax"
        and tuple(
            graph.node(op_id).operation_family for op_id in r.operation_ids
        )
        in (
            _SINK_SCORE_PATH,
            ("add", "attention_softmax_with_sink"),
        )
    ]


def _sink_node(lowered):
    nodes = [
        n for n in lowered.nodes if n.implementation == "region/masked_softmax/sink"
    ]
    assert len(nodes) == 1
    return nodes[0]


def test_masked_softmax_sink_discovers_region() -> None:
    graph = _compose_masked_softmax_sink()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = _sink_regions(graph, _fused_context(hardware="cuda"), registry)
    assert len(regions) == 1
    families = tuple(
        graph.node(op_id).operation_family for op_id in regions[0].operation_ids
    )
    assert families == _SINK_SCORE_PATH


def test_masked_softmax_sink_flops() -> None:
    graph = _compose_masked_softmax_sink()
    lowered = lower(graph, _fused_context(hardware="cuda"))
    op = _sink_node(lowered)
    expected_fwd = 7 * _NUMEL + 5 * _HEADS * _SEQ
    assert op.forward_flops == expected_fwd
    assert op.backward_flops == 4 * _HEADS * _SEQ * (_SEQ + 1)


def test_masked_softmax_sink_cuda_gate() -> None:
    graph = _compose_masked_softmax_sink()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = _fused_context(hardware="mps")
    regions = _sink_regions(graph, ctx, registry)
    assert len(regions) == 1
    assert not region_has_compatible_implementation(
        regions[0], registry, ctx, graph=graph
    )


def test_gqa_sink_excludes_softmax_one_on_full_graph() -> None:
    graph = _compose_gqa_sink()
    registry = LoweringRegistry()
    register_defaults(registry)
    ctx = reference_invocation(
        requested_capabilities=frozenset({"fused", "flash"}),
        hardware="cuda",
    )
    regions = discover_regions(graph, ctx, registry)
    winners = resolve_overlaps(graph, regions, ctx, registry)
    winner_kinds = {r.kind for r in winners}
    assert "region/gqa-sink" in winner_kinds
    assert "region/softmax-one" not in winner_kinds
    lowered = lower(graph, ctx)
    impl_ids = {n.implementation for n in lowered.nodes}
    assert any(i.startswith("region/gqa-sink/") for i in impl_ids)
    assert not any(i.startswith("region/softmax-one/") for i in impl_ids)
    assert "region/masked_softmax/sink" not in impl_ids
