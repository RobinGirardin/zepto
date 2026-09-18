"""Batched Granite attention lowering tests."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.attention.attention import FlexibleAttention
from zepto.modules.attention.attention_config import granite_attention_layer

_SEQ = 8
_BATCH = 2
_HIDDEN = 4096


def _flash_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _compose_granite(*, batch: int):
    return compose_graph(
        lambda _ctx: FlexibleAttention(granite_attention_layer()),
        (
            Tensor(shape=(batch, _SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )


def test_granite_batched_discovers_gqa() -> None:
    graph = _compose_granite(batch=_BATCH)
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = [
        r for r in discover_regions(graph, _flash_context(), registry) if r.kind == "region/gqa"
    ]
    assert len(regions) >= 1
    assert len(regions[0].operation_ids) == 10


def test_granite_batched_flops_scale_with_batch() -> None:
    single = lower(_compose_granite(batch=1), _flash_context())
    batched = lower(_compose_granite(batch=_BATCH), _flash_context())
    single_gqa = [
        n for n in single.nodes if n.implementation.startswith("region/gqa/")
    ][0]
    batched_gqa = [
        n for n in batched.nodes if n.implementation.startswith("region/gqa/")
    ][0]
    assert batched_gqa.forward_flops == _BATCH * single_gqa.forward_flops
