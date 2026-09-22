"""Batched Granite attention lowering tests."""

from __future__ import annotations

from zepto.analysis import discover_regions, lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_defaults
from zepto.compose import Tensor, compose_graph
from zepto.modules.attention.attention import FlexibleAttention
from zepto.modules.attention.attention_config import granite_attention_layer

from tests.lowering.regions._batch_helpers import (
    assert_aux_numel_scales,
    assert_flops_scales,
    aux_tensor,
)

_SEQ = 8
_BATCH = 2
_HIDDEN = 4096


def _flash_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _sdpa_math_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "sdpa"}),
        "attention_backend": "sdpa",
        "state": (("sdpa_mode", "math"),),
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


def _gqa_node(lowered):
    return [n for n in lowered.nodes if n.implementation.startswith("region/gqa/")][0]


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
    assert_flops_scales(
        _gqa_node(batched).forward_flops, _gqa_node(single).forward_flops, _BATCH
    )


def test_granite_batched_flash_row_stats_scale() -> None:
    single = lower(_compose_granite(batch=1), _flash_context())
    batched = lower(_compose_granite(batch=_BATCH), _flash_context())
    single_gqa = _gqa_node(single)
    batched_gqa = _gqa_node(batched)
    assert aux_tensor(single, single_gqa, "row_stats").shape == (32, _SEQ, 2)
    assert aux_tensor(batched, batched_gqa, "row_stats").shape == (_BATCH, 32, _SEQ, 2)
    assert_aux_numel_scales(batched, batched_gqa, single, single_gqa, _BATCH)


def test_granite_batched_sdpa_save_p_scale() -> None:
    single = lower(_compose_granite(batch=1), _sdpa_math_context())
    batched = lower(_compose_granite(batch=_BATCH), _sdpa_math_context())
    single_gqa = _gqa_node(single)
    batched_gqa = _gqa_node(batched)
    assert single_gqa.implementation == "region/gqa/sdpa-math"
    assert_flops_scales(batched_gqa.forward_flops, single_gqa.forward_flops, _BATCH)
    assert aux_tensor(single, single_gqa, ":P").shape == (32, _SEQ, _SEQ)
    assert aux_tensor(batched, batched_gqa, ":P").shape == (_BATCH, 32, _SEQ, _SEQ)
    assert_aux_numel_scales(batched, batched_gqa, single, single_gqa, _BATCH)
