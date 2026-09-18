"""Granite prefill integration estimate."""

from __future__ import annotations

from tests.modules.test_granite_compose import _flash_ctx, _tiny_granite
from zepto.analysis import estimate
from zepto.compose import Tensor, compose_graph


def test_granite_prefill_estimate_produces_cost_report() -> None:
    seq_len = 64
    graph = compose_graph(
        lambda _ctx: _tiny_granite(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    report = estimate(graph, _flash_ctx())
    assert report.flops.total_flops > 0
    assert report.memory.peak_live_bytes > 0
    assert report.memory.breakdown.parameters > 0
