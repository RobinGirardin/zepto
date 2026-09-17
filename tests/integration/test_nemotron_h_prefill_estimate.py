"""Nemotron H prefill integration estimate."""

from __future__ import annotations

from tests.modules.test_nemotron_h_compose import _tiny_nemotron
from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.semantic.metadata import DType


def test_nemotron_h_prefill_estimate_produces_cost_report() -> None:
    seq_len = 16
    graph = compose_graph(
        lambda _ctx: _tiny_nemotron(seq_len),
        (Tensor(shape=(seq_len,)),),
    )
    report = estimate(
        graph,
        reference_invocation(
            requested_capabilities=frozenset({"fused", "flash"}),
            default_dtype=DType.FP16,
        ),
    )
    assert report.flops.total_flops > 0
    assert report.memory.breakdown.parameters > 0
