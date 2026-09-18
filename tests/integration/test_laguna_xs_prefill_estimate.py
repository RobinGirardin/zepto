"""Laguna XS prefill integration estimate."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.models.laguna_xs import LagunaXs, LagunaXsConfig
from zepto.semantic.metadata import DType


def test_laguna_xs_prefill_estimate_produces_cost_report() -> None:
    seq_len = 512
    graph = compose_graph(
        lambda _ctx: LagunaXs(
            config=LagunaXsConfig(
                hidden_size=2048,
                dense_swiglu_intermediate=8192,
                num_layers=2,
                vocab_size=512,
            ),
            seq_len=seq_len,
        ),
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
