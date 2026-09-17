"""RMSNorm centered variant compose tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.rms_norm import RMSNorm


def test_centered_rmsnorm_includes_subtract_in_graph() -> None:
    graph = compose_graph(
        lambda _ctx: RMSNorm(64, center=True),
        (Tensor(shape=(8, 64), requires_grad=True),),
    )
    plain = compose_graph(
        lambda _ctx: RMSNorm(64, center=False),
        (Tensor(shape=(8, 64), requires_grad=True),),
    )
    assert len(graph.nodes) > len(plain.nodes)
