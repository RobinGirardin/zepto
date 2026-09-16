"""Phase 1b estimate() facade tests."""

from __future__ import annotations

from zepto.analysis import (
    account_flops,
    account_memory,
    estimate,
    lower,
    reference_invocation,
)
from zepto.compose import Tensor
from zepto.graph import GraphBuilder, Provenance
from zepto.semantic import Identity, Maximum, Port


def _maximum_graph():
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(3, 4), requires_grad=True))
    right = builder.add_input(Tensor(shape=(3, 4)))
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, right),
        output_tensors=(Tensor(shape=(3, 4)),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    return builder.build()


def test_estimate_matches_explicit_chain() -> None:
    graph = _maximum_graph()
    ctx = reference_invocation()

    report = estimate(graph, ctx)
    lowered = lower(graph, ctx)
    expected_mem = account_memory(lowered)
    expected_flops = account_flops(lowered)

    assert report.memory == expected_mem
    assert report.flops == expected_flops
    assert report.context is ctx


def test_estimate_return_lowered() -> None:
    graph = _maximum_graph()
    ctx = reference_invocation()

    report, lowered = estimate(graph, ctx, return_lowered=True)

    assert report.memory.peak_live_bytes >= 0
    assert len(lowered.nodes) == 1
    assert report.flops.total_flops == 0


def test_estimate_identity_graph() -> None:
    builder = GraphBuilder()
    value = builder.add_input(Tensor(shape=(4,)))
    provenance = Provenance((), "Fixture", "identity", 0)
    output = builder.add_operation(
        operation_family="identity",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(value,),
        output_tensors=(Tensor(shape=(4,)),),
        provenance=provenance,
        operation=Identity(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    report = estimate(graph, reference_invocation(phase="forward"))
    assert report.flops.forward_flops == 0
    assert report.memory.peak_live_bytes > 0
