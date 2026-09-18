"""Phase 1b FLOP accounting tests."""

from __future__ import annotations

from zepto.analysis import account_flops, lower, reference_invocation
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


def test_account_flops_matches_node_sums() -> None:
    graph = _maximum_graph()
    lowered = lower(graph, reference_invocation(phase="forward"))
    report = account_flops(lowered)

    forward = sum(node.forward_flops for node in lowered.nodes)
    backward = sum(node.backward_flops for node in lowered.nodes)

    assert report.forward_flops == forward
    assert report.backward_flops == backward
    assert report.total_flops == forward


def test_account_flops_forward_phase() -> None:
    graph = _maximum_graph()
    lowered = lower(graph, reference_invocation(phase="forward"))
    report = account_flops(lowered)

    forward = sum(node.forward_flops for node in lowered.nodes)
    assert report.forward_flops == 0
    assert report.total_flops == 0
    assert report.backward_flops == sum(node.backward_flops for node in lowered.nodes)


def test_account_flops_backward_phase() -> None:
    graph = _maximum_graph()
    lowered = lower(graph, reference_invocation(phase="backward"))
    report = account_flops(lowered)

    backward = sum(node.backward_flops for node in lowered.nodes)
    assert report.total_flops == backward


def test_account_flops_by_implementation() -> None:
    graph = _maximum_graph()
    lowered = lower(graph, reference_invocation())
    report = account_flops(lowered)

    assert len(report.by_implementation) == 1
    assert report.by_implementation[0].key == "maximum/identity"
    assert report.by_implementation[0].forward_flops == report.forward_flops
