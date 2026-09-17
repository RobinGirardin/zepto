"""Phase 1b memory accounting tests."""

from __future__ import annotations

from zepto.analysis import account_memory, lower, reference_invocation
from zepto.analysis.memory import ResourceEventSimulator
from zepto.compose import Tensor, compose_graph
from zepto.compose.values import Tensor as ComposeTensor
from zepto.graph import GraphBuilder, Provenance
from modules.layers.linear import Linear
from zepto.semantic import Identity, MatMul, Port


def _identity_graph():
    builder = GraphBuilder()
    value = builder.add_input(Tensor(shape=(1024,)))
    provenance = Provenance((), "Fixture", "identity", 0)
    output = builder.add_operation(
        operation_family="identity",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(value,),
        output_tensors=(Tensor(shape=(1024,)),),
        provenance=provenance,
        operation=Identity(),
    )[0]
    builder.mark_output(output)
    return builder.build()


def test_account_memory_matches_simulator() -> None:
    graph = _identity_graph()
    ctx = reference_invocation(phase="forward")
    lowered = lower(graph, ctx)

    report = account_memory(lowered)
    result = ResourceEventSimulator(lowered).run()

    assert report.sum_all_bytes == result.sum_all_bytes
    assert report.peak_live_bytes == result.peak_live_bytes
    assert report.breakdown == result.breakdown
    assert report.by_module == result.by_module
    assert report.by_region == result.by_region
    assert report.by_implementation == result.by_implementation
    assert report.live_timeline == result.live_timeline


def test_account_memory_linear_includes_parameters() -> None:
    graph = compose_graph(
        lambda ctx: Linear(64, 32),
        (ComposeTensor(shape=(2, 64)),),
    )
    lowered = lower(graph, reference_invocation(phase="forward"))
    report = account_memory(lowered)

    assert report.breakdown.parameters > 0
    assert report.peak_live_bytes >= report.breakdown.parameters


def test_account_memory_forward_phase_skips_gradients() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(4, 8), requires_grad=True))
    right = builder.add_input(Tensor(shape=(8, 16), requires_grad=True))
    output = builder.add_operation(
        operation_family="matmul",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, right),
        output_tensors=(Tensor(shape=(4, 16), requires_grad=True),),
        provenance=Provenance((), "Fixture", "matmul", 0),
        operation=MatMul(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    report = account_memory(lower(graph, reference_invocation(phase="forward")))
    assert report.breakdown.gradients == 0
    assert report.breakdown.weight_grads == 0
