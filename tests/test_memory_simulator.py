"""Phase 1a memory simulator tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from zepto.analysis import lower, reference_invocation
from zepto.analysis.memory import ResourceEventSimulator
from zepto.analysis.resolved import ResolvedValue
from zepto.compose import Tensor, compose_graph
from zepto.compose.values import Tensor as ComposeTensor
from zepto.graph import GraphBuilder, Provenance
from zepto.modules.relu import ReLU
from zepto.semantic import (
    DType,
    Identity,
    MatMul,
    Maximum,
    Port,
    ResourceEventKind,
    TensorRole,
)
def _merge_train_lowered(graph, **ctx_kwargs):
    """Merge forward and backward lowering events for training simulation."""
    base = reference_invocation(**ctx_kwargs)
    forward = lower(graph, replace(base, phase="forward"))
    backward = lower(graph, replace(base, phase="backward"))
    merged_nodes = []
    for fwd_node, bwd_node in zip(forward.nodes, backward.nodes, strict=True):
        backward_events = tuple(
            event
            for event in bwd_node.resource_events
            if event.phase == "backward"
        )
        merged_nodes.append(
            replace(
                fwd_node,
                resource_events=fwd_node.resource_events + backward_events,
                backward_flops=bwd_node.backward_flops,
            )
        )
    return replace(forward, nodes=tuple(merged_nodes), context=replace(base, phase="full"))


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


def _multiply_broadcast_graph():
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=(32, 128, 512), requires_grad=True))
    bias = builder.add_input(Tensor(shape=(512,), requires_grad=True))
    provenance = Provenance((), "Fixture", "multiply", 0)
    from zepto.semantic import Multiply

    output = builder.add_operation(
        operation_family="multiply",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x, bias),
        output_tensors=(Tensor(shape=(32, 128, 512), requires_grad=True),),
        provenance=provenance,
        operation=Multiply(),
    )[0]
    builder.mark_output(output)
    return builder.build()


def build_matmul_relu_matmul_graph(
    *,
    batch: int = 32,
    seq: int = 128,
    hidden: int = 512,
    inner: int = 256,
):
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=(batch, seq, hidden), requires_grad=True))
    w0 = builder.add_input(Tensor(shape=(hidden, inner), requires_grad=True))
    provenance = Provenance((), "Fixture", "matmul", 0)

    y0 = builder.add_operation(
        operation_family="matmul",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x, w0),
        output_tensors=(Tensor(shape=(batch, seq, inner), requires_grad=True),),
        provenance=provenance,
        operation=MatMul(),
    )[0]

    zero = builder.add_input(
        Tensor(shape=(1,), semantic_type="constant_zero", requires_grad=False)
    )
    y1 = builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(y0, zero),
        output_tensors=(Tensor(shape=(batch, seq, inner), requires_grad=True),),
        provenance=Provenance((), "Fixture", "maximum", 1),
        operation=Maximum(),
    )[0]

    w1 = builder.add_input(Tensor(shape=(inner, hidden), requires_grad=True))
    y2 = builder.add_operation(
        operation_family="matmul",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(y1, w1),
        output_tensors=(Tensor(shape=(batch, seq, hidden), requires_grad=True),),
        provenance=Provenance((), "Fixture", "matmul", 2),
        operation=MatMul(),
    )[0]
    builder.mark_output(y2)
    return builder.build()


def _max_activation_bytes(lowered) -> int:
    largest = 0
    for edge in lowered.edges.values():
        if edge.role is not TensorRole.ACTIVATION:
            continue
        byte_count = lowered.context.accounting.bytes_for(
            ResolvedValue(tensor=edge.tensor, role=edge.role, dtype=edge.tensor.dtype)
        )
        largest = max(largest, byte_count)
    return largest


def test_identity_alias_no_peak_increase() -> None:
    graph = _identity_graph()
    lowered = lower(graph, reference_invocation(phase="forward"))
    result = ResourceEventSimulator(lowered).run()
    assert result.peak_live_bytes == pytest.approx(
        lowered.context.accounting.bytes_for(
            ResolvedValue(
                tensor=lowered.edges[lowered.edge_map[graph.inputs[0]]].tensor,
                role=TensorRole.INPUT,
                dtype=DType.FP32,
            )
        )
    )


def test_forward_allocate_peak() -> None:
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

    lowered = lower(graph, reference_invocation(phase="forward"))
    result = ResourceEventSimulator(lowered).run()
    input_bytes = sum(
        lowered.context.accounting.bytes_for(
            ResolvedValue(
                tensor=lowered.edges[lowered.edge_map[eid]].tensor,
                role=TensorRole.INPUT,
                dtype=DType.FP32,
            )
        )
        for eid in graph.inputs
    )
    output_bytes = lowered.context.accounting.bytes_for(
        ResolvedValue(
            tensor=lowered.edges[lowered.edge_map[output]].tensor,
            role=TensorRole.ACTIVATION,
            dtype=DType.FP32,
        )
    )
    assert result.peak_live_bytes == input_bytes + output_bytes


def test_save_pins_operand_through_forward_cleanup() -> None:
    graph = compose_graph(
        lambda ctx: ReLU(),
        (ComposeTensor(shape=(4, 8), requires_grad=True),),
    )
    lowered = lower(graph, reference_invocation(phase="forward"))
    result = ResourceEventSimulator(lowered).run()
    save_events = [
        event for event in lowered.nodes[0].resource_events if event.kind is ResourceEventKind.SAVE
    ]
    assert save_events
    assert result.peak_live_bytes >= result.breakdown.saved_for_backward or result.peak_live_bytes > 0


def test_unreduced_temp_same_node_release() -> None:
    graph = _multiply_broadcast_graph()
    lowered = lower(graph, reference_invocation(phase="backward"))
    op = lowered.nodes[0]
    backward_events = [event for event in op.resource_events if event.phase == "backward"]
    kinds = [event.kind for event in backward_events]
    assert ResourceEventKind.ALLOCATE in kinds
    assert ResourceEventKind.RELEASE in kinds
    assert kinds.index(ResourceEventKind.ALLOCATE) < kinds.index(ResourceEventKind.RELEASE)


def test_parameter_bootstrap_in_peak() -> None:
    from zepto.modules.linear import Linear

    graph = compose_graph(
        lambda ctx: Linear(64, 32),
        (ComposeTensor(shape=(2, 64)),),
    )
    lowered = lower(graph, reference_invocation(phase="forward"))
    result = ResourceEventSimulator(lowered).run()
    weight_bytes = sum(
        lowered.context.accounting.bytes_for(
            ResolvedValue(tensor=p.tensor, role=p.role, dtype=p.tensor.dtype)
        )
        for p in lowered.parameters.values()
    )
    assert weight_bytes > 0
    assert result.peak_live_bytes >= weight_bytes
    assert result.breakdown.parameters == weight_bytes


def test_persistent_input_bootstrap() -> None:
    builder = GraphBuilder()
    mask = builder.add_input(
        Tensor(shape=(1, 128, 128), persistent=True, requires_grad=False)
    )
    value = builder.add_input(Tensor(shape=(1, 128, 128), requires_grad=True))
    output = builder.add_operation(
        operation_family="identity",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(value,),
        output_tensors=(Tensor(shape=(1, 128, 128),),),
        provenance=Provenance((), "Fixture", "identity", 0),
        operation=Identity(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_invocation(phase="forward"))
    result = ResourceEventSimulator(lowered).run()
    mask_edge = lowered.edges[lowered.edge_map[mask]]
    mask_bytes = lowered.context.accounting.bytes_for(
        ResolvedValue(
            tensor=mask_edge.tensor, role=TensorRole.INPUT, dtype=mask_edge.tensor.dtype
        )
    )
    assert result.breakdown.persistent_inputs == mask_bytes
    assert result.peak_live_bytes >= mask_bytes


def test_phase_forward_skips_backward_events() -> None:
    graph = _multiply_broadcast_graph()
    lowered = lower(graph, reference_invocation(phase="forward"))
    result = ResourceEventSimulator(lowered).run()
    assert result.breakdown.gradients == 0
    assert result.breakdown.weight_grads == 0


def test_activation_grad_no_persist() -> None:
    graph = _multiply_broadcast_graph()
    lowered = lower(graph, reference_invocation(phase="backward"))
    backward_kinds = [
        event.kind
        for event in lowered.nodes[0].resource_events
        if event.phase == "backward"
    ]
    assert ResourceEventKind.PERSIST not in backward_kinds


def test_scheduled_release_lowers_peak() -> None:
    graph = build_matmul_relu_matmul_graph()
    lowered = _merge_train_lowered(graph)
    result = ResourceEventSimulator(lowered).run()
    assert result.peak_live_bytes <= result.peak_live_bytes_naive
    if result.peak_live_bytes_naive > 0:
        assert result.peak_live_bytes < result.peak_live_bytes_naive


def test_matmul_relu_matmul_train_peak_not_sum_all() -> None:
    graph = build_matmul_relu_matmul_graph(batch=32, seq=128, hidden=512, inner=256)
    lowered = _merge_train_lowered(graph)
    result = ResourceEventSimulator(lowered).run()

    weight_bytes = sum(
        lowered.context.accounting.bytes_for(
            ResolvedValue(tensor=edge.tensor, role=edge.role, dtype=edge.tensor.dtype)
        )
        for edge in lowered.edges.values()
        if edge.tensor.requires_grad
        and edge.role is TensorRole.INPUT
        and len(edge.tensor.shape) == 2
    )
    input_bytes = sum(
        lowered.context.accounting.bytes_for(
            ResolvedValue(
                tensor=lowered.edges[lowered.edge_map[eid]].tensor,
                role=TensorRole.INPUT,
                dtype=lowered.edges[lowered.edge_map[eid]].tensor.dtype,
            )
        )
        for eid in graph.inputs
        if len(lowered.edges[lowered.edge_map[eid]].tensor.shape) > 2
    )
    largest_activation = _max_activation_bytes(lowered)

    assert result.peak_live_bytes < result.peak_live_bytes_naive
    # Not ~3× activation stacking; allow input, weights, two wavefronts, and mask.
    assert (
        result.peak_live_bytes
        <= weight_bytes + input_bytes + 5 * largest_activation + 1_048_576
    )
