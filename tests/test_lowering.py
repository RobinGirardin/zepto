"""Tests for the lowering pass."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from zepto.analysis import (
    AccountingPolicy,
    LoweringError,
    PrecisionPolicy,
    lower,
    reference_invocation,
)
from zepto.analysis.lowering import InvocationContext, LoweringRegistry
from zepto.analysis.lowering.helpers import build_estimation_context
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.implementations.identity import IdentityImplementation
from zepto.analysis.lowering.registry import (
    ImplementationDescriptor,
    select_implementation,
)
from zepto.compose.values import Tensor
from zepto.graph import Provenance, GraphBuilder
from zepto.semantic import (
    BackwardSpec,
    DType,
    Identity,
    MatMul,
    Maximum,
    Multiply,
    Operation,
    Port,
    ResourceEventKind,
    TensorRole,
)
from zepto.semantic.operations.helpers import GRAD_RIGHT, GRAD_RIGHT_UNREDUCED, allocate


def _identity_graph() -> tuple:
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
    return builder.build(), value, output


def test_identity_lowering_preserves_port_wiring() -> None:
    graph, value, output = _identity_graph()
    ctx = reference_invocation()
    lowered = lower(graph, ctx)

    assert len(lowered.nodes) == 1
    op = lowered.nodes[0]
    assert op.implementation == "identity/identity"
    assert op.input_edges == (lowered.edge_map[value],)
    assert op.output_edges == (lowered.edge_map[output],)


def test_structural_to_lowered_maps_are_bijective_for_identity_pass() -> None:
    graph, value, output = _identity_graph()
    lowered = lower(graph, reference_invocation())

    assert set(lowered.edge_map.keys()) == {value, output}
    assert len(lowered.edge_map) == len(set(lowered.edge_map.values()))
    assert len(lowered.node_map) == 1
    assert graph.node_order[0] in lowered.node_map


def test_resolved_dtype_matches_precision_policy() -> None:
    graph, value, _output = _identity_graph()
    ctx = reference_invocation(default_dtype=DType.FP16)
    lowered = lower(graph, ctx)

    edge = lowered.edges[lowered.edge_map[value]]
    assert edge.tensor.dtype is DType.FP16


def test_identity_view_emits_alias_event() -> None:
    graph, _value, _output = _identity_graph()
    lowered = lower(graph, reference_invocation())
    events = lowered.nodes[0].resource_events

    assert any(event.kind is ResourceEventKind.ALIAS for event in events)
    assert not any(event.kind is ResourceEventKind.ALLOCATE for event in events)


def test_maximum_identity_produces_allocate_and_flops() -> None:
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
    graph = builder.build()

    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]

    assert op.implementation == "maximum/identity"
    assert op.forward_flops == 12
    assert op.backward_flops == 12
    assert any(event.kind is ResourceEventKind.ALLOCATE for event in op.resource_events)


def test_saved_for_backward_emits_save_events() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(2,), requires_grad=True))
    right = builder.add_input(Tensor(shape=(2,), requires_grad=True))
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, right),
        output_tensors=(Tensor(shape=(2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_invocation())
    events = lowered.nodes[0].resource_events
    save_kinds = [event for event in events if event.kind is ResourceEventKind.SAVE]

    assert len(save_kinds) == 2


def test_negative_flops_fail_at_lowering() -> None:
    class BadFlops(Operation):
        @property
        def family(self):
            return "bad_flops"

        @property
        def input_ports(self):
            return (Port("input"),)

        @property
        def output_ports(self):
            return (Port("output"),)

        @property
        def backward(self):
            return BackwardSpec()

        def infer_outputs(self, inputs, parameters=()):
            return inputs

        def saved_for_backward(self, inputs, parameters=(), outputs=()):
            return ()

        def forward_flops(self, context):
            return -1

        def backward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            return allocate(len(self.output_ports))

    registry = LoweringRegistry()
    register_defaults(registry)
    registry.register(
        IdentityImplementation(
            operation=BadFlops(),
            descriptor=ImplementationDescriptor(
                id="bad_flops/identity",
                family="bad_flops",
            ),
        )
    )

    builder = GraphBuilder()
    value = builder.add_input(Tensor(shape=(2,)))
    builder.add_operation(
        operation_family="bad_flops",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(value,),
        output_tensors=(Tensor(shape=(2,)),),
        provenance=Provenance((), "Fixture", "bad_flops", 0),
        operation=BadFlops(),
    )
    builder.mark_output(value)
    graph = builder.build()

    with pytest.raises(ValueError, match="forward_flops"):
        lower(graph, reference_invocation(), registry=registry)


def test_implementation_pin_selects_descriptor() -> None:
    graph, _value, _output = _identity_graph()
    ctx = reference_invocation(
        implementation_pins=MappingProxyType({"identity": "identity/identity"})
    )
    lowered = lower(graph, ctx)

    assert lowered.nodes[0].implementation == "identity/identity"
    assert lowered.selections[0].reason == "pinned"


def test_implementation_pin_incompatible_raises() -> None:
    graph, _value, _output = _identity_graph()
    ctx = reference_invocation(
        implementation_pins=MappingProxyType({"identity": "maximum/identity"})
    )

    with pytest.raises(LoweringError, match="incompatible"):
        lower(graph, ctx)


def test_relu_pattern_selects_relu_mask_implementation() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(4,), requires_grad=True))
    zero = builder.add_input(
        Tensor(shape=(4,), semantic_type="constant_zero", requires_grad=False)
    )
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, zero),
        output_tensors=(Tensor(shape=(4,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]

    assert op.implementation == "region/relu"
    assert lowered.region_selections[0].chosen.id == "region/relu"
    assert len(op.auxiliary_edges) == 1
    save_events = [
        event for event in op.resource_events if event.kind is ResourceEventKind.SAVE
    ]
    assert len(save_events) == 1
    assert save_events[0].value == op.auxiliary_edges[0]
    assert op.backward_flops == 4
    mask_edge = lowered.edges[op.auxiliary_edges[0]]
    assert mask_edge.role is TensorRole.AUXILIARY


def test_relu_mask_does_not_save_operands() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(2,), requires_grad=True))
    zero = builder.add_input(
        Tensor(shape=(2,), semantic_type="constant_zero")
    )
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, zero),
        output_tensors=(Tensor(shape=(2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_invocation())
    op = lowered.nodes[0]
    saved_targets = {
        event.value
        for event in op.resource_events
        if event.kind is ResourceEventKind.SAVE
    }

    assert lowered.edge_map[left] not in saved_targets
    assert lowered.edge_map[zero] not in saved_targets


def test_generic_maximum_selects_identity() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(2,), requires_grad=True))
    right = builder.add_input(Tensor(shape=(2,), requires_grad=True))
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, right),
        output_tensors=(Tensor(shape=(2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_invocation())

    assert lowered.nodes[0].implementation == "maximum/identity"
    assert lowered.selections[0].reason == "only_candidate"


def test_selection_records_rejected_candidates() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(2,), requires_grad=True))
    zero = builder.add_input(
        Tensor(shape=(2,), semantic_type="constant_zero")
    )
    builder.add_operation(
        operation_family="maximum",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, zero),
        output_tensors=(Tensor(shape=(2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )
    graph = builder.build()

    registry = LoweringRegistry()
    register_defaults(registry)
    structural = graph.node(graph.node_order[0])
    _impl, selection = select_implementation(
        structural, graph, reference_invocation(), registry
    )

    assert selection.chosen.id == "maximum/relu-mask"
    assert selection.reason == "highest_priority"
    assert not selection.rejected


def _multiply_broadcast_graph():
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=(32, 128, 512), requires_grad=True))
    bias = builder.add_input(Tensor(shape=(512,), requires_grad=True))
    output = builder.add_operation(
        operation_family="multiply",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x, bias),
        output_tensors=(Tensor(shape=(32, 128, 512), requires_grad=True),),
        provenance=Provenance((), "Fixture", "multiply", 0),
        operation=Multiply(),
    )[0]
    builder.mark_output(output)
    return builder.build(), output


class TestBroadcastBackwardLowering:
    def test_multiply_backward_remaps_aux_port_events(self) -> None:
        graph, _output = _multiply_broadcast_graph()
        lowered = lower(graph, reference_invocation(phase="backward"))
        op = lowered.nodes[0]
        assert len(op.auxiliary_edges) == 2
        event_targets = {event.value for event in op.resource_events}
        assert event_targets.issuperset(set(op.auxiliary_edges))

    def test_multiply_forward_has_no_backward_events(self) -> None:
        graph, _output = _multiply_broadcast_graph()
        lowered = lower(graph, reference_invocation(phase="forward"))
        op = lowered.nodes[0]
        assert not any(
            event.phase == "backward" for event in op.resource_events
        )

    def test_all_backward_event_targets_are_known_tensors(self) -> None:
        graph, _output = _multiply_broadcast_graph()
        lowered = lower(graph, reference_invocation(phase="backward"))
        op = lowered.nodes[0]
        known = set(op.input_edges + op.output_edges + op.auxiliary_edges)
        for event in op.resource_events:
            assert event.value in known

    def test_unreduced_allocate_release_order(self) -> None:
        graph, _output = _multiply_broadcast_graph()
        lowered = lower(graph, reference_invocation(phase="backward"))
        op = lowered.nodes[0]
        kinds = [
            event.kind
            for event in op.resource_events
            if event.phase == "backward"
        ]
        assert ResourceEventKind.ALLOCATE in kinds
        assert ResourceEventKind.RELEASE in kinds
        assert ResourceEventKind.PERSIST in kinds
        allocate_idx = next(
            i
            for i, event in enumerate(op.resource_events)
            if event.kind is ResourceEventKind.ALLOCATE
            and event.phase == "backward"
        )
        release_idx = next(
            i
            for i, event in enumerate(op.resource_events)
            if event.kind is ResourceEventKind.RELEASE
        )
        assert allocate_idx < release_idx

    def test_matmul_weight_backward_aux_shapes(self) -> None:
        builder = GraphBuilder()
        left = builder.add_input(Tensor(shape=(32, 128, 512), requires_grad=True))
        right = builder.add_input(Tensor(shape=(512, 64), requires_grad=True))
        output = builder.add_operation(
            operation_family="matmul",
            input_ports=(Port("left"), Port("right")),
            output_ports=(Port("output"),),
            input_edges=(left, right),
            output_tensors=(Tensor(shape=(32, 128, 64), requires_grad=True),),
            provenance=Provenance((), "Fixture", "matmul", 0),
            operation=MatMul(),
        )[0]
        builder.mark_output(output)
        graph = builder.build()
        operation = graph.node(graph.node_order[0])
        assert operation.result is not None
        unreduced_id = operation.auxiliary_edges[GRAD_RIGHT_UNREDUCED]
        assert graph.edge(unreduced_id).tensor.shape == (32, 512, 64)
        grad_id = operation.auxiliary_edges[GRAD_RIGHT]
        assert graph.edge(grad_id).tensor.shape == (512, 64)


def test_input_role_consistent_in_estimation_and_lowering() -> None:
    graph, value, _output = _identity_graph()
    precision = PrecisionPolicy(
        default_dtype=DType.FP32,
        by_role=(
            (TensorRole.INPUT, DType.FP16),
            (TensorRole.ACTIVATION, DType.BF16),
        ),
    )
    ctx = InvocationContext(
        phase="forward",
        hardware="generic",
        backend="reference",
        precision=precision,
        accounting=AccountingPolicy(precision),
    )
    node = graph.node(graph.node_order[0])
    estimation = build_estimation_context(node, graph, ctx)
    input_value = estimation.value_for("input")
    assert input_value is not None
    assert input_value.role is TensorRole.INPUT
    assert input_value.dtype is DType.FP16

    lowered = lower(graph, ctx)
    lowered_input = lowered.edges[lowered.edge_map[value]]
    assert lowered_input.role is TensorRole.INPUT
    assert lowered_input.tensor.dtype is DType.FP16


def test_structural_aux_gradient_role() -> None:
    graph, _output = _multiply_broadcast_graph()
    ctx = reference_invocation()
    node = graph.node(graph.node_order[0])
    lowered = lower(graph, ctx)
    grad_id = node.auxiliary_edges[GRAD_RIGHT]
    unreduced_id = node.auxiliary_edges[GRAD_RIGHT_UNREDUCED]
    assert lowered.edges[lowered.edge_map[grad_id]].role is TensorRole.GRADIENT
    assert lowered.edges[lowered.edge_map[unreduced_id]].role is TensorRole.WORKSPACE

    estimation = build_estimation_context(node, graph, ctx)
    assert estimation.value_for(GRAD_RIGHT).role is TensorRole.GRADIENT
    assert estimation.value_for(GRAD_RIGHT_UNREDUCED).role is TensorRole.WORKSPACE


def test_graph_input_lowered_role_is_input() -> None:
    graph, value, output = _identity_graph()
    lowered = lower(graph, reference_invocation())
    assert lowered.edges[lowered.edge_map[value]].role is TensorRole.INPUT
    assert lowered.edges[lowered.edge_map[output]].role is TensorRole.ACTIVATION
