"""Tests for the lowering pass."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from zepto.core import (
    DType,
    Identity,
    LoweringError,
    Maximum,
    Operation,
    PortSpec,
    Provenance,
    ResourceEventKind,
    StructuralGraphBuilder,
    TensorMetadata,
    lower,
    reference_context,
)
from zepto.core.lowering import LoweringRegistry
from zepto.core.lowering.implementations import register_defaults
from zepto.core.lowering.implementations.identity import IdentityImplementation
from zepto.core.lowering.registry import (
    ImplementationDescriptor,
    select_implementation,
)


def _identity_graph() -> tuple:
    builder = StructuralGraphBuilder()
    value = builder.add_input(TensorMetadata((4,)))
    provenance = Provenance((), "Fixture", "identity", 0)
    output = builder.add_operation(
        operation_family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(value,),
        output_metadata=(TensorMetadata((4,)),),
        provenance=provenance,
        operation=Identity(),
    )[0]
    builder.mark_output(output)
    return builder.build(), value, output


def test_identity_lowering_preserves_port_wiring() -> None:
    graph, value, output = _identity_graph()
    ctx = reference_context()
    lowered = lower(graph, ctx)

    assert len(lowered.operations) == 1
    op = lowered.operations[0]
    assert op.implementation == "identity/identity"
    assert op.input_tensors == (lowered.tensor_map[value],)
    assert op.output_tensors == (lowered.tensor_map[output],)


def test_structural_to_lowered_maps_are_bijective_for_identity_pass() -> None:
    graph, value, output = _identity_graph()
    lowered = lower(graph, reference_context())

    assert set(lowered.tensor_map.keys()) == {value, output}
    assert len(lowered.tensor_map) == len(set(lowered.tensor_map.values()))
    assert len(lowered.operation_map) == 1
    assert graph.operations[0] in lowered.operation_map


def test_resolved_dtype_matches_precision_policy() -> None:
    graph, value, _output = _identity_graph()
    ctx = reference_context(default_dtype=DType.FP16)
    lowered = lower(graph, ctx)

    tensor = lowered.tensors[lowered.tensor_map[value]]
    assert tensor.metadata.dtype is DType.FP16


def test_identity_view_emits_alias_event() -> None:
    graph, _value, _output = _identity_graph()
    lowered = lower(graph, reference_context())
    events = lowered.operations[0].resource_events

    assert any(event.kind is ResourceEventKind.ALIAS for event in events)
    assert not any(event.kind is ResourceEventKind.ALLOCATE for event in events)


def test_maximum_identity_produces_allocate_and_flops() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(TensorMetadata((3, 4), requires_grad=True))
    right = builder.add_input(TensorMetadata((3, 4)))
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, right),
        output_metadata=(TensorMetadata((3, 4)),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_context())
    op = lowered.operations[0]

    assert op.implementation == "maximum/identity"
    assert op.forward_flops == 12
    assert op.backward_flops == 12
    assert any(event.kind is ResourceEventKind.ALLOCATE for event in op.resource_events)


def test_saved_for_backward_emits_save_events() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(TensorMetadata((2,), requires_grad=True))
    right = builder.add_input(TensorMetadata((2,), requires_grad=True))
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, right),
        output_metadata=(TensorMetadata((2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_context())
    events = lowered.operations[0].resource_events
    save_kinds = [event for event in events if event.kind is ResourceEventKind.SAVE]

    assert len(save_kinds) == 2


def test_negative_flops_fail_at_lowering() -> None:
    class BadFlops(Operation):
        @property
        def family(self):
            return "bad_flops"

        @property
        def input_ports(self):
            return (PortSpec("input"),)

        @property
        def output_ports(self):
            return (PortSpec("output"),)

        @property
        def backward(self):
            from zepto.core import BackwardSpec

            return BackwardSpec()

        def infer_outputs(self, inputs):
            return inputs

        def saved_for_backward(self, inputs, outputs):
            return ()

        def forward_flops(self, context):
            return -1

        def backward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            from zepto.core.operation.helpers import allocate

            return allocate(result)

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

    builder = StructuralGraphBuilder()
    value = builder.add_input(TensorMetadata((2,)))
    builder.add_operation(
        operation_family="bad_flops",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(value,),
        output_metadata=(TensorMetadata((2,)),),
        provenance=Provenance((), "Fixture", "bad_flops", 0),
        operation=BadFlops(),
    )
    builder.mark_output(value)
    graph = builder.build()

    with pytest.raises(ValueError, match="forward_flops"):
        lower(graph, reference_context(), registry=registry)


def test_implementation_pin_selects_descriptor() -> None:
    graph, _value, _output = _identity_graph()
    ctx = reference_context(
        implementation_pins=MappingProxyType({"identity": "identity/identity"})
    )
    lowered = lower(graph, ctx)

    assert lowered.operations[0].implementation == "identity/identity"
    assert lowered.selections[0].reason == "pinned"


def test_implementation_pin_incompatible_raises() -> None:
    graph, _value, _output = _identity_graph()
    ctx = reference_context(
        implementation_pins=MappingProxyType({"identity": "maximum/identity"})
    )

    with pytest.raises(LoweringError, match="incompatible"):
        lower(graph, ctx)


def test_relu_pattern_selects_relu_mask_implementation() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(TensorMetadata((4,), requires_grad=True))
    zero = builder.add_input(
        TensorMetadata((4,), semantic_type="constant_zero", requires_grad=False)
    )
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, zero),
        output_metadata=(TensorMetadata((4,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_context())
    op = lowered.operations[0]

    assert op.implementation == "maximum/relu-mask"
    assert lowered.selections[0].chosen.id == "maximum/relu-mask"
    assert len(op.auxiliary_tensors) == 1
    save_events = [
        event for event in op.resource_events if event.kind is ResourceEventKind.SAVE
    ]
    assert len(save_events) == 1
    assert save_events[0].value == op.auxiliary_tensors[0]
    assert op.backward_flops == 4


def test_relu_mask_does_not_save_operands() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(TensorMetadata((2,), requires_grad=True))
    zero = builder.add_input(
        TensorMetadata((2,), semantic_type="constant_zero")
    )
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, zero),
        output_metadata=(TensorMetadata((2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_context())
    op = lowered.operations[0]
    saved_targets = {
        event.value
        for event in op.resource_events
        if event.kind is ResourceEventKind.SAVE
    }

    assert lowered.tensor_map[left] not in saved_targets
    assert lowered.tensor_map[zero] not in saved_targets


def test_generic_maximum_selects_identity() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(TensorMetadata((2,), requires_grad=True))
    right = builder.add_input(TensorMetadata((2,), requires_grad=True))
    output = builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, right),
        output_metadata=(TensorMetadata((2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "maximum", 0),
        operation=Maximum(),
    )[0]
    builder.mark_output(output)
    graph = builder.build()

    lowered = lower(graph, reference_context())

    assert lowered.operations[0].implementation == "maximum/identity"
    assert lowered.selections[0].reason == "only_candidate"


def test_selection_records_rejected_candidates() -> None:
    builder = StructuralGraphBuilder()
    left = builder.add_input(TensorMetadata((2,), requires_grad=True))
    zero = builder.add_input(
        TensorMetadata((2,), semantic_type="constant_zero")
    )
    builder.add_operation(
        operation_family="maximum",
        input_ports=(PortSpec("left"), PortSpec("right")),
        output_ports=(PortSpec("output"),),
        input_tensors=(left, zero),
        output_metadata=(TensorMetadata((2,), requires_grad=True),),
        provenance=Provenance((), "Fixture", "relu", 0),
        operation=Maximum(),
    )
    graph = builder.build()

    registry = LoweringRegistry()
    register_defaults(registry)
    structural = graph.operation(graph.operations[0])
    _impl, selection = select_implementation(
        structural, graph, reference_context(), registry
    )

    assert selection.chosen.id == "maximum/relu-mask"
    assert selection.reason == "highest_priority"
    assert not selection.rejected
