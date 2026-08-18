import pytest

from zepto.core import (
    CrossGraphReferenceError,
    GraphAlreadyFinalizedError,
    GraphCompositionContext,
    Module,
    MetadataMismatchError,
    OperationSpec,
    PortSpec,
    Provenance,
    StructuralGraphBuilder,
    TensorMetadata,
    UnknownOperationError,
    build_graph,
    identity,
    linear,
)


def test_builder_preserves_order_and_shared_parameters() -> None:
    builder = StructuralGraphBuilder()
    value = builder.add_input(TensorMetadata((8,)))
    parameter = builder.add_parameter(TensorMetadata((8,)))
    provenance = Provenance((), "Fixture", "identity", 0)
    first = builder.add_operation(
        operation_family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(value,),
        output_metadata=(TensorMetadata((8,)),),
        parameter_ids=(parameter,),
        provenance=provenance,
    )[0]
    second = builder.add_operation(
        operation_family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(first,),
        output_metadata=(TensorMetadata((8,)),),
        parameter_ids=(parameter,),
        provenance=Provenance((), "Fixture", "identity", 1),
    )[0]
    builder.mark_output(second)

    graph = builder.build()

    assert len(graph.operations) == 2
    assert graph.operation(graph.operations[0]).parameter_ids == (parameter,)
    assert graph.tensor(first).producer is not None
    assert graph.tensor(first).consumers[0].port_name == "input"


def test_cross_graph_reference_is_rejected() -> None:
    first = StructuralGraphBuilder()
    foreign = first.add_input(TensorMetadata((1,)))
    first.build()
    second = StructuralGraphBuilder()

    with pytest.raises(CrossGraphReferenceError):
        second.add_operation(
            operation_family="identity",
            input_ports=(PortSpec("input"),),
            output_ports=(PortSpec("output"),),
            input_tensors=(foreign,),
            output_metadata=(TensorMetadata((1,)),),
            provenance=Provenance((), None, "identity", 0),
        )


def test_builder_is_explicitly_finalized() -> None:
    builder = StructuralGraphBuilder()
    builder.build()
    with pytest.raises(GraphAlreadyFinalizedError):
        builder.add_input(TensorMetadata((1,)))


def test_missing_operation_uses_domain_error() -> None:
    builder = StructuralGraphBuilder()
    graph = builder.build()

    with pytest.raises(UnknownOperationError):
        graph.operation(graph.operations[0] if graph.operations else None)


class IdentityModule(Module):
    def forward(self, value):
        return identity(value)


def test_module_composition_records_provenance() -> None:
    graph = build_graph(IdentityModule(), (TensorMetadata((2,)),))
    operation = graph.operation(graph.operations[0])
    assert operation.operation_family == "identity"
    assert operation.provenance.module_path == ("IdentityModule",)
    assert operation.input_ports[0].metadata == TensorMetadata((2,))
    assert operation.output_ports[0].metadata == TensorMetadata((2,))


def test_linear_binds_inferred_metadata_to_output_port() -> None:
    class LinearModule(Module):
        def forward(self, value):
            return linear(value, output_features=4)

    graph = build_graph(LinearModule(), (TensorMetadata((2, 8)),))
    operation = graph.operation(graph.operations[0])

    assert operation.output_ports[0].metadata == TensorMetadata((2, 4))


def test_operation_spec_is_not_mutated_by_metadata_binding() -> None:
    spec = OperationSpec(
        family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        infer_outputs=lambda inputs: inputs,
    )

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        context.apply(spec, value)

    assert spec.input_ports[0].metadata is None
    assert spec.output_ports[0].metadata is None


def test_declared_port_metadata_mismatch_is_rejected() -> None:
    spec = OperationSpec(
        family="identity",
        input_ports=(
            PortSpec("input", metadata=TensorMetadata((4,))),
        ),
        output_ports=(PortSpec("output"),),
        infer_outputs=lambda inputs: inputs,
    )

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        with pytest.raises(MetadataMismatchError):
            context.apply(spec, value)


def test_declared_output_metadata_mismatch_is_rejected() -> None:
    spec = OperationSpec(
        family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(
            PortSpec("output", metadata=TensorMetadata((4,))),
        ),
        infer_outputs=lambda inputs: inputs,
    )

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        with pytest.raises(MetadataMismatchError):
            context.apply(spec, value)


def test_output_inference_is_called_once() -> None:
    calls = 0

    def infer(inputs):
        nonlocal calls
        calls += 1
        return inputs

    spec = OperationSpec(
        family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        infer_outputs=infer,
    )

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        context.apply(spec, value)

    assert calls == 1
