import pytest

from zepto.core import (
    BackwardSpec,
    CrossGraphReferenceError,
    GraphAlreadyFinalizedError,
    GraphCompositionContext,
    Identity,
    Module,
    MetadataMismatchError,
    Operation,
    PortSpec,
    Provenance,
    StructuralGraphBuilder,
    TensorMetadata,
    UnknownOperationError,
    build_graph,
    identity,
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
        operation=Identity(),
    )[0]
    second = builder.add_operation(
        operation_family="identity",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(first,),
        output_metadata=(TensorMetadata((8,)),),
        parameter_ids=(parameter,),
        provenance=Provenance((), "Fixture", "identity", 1),
        operation=Identity(),
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
            operation=Identity(),
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


class MetadataIdentity(Operation):
    def __init__(self, input_metadata=None, output_metadata=None):
        self._input_metadata = input_metadata
        self._output_metadata = output_metadata

    @property
    def family(self):
        return "metadata_identity"

    @property
    def input_ports(self):
        return (PortSpec("input", metadata=self._input_metadata),)

    @property
    def output_ports(self):
        return (PortSpec("output", metadata=self._output_metadata),)

    def infer_outputs(self, inputs):
        return inputs

    def saved_for_backward(self, inputs, outputs):
        return ()

    @property
    def backward(self):
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def forward_flops(self, context):
        return 0

    def resource_events(self, context, result):
        return ()


def test_operation_declaration_is_not_mutated_by_metadata_binding() -> None:
    operation = MetadataIdentity()

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        context.apply(operation, value)

    assert operation.input_ports[0].metadata is None
    assert operation.output_ports[0].metadata is None


def test_declared_port_metadata_mismatch_is_rejected() -> None:
    operation = MetadataIdentity(input_metadata=TensorMetadata((4,)))

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        with pytest.raises(MetadataMismatchError):
            context.apply(operation, value)


def test_declared_output_metadata_mismatch_is_rejected() -> None:
    operation = MetadataIdentity(output_metadata=TensorMetadata((4,)))

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        with pytest.raises(MetadataMismatchError):
            context.apply(operation, value)


def test_output_inference_is_called_once() -> None:
    calls = 0

    def infer(inputs):
        nonlocal calls
        calls += 1
        return inputs

    class CountingIdentity(MetadataIdentity):
        def infer_outputs(self, inputs):
            infer(inputs)
            return super().infer_outputs(inputs)

    with GraphCompositionContext() as context:
        value = context.input(TensorMetadata((3,)))
        context.apply(CountingIdentity(), value)

    assert calls == 1
