import pytest
from dataclasses import replace

from zepto.core import (
    AliasSpec,
    BackwardSpec,
    CrossGraphReferenceError,
    GraphAlreadyFinalizedError,
    GraphCompositionContext,
    Identity,
    Materialization,
    MetadataMismatchError,
    Module,
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


class Contiguous(Operation):
    @property
    def family(self):
        return "contiguous"

    @property
    def input_ports(self):
        return (PortSpec("input"),)

    @property
    def output_ports(self):
        return (PortSpec("output"),)

    @property
    def backward(self):
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(self, inputs):
        return (inputs[0],)

    def output_aliases(self):
        return (AliasSpec("input", Materialization.CONTIGUOUS_COPY),)

    def saved_for_backward(self, inputs, outputs):
        return ()

    def forward_flops(self, context):
        return 0

    def resource_events(self, context, result):
        return ()


def test_view_operations_share_input_storage() -> None:
    builder = StructuralGraphBuilder()
    value = builder.add_input(TensorMetadata((4,)))
    provenance = Provenance((), "Fixture", "view", 0)
    for operation in (Identity(),):
        output = builder.add_operation(
            operation_family=operation.family,
            input_ports=operation.input_ports,
            output_ports=operation.output_ports,
            input_tensors=(value,),
            output_metadata=(TensorMetadata((4,)),),
            provenance=provenance,
            operation=operation,
        )[0]
        assert builder._tensors[output].storage_id == builder._tensors[value].storage_id
        value = output


def test_contiguous_copy_allocates_distinct_storage() -> None:
    builder = StructuralGraphBuilder()
    value = builder.add_input(TensorMetadata((4,)))
    provenance = Provenance((), "Fixture", "contiguous", 0)
    output = builder.add_operation(
        operation_family="contiguous",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(value,),
        output_metadata=(TensorMetadata((4,)),),
        provenance=provenance,
        operation=Contiguous(),
    )[0]
    assert builder._tensors[output].storage_id != builder._tensors[value].storage_id


def test_graph_validator_rejects_illegal_shared_copy_storage() -> None:
    builder = StructuralGraphBuilder()
    value = builder.add_input(TensorMetadata((4,)))
    provenance = Provenance((), "Fixture", "contiguous", 0)
    output = builder.add_operation(
        operation_family="contiguous",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        input_tensors=(value,),
        output_metadata=(TensorMetadata((4,)),),
        provenance=provenance,
        operation=Contiguous(),
    )[0]
    input_storage = builder._tensors[value].storage_id
    builder._tensors[output] = replace(
        builder._tensors[output], storage_id=input_storage
    )

    with pytest.raises(ValueError, match="illegally shares input storage"):
        builder.build()


def test_add_parameter_trainable_round_trips() -> None:
    builder = StructuralGraphBuilder()
    parameter_id = builder.add_parameter(
        TensorMetadata((8,), requires_grad=True),
        trainable=False,
    )
    graph = builder.build()
    parameter = graph.parameter(parameter_id)
    assert parameter.trainable is False
    assert parameter.metadata.requires_grad is True
