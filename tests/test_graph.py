import pytest
from dataclasses import replace

from zepto.compose import (
    Compose,
    Module,
    Parameter,
    Tensor,
    compose_graph,
)
from zepto.graph import (
    CrossGraphReferenceError,
    GraphAlreadyFinalizedError,
    MetadataMismatchError,
    Provenance,
    GraphBuilder,
    UnknownOperationError,
)
from zepto.semantic import (
    AliasSpec,
    BackwardSpec,
    Identity,
    LinearMatMul,
    Materialization,
    Multiply,
    Operation,
    Port,
    PortContract,
    ValueKind,
)
from zepto.semantic.operations.helpers import GRAD_RIGHT, GRAD_RIGHT_UNREDUCED


def test_builder_preserves_order_and_shared_parameters() -> None:
    builder = GraphBuilder()
    activation = builder.add_input(Tensor(shape=(2, 8)))
    parameter = builder.add_parameter(
        Parameter(shape=(8, 4), semantic_type="weight")
    )
    provenance = Provenance((), "Fixture", "linear_matmul", 0)
    weight_port = Port("weight", value_kind=ValueKind.PARAMETER)
    first = builder.add_operation(
        operation_family="linear_matmul",
        input_ports=(Port("input"),),
        parameter_ports=(weight_port,),
        output_ports=(Port("output"),),
        input_edges=(activation,),
        parameter_ids=(parameter,),
        output_tensors=(Tensor(shape=(2, 4)),),
        provenance=provenance,
        operation=LinearMatMul(),
    )[0]
    second = builder.add_operation(
        operation_family="linear_matmul",
        input_ports=(Port("input"),),
        parameter_ports=(weight_port,),
        output_ports=(Port("output"),),
        input_edges=(activation,),
        parameter_ids=(parameter,),
        output_tensors=(Tensor(shape=(2, 4)),),
        provenance=Provenance((), "Fixture", "linear_matmul", 1),
        operation=LinearMatMul(),
    )[0]
    builder.mark_output(second)

    graph = builder.build()

    assert len(graph.node_order) == 2
    assert graph.node(graph.node_order[0]).parameter_ids == (parameter,)
    assert graph.node(graph.node_order[1]).parameter_ids == (parameter,)
    assert len(graph.edge(activation).consumers) == 2
    assert graph.edge(activation).consumers[0].port_name == "input"


def test_cross_graph_reference_is_rejected() -> None:
    first = GraphBuilder()
    foreign = first.add_input(Tensor(shape=(1,)))
    first.build()
    second = GraphBuilder()

    with pytest.raises(CrossGraphReferenceError):
        second.add_operation(
            operation_family="identity",
            input_ports=(Port("input"),),
            output_ports=(Port("output"),),
            input_edges=(foreign,),
            output_tensors=(Tensor(shape=(1,)),),
            provenance=Provenance((), None, "identity", 0),
            operation=Identity(),
        )


def test_builder_is_explicitly_finalized() -> None:
    builder = GraphBuilder()
    builder.build()
    with pytest.raises(GraphAlreadyFinalizedError):
        builder.add_input(Tensor(shape=(1,)))


def test_missing_operation_uses_domain_error() -> None:
    builder = GraphBuilder()
    graph = builder.build()

    with pytest.raises(UnknownOperationError):
        graph.node(graph.node_order[0] if graph.node_order else None)


class IdentityModule(Module):
    def forward(self, value):
        return Identity()(value)


def test_graph_edges_embed_frozen_tensor_snapshots() -> None:
    graph = compose_graph(IdentityModule(), (Tensor(shape=(2, 3)),))
    edge = graph.edge(graph.outputs[0])
    assert edge.tensor.shape == (2, 3)
    assert edge.tensor.requires_grad is False
    assert edge.id == edge.tensor._edge_id


def test_module_composition_records_provenance() -> None:
    graph = compose_graph(IdentityModule(), (Tensor(shape=(2,)),))
    operation = graph.node(graph.node_order[0])
    assert operation.operation_family == "identity"
    assert operation.provenance.module_path == ("IdentityModule",)
    assert operation.input_ports[0].contract is None
    assert graph.edge(operation.input_edges[0]).tensor.shape == (2,)
    assert graph.edge(operation.output_edges[0]).tensor.shape == (2,)


class ContractIdentity(Operation):
    def __init__(self, input_contract=None, output_contract=None):
        self._input_contract = input_contract
        self._output_contract = output_contract

    @property
    def family(self):
        return "contract_identity"

    @property
    def input_ports(self):
        return (Port("input", contract=self._input_contract),)

    @property
    def output_ports(self):
        return (Port("output", contract=self._output_contract),)

    def infer_outputs(self, inputs, parameters=()):
        return inputs

    def saved_for_backward(self, inputs, parameters=(), outputs=()):
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


def test_operation_declaration_is_not_mutated_by_binding() -> None:
    operation = ContractIdentity()

    with Compose() as context:
        value = context.input(Tensor(shape=(3,)))
        operation(value)

    assert operation.input_ports[0].contract is None
    assert operation.output_ports[0].contract is None


def test_declared_port_contract_mismatch_is_rejected() -> None:
    operation = ContractIdentity(input_contract=PortContract(shape=(4,)))

    with Compose() as context:
        value = context.input(Tensor(shape=(3,)))
        with pytest.raises(MetadataMismatchError):
            operation(value)


def test_declared_output_contract_mismatch_is_rejected() -> None:
    operation = ContractIdentity(output_contract=PortContract(shape=(4,)))

    with Compose() as context:
        value = context.input(Tensor(shape=(3,)))
        with pytest.raises(MetadataMismatchError):
            operation(value)


def test_output_inference_is_called_once() -> None:
    calls = 0

    def infer(inputs):
        nonlocal calls
        calls += 1
        return inputs

    class CountingIdentity(ContractIdentity):
        def infer_outputs(self, inputs, parameters=()):
            infer(inputs)
            return super().infer_outputs(inputs)

    with Compose() as context:
        value = context.input(Tensor(shape=(3,)))
        CountingIdentity()(value)

    assert calls == 1


class Contiguous(Operation):
    @property
    def family(self):
        return "contiguous"

    @property
    def input_ports(self):
        return (Port("input"),)

    @property
    def output_ports(self):
        return (Port("output"),)

    @property
    def backward(self):
        return BackwardSpec(
            supported=True,
            gradient_inputs=("output",),
            gradient_outputs=("input",),
        )

    def infer_outputs(self, inputs, parameters=()):
        return (inputs[0],)

    def output_aliases(self):
        return (AliasSpec("input", Materialization.CONTIGUOUS_COPY),)

    def saved_for_backward(self, inputs, parameters=(), outputs=()):
        return ()

    def forward_flops(self, context):
        return 0

    def resource_events(self, context, result):
        return ()


def test_view_operations_share_input_storage() -> None:
    builder = GraphBuilder()
    value = builder.add_input(Tensor(shape=(4,)))
    provenance = Provenance((), "Fixture", "view", 0)
    for operation in (Identity(),):
        output = builder.add_operation(
            operation_family=operation.family,
            input_ports=operation.input_ports,
            output_ports=operation.output_ports,
            input_edges=(value,),
            output_tensors=(Tensor(shape=(4,)),),
            provenance=provenance,
            operation=operation,
        )[0]
        assert builder._edges[output].storage_id == builder._edges[value].storage_id
        value = output


def test_contiguous_copy_allocates_distinct_storage() -> None:
    builder = GraphBuilder()
    value = builder.add_input(Tensor(shape=(4,)))
    provenance = Provenance((), "Fixture", "contiguous", 0)
    output = builder.add_operation(
        operation_family="contiguous",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(value,),
        output_tensors=(Tensor(shape=(4,)),),
        provenance=provenance,
        operation=Contiguous(),
    )[0]
    assert builder._edges[output].storage_id != builder._edges[value].storage_id


def test_graph_validator_rejects_illegal_shared_copy_storage() -> None:
    builder = GraphBuilder()
    value = builder.add_input(Tensor(shape=(4,)))
    provenance = Provenance((), "Fixture", "contiguous", 0)
    output = builder.add_operation(
        operation_family="contiguous",
        input_ports=(Port("input"),),
        output_ports=(Port("output"),),
        input_edges=(value,),
        output_tensors=(Tensor(shape=(4,)),),
        provenance=provenance,
        operation=Contiguous(),
    )[0]
    input_storage = builder._edges[value].storage_id
    builder._edges[output] = replace(
        builder._edges[output], storage_id=input_storage
    )

    with pytest.raises(ValueError, match="illegally shares input storage"):
        builder.build()


def test_add_parameter_trainable_round_trips() -> None:
    builder = GraphBuilder()
    parameter_id = builder.add_parameter(Parameter(shape=(8,), trainable=False))
    graph = builder.build()
    parameter = graph.parameter(parameter_id)
    assert parameter.trainable is False
    assert parameter.shape == (8,)


def test_multiply_allocates_active_aux_tensors_only() -> None:
    builder = GraphBuilder()
    x = builder.add_input(Tensor(shape=(32, 128, 512), requires_grad=True))
    bias = builder.add_input(Tensor(shape=(512,), requires_grad=True))
    provenance = Provenance((), "Fixture", "multiply", 0)
    output = builder.add_operation(
        operation_family="multiply",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(x, bias),
        output_tensors=(Tensor(shape=(32, 128, 512)),),
        provenance=provenance,
        operation=Multiply(),
    )[0]
    graph = builder.build()
    operation = graph.node(graph.node_order[0])
    assert operation.auxiliary_edges is not None
    assert set(operation.auxiliary_edges) == {
        GRAD_RIGHT,
        GRAD_RIGHT_UNREDUCED,
    }
    assert output in graph.edges
    for aux_id in operation.auxiliary_edges.values():
        assert aux_id in graph.edges


def test_inactive_aux_ports_have_no_tensor_id() -> None:
    builder = GraphBuilder()
    left = builder.add_input(Tensor(shape=(2, 3), requires_grad=True))
    right = builder.add_input(Tensor(shape=(2, 3), requires_grad=True))
    builder.add_operation(
        operation_family="multiply",
        input_ports=(Port("left"), Port("right")),
        output_ports=(Port("output"),),
        input_edges=(left, right),
        output_tensors=(Tensor(shape=(2, 3)),),
        provenance=Provenance((), "Fixture", "multiply", 0),
        operation=Multiply(),
    )
    graph = builder.build()
    operation = graph.node(graph.node_order[0])
    assert GRAD_RIGHT_UNREDUCED not in (operation.auxiliary_edges or {})


def test_operation_result_has_no_outputs() -> None:
    graph = compose_graph(IdentityModule(), (Tensor(shape=(2, 3)),))
    node = graph.node(graph.node_order[0])
    assert not hasattr(node.result, "outputs")
    assert graph.edge(node.output_edges[0]).tensor.shape == (2, 3)
