"""Tests for parameter-aware composition and operation contracts."""

import pytest

from zepto.analysis import ResolvedValue, lower, reference_invocation
from zepto.analysis.lowering.helpers import build_estimation_context
from zepto.compose import Compose, Parameter, Tensor, compose_graph
from zepto.graph import Provenance, GraphBuilder
from zepto.semantic import (
    DType,
    EstimationContext,
    LinearMatMul,
    OperationError,
    Port,
    TensorRole,
    ValueKind,
)
from zepto.modules import Linear


def _resolved(tensor: Tensor, role: TensorRole = TensorRole.ACTIVATION) -> ResolvedValue:
    return ResolvedValue(tensor=tensor, role=role, dtype=tensor.dtype or DType.FP32)


def test_graph_parameter_round_trips_through_context() -> None:
    with Compose() as ctx:
        param = ctx.parameter(Parameter(shape=(4, 8), trainable=False))
        assert isinstance(param, Parameter)
        assert param.trainable is False
        assert param.shape == (4, 8)


def test_operation_call_resolves_parameter_tensor() -> None:
    with Compose() as ctx:
        activation = ctx.input(Tensor(shape=(2, 3, 4)))
        weight = ctx.parameter(Parameter(shape=(4, 5), semantic_type="weight"))
        output = LinearMatMul()(activation, parameters=(weight,))  # type: ignore[arg-type]
        assert output.shape == (2, 3, 5)

        graph = ctx.build()
        op = graph.node(graph.node_order[0])
        assert op.parameter_ports[0].name == "weight"
        assert op.parameter_ports[0].value_kind is ValueKind.PARAMETER
        assert op.parameter_ids[0] == weight.id
        stored = graph.parameter(weight.id)
        assert stored.trainable is True
        assert stored.as_tensor().requires_grad is True
        assert stored.as_tensor().persistent is True


def test_parameter_role_via_estimation_and_lowering() -> None:
    with Compose() as ctx:
        activation = ctx.input(Tensor(shape=(2, 4)))
        weight = ctx.parameter(Parameter(shape=(4, 8), semantic_type="weight"))
        LinearMatMul()(activation, parameters=(weight,))  # type: ignore[arg-type]
        graph = ctx.build()

    context = reference_invocation()
    node = graph.node(graph.node_order[0])
    estimation = build_estimation_context(node, graph, context)
    weight_value = estimation.value_for("weight")
    assert weight_value is not None
    assert weight_value.role is TensorRole.PARAMETER

    lowered = lower(graph, context)
    input_edge = lowered.edges[lowered.edge_map[graph.inputs[0]]]
    assert input_edge.role is TensorRole.INPUT


def test_matmul_dispatches_to_linear_matmul_for_graph_parameter() -> None:
    with Compose() as ctx:
        activation = ctx.input(Tensor(shape=(2, 4)))
        weight = ctx.parameter(Parameter(shape=(4, 8), semantic_type="weight"))
        output = LinearMatMul()(activation, parameters=(weight,))  # type: ignore[arg-type]
        assert output.shape == (2, 8)
        graph = ctx.build()
        op = graph.node(graph.node_order[0])
        assert op.operation_family == "linear_matmul"


def test_linear_matmul_saved_for_backward_selects_by_operand_grad() -> None:
    operation = LinearMatMul()
    activation = Tensor(shape=(2, 4), requires_grad=True)
    weight = Tensor(shape=(4, 8), semantic_type="weight", requires_grad=False)
    inferred = operation.infer_result((activation,), (weight,))
    assert inferred.result.saved_for_backward == ("weight",)

    frozen_activation = Tensor(shape=(2, 4), requires_grad=False)
    trainable_weight = Tensor(shape=(4, 8), semantic_type="weight", requires_grad=True)
    inferred = operation.infer_result((frozen_activation,), (trainable_weight,))
    assert inferred.result.saved_for_backward == ("input",)


def test_linear_matmul_forward_flops_matches_matmul_formula() -> None:
    operation = LinearMatMul()
    activation = Tensor(shape=(2, 3, 4))
    weight = Tensor(shape=(4, 5), semantic_type="weight")
    output = operation.infer_outputs((activation,), (weight,))[0]
    context = EstimationContext(
        port_values=(
            ("input", _resolved(activation)),
            ("weight", _resolved(weight, TensorRole.PARAMETER)),
            ("output", _resolved(output)),
        )
    )
    assert operation.forward_flops(context) == 2 * 2 * 3 * 4 * 5


def test_linear_matmul_validates_in_features() -> None:
    operation = LinearMatMul()
    with pytest.raises(OperationError, match="in_features mismatch"):
        operation.infer_result(
            (Tensor(shape=(2, 3)),),
            (Tensor(shape=(5, 4), semantic_type="weight"),),
        )


def test_add_operation_rejects_parameter_port_mismatch() -> None:
    builder = GraphBuilder()
    value = builder.add_input(Tensor(shape=(4,)))
    parameter = builder.add_parameter(Parameter(shape=(4, 4), semantic_type="weight"))
    provenance = Provenance((), "Fixture", "linear_matmul", 0)
    with pytest.raises(OperationError, match="Bound ports do not match"):
        builder.add_operation(
            operation_family="linear_matmul",
            input_ports=(Port("input"),),
            parameter_ports=(Port("wrong", value_kind=ValueKind.PARAMETER),),
            output_ports=(Port("output"),),
            input_edges=(value,),
            parameter_ids=(parameter,),
            output_tensors=(Tensor(shape=(4,)),),
            provenance=provenance,
            operation=LinearMatMul(),
        )


def test_saved_for_backward_resolves_parameter_role() -> None:
    with Compose() as ctx:
        activation = ctx.input(Tensor(shape=(2, 4), requires_grad=True))
        weight = ctx.parameter(Parameter(shape=(4, 8), semantic_type="weight"))
        LinearMatMul()(activation, parameters=(weight,))  # type: ignore[arg-type]
        graph = ctx.build()
        op = graph.node(graph.node_order[0])
        roles = {ref.port_name: ref.role for ref in op.saved_for_backward}
        assert roles["input"] == "input"
        assert roles["weight"] == "parameter"


def test_non_trainable_parameter_projects_requires_grad_false() -> None:
    with Compose() as ctx:
        param = ctx.parameter(Parameter(shape=(2, 2), trainable=False))
        graph = ctx.build()
        assert graph.parameter(param.id).as_tensor().requires_grad is False


def test_flat_parameter_from_context() -> None:
    with Compose() as ctx:
        param = ctx.parameter(Parameter(shape=(4, 8), trainable=False))
        assert param.shape == (4, 8)
        assert param.trainable is False
        assert param.id.graph_id == ctx.builder.graph_id


def test_linear_module_builds_parameter_bound_graph() -> None:
    def make_model(ctx: Compose) -> Linear:
        return Linear(512, 256)

    graph = compose_graph(
        make_model,
        (Tensor(shape=(2, 128, 512)),),
    )
    assert len(graph.parameters) == 1
    weight = graph.parameter(next(iter(graph.parameters)))
    assert weight.shape == (512, 256)

    op = graph.node(next(iter(graph.node_order)))
    assert op.operation_family == "linear_matmul"
    assert len(op.parameter_ports) == 1
    assert op.parameter_ports[0].value_kind is ValueKind.PARAMETER
