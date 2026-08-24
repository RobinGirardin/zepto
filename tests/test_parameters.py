"""Tests for parameter-aware composition and operation contracts."""

import pytest

from zepto.core import (
    GraphCompositionContext,
    GraphParameter,
    LinearMatMul,
    OperationError,
    PortSpec,
    ValueKind,
    ValueMetadata,
    build_graph,
    linear_matmul,
)
from zepto.core.graph import StructuralGraphBuilder
from zepto.core.parameter import bound_parameter_metadata
from zepto.core.provenance import Provenance
from zepto.modules import Linear


def test_graph_parameter_round_trips_through_context() -> None:
    with GraphCompositionContext() as ctx:
        param = ctx.parameter(ValueMetadata((4, 8)), trainable=False)
        assert isinstance(param, GraphParameter)
        assert param.trainable is False
        assert param.metadata.shape == (4, 8)


def test_apply_resolves_parameter_metadata() -> None:
    with GraphCompositionContext() as ctx:
        activation = ctx.input(ValueMetadata((2, 3, 4)))
        weight = ctx.parameter(ValueMetadata((4, 5), semantic_type="weight"))
        output = linear_matmul(activation, weight)
        assert output.metadata.shape == (2, 3, 5)

        graph = ctx.build()
        op = graph.operation(graph.operations[0])
        assert op.parameter_ports[0].name == "weight"
        assert op.parameter_ports[0].value_kind is ValueKind.PARAMETER
        assert op.parameter_ids[0] == weight.id
        bound = bound_parameter_metadata(graph.parameter(weight.id))
        assert bound.requires_grad is True
        assert bound.role.value == "parameter"
        assert bound.persistent is True


def test_matmul_dispatches_to_linear_matmul_for_graph_parameter() -> None:
    from zepto.core import matmul

    with GraphCompositionContext() as ctx:
        activation = ctx.input(ValueMetadata((2, 4)))
        weight = ctx.parameter(ValueMetadata((4, 8), semantic_type="weight"))
        output = matmul(activation, weight)
        assert output.metadata.shape == (2, 8)
        graph = ctx.build()
        op = graph.operation(graph.operations[0])
        assert op.operation_family == "linear_matmul"


def test_linear_matmul_saved_for_backward_selects_by_operand_grad() -> None:
    operation = LinearMatMul()
    activation = ValueMetadata((2, 4), requires_grad=True)
    weight = ValueMetadata((4, 8), semantic_type="weight", requires_grad=False)
    result = operation.infer_result((activation,), (weight,))
    assert result.saved_for_backward == ("weight",)

    frozen_activation = ValueMetadata((2, 4), requires_grad=False)
    trainable_weight = ValueMetadata(
        (4, 8), semantic_type="weight", requires_grad=True
    )
    result = operation.infer_result((frozen_activation,), (trainable_weight,))
    assert result.saved_for_backward == ("input",)


def test_linear_matmul_forward_flops_matches_matmul_formula() -> None:
    from zepto.core import EstimationContext

    operation = LinearMatMul()
    activation = ValueMetadata((2, 3, 4))
    weight = ValueMetadata((4, 5), semantic_type="weight")
    output = operation.infer_outputs((activation,), (weight,))[0]
    context = EstimationContext(
        port_metadata=(
            ("input", activation),
            ("weight", weight),
            ("output", output),
        )
    )
    assert operation.forward_flops(context) == 2 * 2 * 3 * 4 * 5


def test_linear_matmul_validates_in_features() -> None:
    operation = LinearMatMul()
    with pytest.raises(OperationError, match="in_features mismatch"):
        operation.infer_result(
            (ValueMetadata((2, 3)),),
            (ValueMetadata((5, 4), semantic_type="weight"),),
        )


def test_add_operation_rejects_parameter_port_mismatch() -> None:
    builder = StructuralGraphBuilder()
    value = builder.add_input(ValueMetadata((4,)))
    parameter = builder.add_parameter(ValueMetadata((4, 4), semantic_type="weight"))
    provenance = Provenance((), "Fixture", "linear_matmul", 0)
    with pytest.raises(OperationError, match="Bound ports do not match"):
        builder.add_operation(
            operation_family="linear_matmul",
            input_ports=(PortSpec("input"),),
            parameter_ports=(PortSpec("wrong", value_kind=ValueKind.PARAMETER),),
            output_ports=(PortSpec("output"),),
            input_tensors=(value,),
            parameter_ids=(parameter,),
            output_metadata=(ValueMetadata((4,)),),
            provenance=provenance,
            operation=LinearMatMul(),
        )


def test_saved_for_backward_resolves_parameter_role() -> None:
    with GraphCompositionContext() as ctx:
        activation = ctx.input(ValueMetadata((2, 4), requires_grad=True))
        weight = ctx.parameter(ValueMetadata((4, 8), semantic_type="weight"))
        linear_matmul(activation, weight)
        graph = ctx.build()
        op = graph.operation(graph.operations[0])
        roles = {ref.port_name: ref.role for ref in op.saved_for_backward}
        assert roles["input"] == "input"
        assert roles["weight"] == "parameter"


def test_bound_parameter_metadata_derives_requires_grad_from_trainable() -> None:
    with GraphCompositionContext() as ctx:
        param = ctx.parameter(ValueMetadata((2, 2)), trainable=False)
        graph = ctx.build()
        bound = bound_parameter_metadata(graph.parameter(param.id))
        assert bound.requires_grad is False


def test_linear_module_builds_parameter_bound_graph() -> None:
    def make_model(ctx: GraphCompositionContext) -> Linear:
        return Linear(512, 256)

    graph = build_graph(
        make_model,
        (ValueMetadata((2, 128, 512)),),
    )
    assert len(graph.parameters) == 1
    weight = graph.parameter(next(iter(graph.parameters)))
    assert weight.metadata.shape == (512, 256)

    op = graph.operation(next(iter(graph.operations)))
    assert op.operation_family == "linear_matmul"
    assert len(op.parameter_ports) == 1
    assert op.parameter_ports[0].value_kind is ValueKind.PARAMETER
