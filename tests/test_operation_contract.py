from dataclasses import replace

import pytest

from zepto.core import (
    AliasSpec,
    Add,
    BackwardSpec,
    EstimationContext,
    GraphCompositionContext,
    Identity,
    Module,
    MatMul,
    Multiply,
    Operation,
    OperationError,
    PortSpec,
    ReLU,
    Reshape,
    ResourceEvent,
    Split,
    TensorMetadata,
    Transpose,
    build_graph,
    identity,
    matmul,
    relu,
)


class _IdentityOperation(Operation):
    @property
    def family(self):
        return "custom_identity"

    @property
    def input_ports(self):
        return (PortSpec("input"),)

    @property
    def output_ports(self):
        return (PortSpec("output"),)

    def infer_outputs(self, inputs):
        return inputs

    def output_aliases(self):
        return (AliasSpec("input"),)

    def saved_for_backward(self, inputs, outputs):
        return ()

    @property
    def backward(self):
        return BackwardSpec(True, ("input",))


    def forward_flops(self, context, result):
        return 0

    def resource_events(self, context, result):
        return (ResourceEvent("alias", "output:0"),)


def test_complete_operation_is_validated_and_reused_by_graph_nodes():
    operation = _IdentityOperation()
    class Custom(Module):
        def forward(self, value):
            context = GraphCompositionContext.current()
            return context.apply(operation, value)

    graph = build_graph(Custom(), (TensorMetadata((4,)),))
    node = graph.operation(graph.operations[0])
    assert node.declaration is operation
    assert node.result.aliases == (AliasSpec("input"),)


def test_incomplete_operation_is_rejected_at_construction():
    class Incomplete(Operation):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_matmul_uses_two_flops_per_multiply_add():
    class MatmulModule(Module):
        def forward(self, left, right):
            return matmul(left, right)

    graph = build_graph(
        MatmulModule(), (TensorMetadata((2, 3)), TensorMetadata((3, 5)))
    )
    operation = graph.operation(graph.operations[0])
    result = operation.result
    assert result is not None
    assert operation.declaration is not None
    assert (
        operation.declaration.forward_flops(
            EstimationContext(
                port_metadata=(
                    ("left", TensorMetadata((2, 3))),
                    ("right", TensorMetadata((3, 5))),
                )
            ),
            result,
        )
        == 2 * 2 * 3 * 5
    )


def test_identity_is_an_alias_and_not_a_composite_module():
    class Identity(Module):
        def forward(self, value):
            return identity(value)

    graph = build_graph(Identity(), (TensorMetadata((2,)),))
    operation = graph.operation(graph.operations[0])
    assert operation.result.aliases == (AliasSpec("input"),)
    assert graph.tensor(operation.input_tensors[0]).storage_id == graph.tensor(
        operation.output_tensors[0]
    ).storage_id


def test_saved_backward_names_are_resolved_to_graph_port_references():
    class ReLUModule(Module):
        def forward(self, value):
            return relu(value)

    graph = build_graph(ReLUModule(), (TensorMetadata((2,)),))
    operation = graph.operation(graph.operations[0])

    assert operation.result.saved_for_backward == ("output",)
    assert len(operation.saved_for_backward) == 1
    saved = operation.saved_for_backward[0]
    assert saved.operation_id == operation.id
    assert saved.port_name == "output"
    assert saved.role == "output"


def test_tensor_metadata_requires_concrete_non_negative_dimensions():
    with pytest.raises(ValueError, match="non-negative integers"):
        TensorMetadata((2.5,))

    with pytest.raises(ValueError, match="non-negative integers"):
        TensorMetadata((-1,))


def test_shape_scenarios_are_represented_by_rebuilt_graphs():
    class IdentityModule(Module):
        def forward(self, value):
            return identity(value)

    first = build_graph(IdentityModule(), (TensorMetadata((8, 128)),))
    second = build_graph(IdentityModule(), (TensorMetadata((32, 128)),))

    assert first.tensor(first.inputs[0]).metadata.shape == (8, 128)
    assert second.tensor(second.inputs[0]).metadata.shape == (32, 128)
    assert first.id != second.id


@pytest.mark.parametrize(
    ("left_requires_grad", "right_requires_grad", "saved", "factor"),
    (
        (False, False, (), 0),
        (True, False, ("right",), 1),
        (False, True, ("left",), 1),
        (True, True, ("left", "right"), 2),
    ),
)
def test_multiply_declares_product_rule_backward_state(
    left_requires_grad,
    right_requires_grad,
    saved,
    factor,
):
    left = TensorMetadata((2, 3), require_grad=left_requires_grad)
    right = TensorMetadata((2, 3), require_grad=right_requires_grad)
    operation = Multiply()
    result = operation.infer_result((left, right))
    context = EstimationContext(
        port_metadata=(("left", left), ("right", right))
    )

    assert not isinstance(operation, Add)
    assert result.outputs[0].require_grad is (
        left_requires_grad or right_requires_grad
    )
    assert result.saved_for_backward == saved
    assert operation.backward_flops(context, result) == factor * 6


@pytest.mark.parametrize(
    ("left_requires_grad", "right_requires_grad", "saved", "expected_backward"),
    (
        (False, False, (), 0),
        (True, False, ("right",), 60),
        (False, True, ("left",), 60),
        (True, True, ("left", "right"), 120),
    ),
)
def test_matmul_gradient_contract_follows_inputs(
    left_requires_grad,
    right_requires_grad,
    saved,
    expected_backward,
):
    left = TensorMetadata((2, 3), require_grad=left_requires_grad)
    right = TensorMetadata((3, 5), require_grad=right_requires_grad)
    operation = MatMul()
    result = operation.infer_result((left, right))
    context = EstimationContext(
        port_metadata=(("left", left), ("right", right))
    )
    require_grad = left_requires_grad or right_requires_grad

    assert result.outputs[0].require_grad is require_grad
    assert result.saved_for_backward == saved
    assert operation.forward_flops(context, result) == 60
    assert operation.backward_flops(context, result) == expected_backward


@pytest.mark.parametrize("operation", (Add(), Multiply()))
def test_elementwise_operations_reject_incompatible_broadcast_shapes(operation):
    with pytest.raises(ValueError, match=f"{operation.family} shapes"):
        operation.infer_result((TensorMetadata((2, 3)), TensorMetadata((4, 3))))


def test_multiply_broadcast_metadata_matches_add():
    left = TensorMetadata((2, 1, 3), require_grad=True)
    right = TensorMetadata((3,), require_grad=False)

    add_result = Add().infer_result((left, right))
    multiply_result = Multiply().infer_result((left, right))

    assert multiply_result.outputs == add_result.outputs


@pytest.mark.parametrize(
    ("operation", "inputs"),
    (
        (Add(), (TensorMetadata((2,), require_grad=True),) * 2),
        (Identity(), (TensorMetadata((2,), require_grad=True),)),
        (Reshape((4,)), (TensorMetadata((2, 2), require_grad=True),)),
        (Transpose((1, 0)), (TensorMetadata((2, 3), require_grad=True),)),
        (Split((1, 1)), (TensorMetadata((2,), require_grad=True),)),
    ),
)
def test_operations_without_backward_state_save_nothing(operation, inputs):
    result = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, result.outputs) == ()
    assert result.saved_for_backward == ()


@pytest.mark.parametrize("require_grad", (False, True))
def test_relu_always_saves_its_output_for_the_backward_mask(require_grad):
    inputs = (TensorMetadata((3,), require_grad=require_grad),)
    operation = ReLU()
    result = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, result.outputs) == ("output",)
    assert result.saved_for_backward == ("output",)


@pytest.mark.parametrize("operation", (Multiply(), MatMul()))
def test_saved_selection_uses_only_the_opposite_operand(operation):
    left = TensorMetadata((2, 2), require_grad=True)
    right = TensorMetadata((2, 2), require_grad=False)
    result = operation.infer_result((left, right))

    assert operation.saved_for_backward((left, right), result.outputs) == ("right",)


def test_hand_built_result_cannot_bypass_the_saved_selection():
    left = TensorMetadata((2, 2), require_grad=True)
    right = TensorMetadata((2, 2), require_grad=False)
    operation = MatMul()
    result = operation.infer_result((left, right))
    tampered = replace(result, saved_for_backward=("left", "right"))

    with pytest.raises(OperationError, match="saved_for_backward selection"):
        operation.validate_result((left, right), tampered)
