from dataclasses import replace

import pytest

from zepto.core import (
    AliasSpec,
    Add,
    BackwardSpec,
    DeclarationValidator,
    EstimationContext,
    GraphCompositionContext,
    Identity,
    InvocationValidator,
    Materialization,
    Module,
    MatMul,
    Maximum,
    Minimum,
    Multiply,
    Operation,
    OperationError,
    OperationResult,
    PortSpec,
    Reshape,
    ResourceEvent,
    ResourceEventKind,
    Split,
    Subtract,
    TensorMetadata,
    TensorRole,
    Transpose,
    build_graph,
    identity,
    matmul,
    maximum,
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


    def forward_flops(self, context):
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
    class MaximumModule(Module):
        def forward(self, left, right):
            return maximum(left, right)

    graph = build_graph(
        MaximumModule(),
        (
            TensorMetadata((2,), requires_grad=True),
            TensorMetadata((2,), requires_grad=True),
        ),
    )
    operation = graph.operation(graph.operations[0])

    assert operation.result.saved_for_backward == ("left", "right")
    assert len(operation.saved_for_backward) == 2
    for saved, port_name in zip(
        operation.saved_for_backward, ("left", "right"), strict=True
    ):
        assert saved.operation_id == operation.id
        assert saved.port_name == port_name
        assert saved.role == "input"


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
    left = TensorMetadata((2, 3), requires_grad=left_requires_grad)
    right = TensorMetadata((2, 3), requires_grad=right_requires_grad)
    operation = Multiply()
    result = operation.infer_result((left, right))
    context = EstimationContext(
        port_metadata=(
            ("left", left),
            ("right", right),
            ("output", result.outputs[0]),
        )
    )

    assert not isinstance(operation, Add)
    assert result.outputs[0].requires_grad is (
        left_requires_grad or right_requires_grad
    )
    assert result.saved_for_backward == saved
    assert operation.backward_flops(context) == factor * 6


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
    left = TensorMetadata((2, 3), requires_grad=left_requires_grad)
    right = TensorMetadata((3, 5), requires_grad=right_requires_grad)
    operation = MatMul()
    result = operation.infer_result((left, right))
    context = EstimationContext(
        port_metadata=(
            ("left", left),
            ("right", right),
            ("output", result.outputs[0]),
        )
    )
    requires_grad = left_requires_grad or right_requires_grad

    assert result.outputs[0].requires_grad is requires_grad
    assert result.saved_for_backward == saved
    assert operation.forward_flops(context) == 60
    assert operation.backward_flops(context) == expected_backward


@pytest.mark.parametrize("operation", (Add(), Multiply()))
def test_elementwise_operations_reject_incompatible_broadcast_shapes(operation):
    with pytest.raises(ValueError, match=f"{operation.family} shapes"):
        operation.infer_result((TensorMetadata((2, 3)), TensorMetadata((4, 3))))


def test_multiply_broadcast_metadata_matches_add():
    left = TensorMetadata((2, 1, 3), requires_grad=True)
    right = TensorMetadata((3,), requires_grad=False)

    add_result = Add().infer_result((left, right))
    multiply_result = Multiply().infer_result((left, right))

    assert multiply_result.outputs == add_result.outputs


@pytest.mark.parametrize(
    ("operation", "inputs"),
    (
        (Add(), (TensorMetadata((2,), requires_grad=True),) * 2),
        (Identity(), (TensorMetadata((2,), requires_grad=True),)),
        (Reshape((4,)), (TensorMetadata((2, 2), requires_grad=True),)),
        (Transpose((1, 0)), (TensorMetadata((2, 3), requires_grad=True),)),
        (Split((1, 1)), (TensorMetadata((2,), requires_grad=True),)),
    ),
)
def test_operations_without_backward_state_save_nothing(operation, inputs):
    result = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, result.outputs) == ()
    assert result.saved_for_backward == ()


@pytest.mark.parametrize("operation", (Maximum(), Minimum()))
@pytest.mark.parametrize(
    ("left_requires_grad", "right_requires_grad", "saved"),
    (
        (False, False, ()),
        (True, False, ("left", "right")),
        (False, True, ("left", "right")),
        (True, True, ("left", "right")),
    ),
)
def test_extremum_saves_both_operands_for_the_backward_mask(
    operation,
    left_requires_grad,
    right_requires_grad,
    saved,
):
    inputs = (
        TensorMetadata((3,), requires_grad=left_requires_grad),
        TensorMetadata((3,), requires_grad=right_requires_grad),
    )
    result = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, result.outputs) == saved
    assert result.saved_for_backward == saved


@pytest.mark.parametrize("operation", (Multiply(), MatMul()))
def test_saved_selection_uses_only_the_opposite_operand(operation):
    left = TensorMetadata((2, 2), requires_grad=True)
    right = TensorMetadata((2, 2), requires_grad=False)
    result = operation.infer_result((left, right))

    assert operation.saved_for_backward((left, right), result.outputs) == ("right",)


def test_hand_built_result_cannot_bypass_the_saved_selection():
    left = TensorMetadata((2, 2), requires_grad=True)
    right = TensorMetadata((2, 2), requires_grad=False)
    operation = MatMul()
    result = operation.infer_result((left, right))
    tampered = replace(result, saved_for_backward=("left", "right"))

    with pytest.raises(OperationError, match="saved_for_backward selection"):
        operation.validate_result((left, right), tampered)


def test_declaration_validator_reports_bad_family() -> None:
    class BadFamily(Operation):
        @property
        def family(self):
            return ""

        @property
        def input_ports(self):
            return (PortSpec("input"),)

        @property
        def output_ports(self):
            return (PortSpec("output"),)

        def infer_outputs(self, inputs):
            return inputs

        def saved_for_backward(self, inputs, outputs):
            return ()

        @property
        def backward(self):
            return BackwardSpec()

        def forward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            return ()

    with pytest.raises(OperationError, match="family cannot be empty"):
        DeclarationValidator().validate_family(BadFamily())


def test_invocation_validator_reports_unknown_saved_port() -> None:
    from types import SimpleNamespace

    operation = SimpleNamespace(
        family="bad_saved",
        input_ports=(PortSpec("input"),),
        output_ports=(PortSpec("output"),),
        backward=BackwardSpec(
            supported=True,
            saved_for_backward=("missing",),
        ),
        saved_for_backward=lambda inputs, outputs: ("missing",),
    )
    inputs = (TensorMetadata((2,)),)
    result = OperationResult(
        outputs=inputs,
        saved_for_backward=("missing",),
    )
    with pytest.raises(OperationError, match="unknown operation port"):
        InvocationValidator().validate_saved_state(operation, inputs, result)


def test_invocation_validator_reports_bad_view_alias() -> None:
    class ViewMismatch(Operation):
        @property
        def family(self):
            return "view_mismatch"

        @property
        def input_ports(self):
            return (PortSpec("input"),)

        @property
        def output_ports(self):
            return (PortSpec("output"),)

        def infer_outputs(self, inputs):
            return (TensorMetadata((4,)),)

        def output_aliases(self):
            return (AliasSpec("input"),)

        def saved_for_backward(self, inputs, outputs):
            return ()

        @property
        def backward(self):
            return BackwardSpec()

        def forward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            return ()

    operation = ViewMismatch()
    inputs = (TensorMetadata((2, 3)),)
    result = OperationResult(
        outputs=(TensorMetadata((2, 4)),),
        aliases=(AliasSpec("input"),),
    )
    with pytest.raises(OperationError, match="View aliases must preserve"):
        InvocationValidator().validate_aliases(operation, inputs, result)


def test_validate_result_does_not_invoke_resource_events() -> None:
    class RaisingEvents(Operation):
        @property
        def family(self):
            return "raising_events"

        @property
        def input_ports(self):
            return (PortSpec("input"),)

        @property
        def output_ports(self):
            return (PortSpec("output"),)

        def infer_outputs(self, inputs):
            return inputs

        def saved_for_backward(self, inputs, outputs):
            return ()

        @property
        def backward(self):
            return BackwardSpec()

        def forward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            raise RuntimeError("resource_events must not run during validation")

    operation = RaisingEvents()
    inputs = (TensorMetadata((2,)),)
    result = OperationResult(outputs=inputs)
    operation.validate_result(inputs, result)


def test_subtract_family_and_backward_contract() -> None:
    operation = Subtract()
    assert operation.family == "subtract"
    assert operation.backward.gradient_outputs == ("left", "right")
    result = operation.infer_result(
        (TensorMetadata((2,)), TensorMetadata((2,)))
    )
    assert result.saved_for_backward == ()


def test_gradient_accumulation_without_gradient_event_kind() -> None:
    metadata = TensorMetadata((4,), role=TensorRole.GRADIENT, requires_grad=False)
    events = (
        ResourceEvent(ResourceEventKind.ALLOCATE, "grad:0"),
        ResourceEvent(ResourceEventKind.PERSIST, "grad:0"),
        ResourceEvent(ResourceEventKind.RELEASE, "grad:0"),
    )
    assert not hasattr(ResourceEventKind, "GRADIENT")
    assert metadata.role is TensorRole.GRADIENT
    assert {event.kind for event in events} == {
        ResourceEventKind.ALLOCATE,
        ResourceEventKind.PERSIST,
        ResourceEventKind.RELEASE,
    }
