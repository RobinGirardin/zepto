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
    ValueMetadata,
    TensorRole,
    Transpose,
    build_graph,
    identity,
    matmul,
    maximum,
)
from zepto.core.operation.helpers import (
    GRAD_LEFT,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    broadcast_reduction_flops,
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

    def infer_outputs(self, inputs, parameters=()):
        return inputs

    def output_aliases(self):
        return (AliasSpec("input"),)

    def saved_for_backward(self, inputs, parameters=(), outputs=()):
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

    graph = build_graph(Custom(), (ValueMetadata((4,)),))
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
        MatmulModule(), (ValueMetadata((2, 3)), ValueMetadata((3, 5)))
    )
    operation = graph.operation(graph.operations[0])
    result = operation.result
    assert result is not None
    assert operation.declaration is not None
    assert (
        operation.declaration.forward_flops(
            EstimationContext(
                port_metadata=(
                    ("left", ValueMetadata((2, 3))),
                    ("right", ValueMetadata((3, 5))),
                    ("output", result.outputs[0]),
                )
            ),
        )
        == 2 * 2 * 3 * 5
    )


def test_identity_is_an_alias_and_not_a_composite_module():
    class Identity(Module):
        def forward(self, value):
            return identity(value)

    graph = build_graph(Identity(), (ValueMetadata((2,)),))
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
            ValueMetadata((2,), requires_grad=True),
            ValueMetadata((2,), requires_grad=True),
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
        ValueMetadata((2.5,))

    with pytest.raises(ValueError, match="non-negative integers"):
        ValueMetadata((-1,))


def test_shape_scenarios_are_represented_by_rebuilt_graphs():
    class IdentityModule(Module):
        def forward(self, value):
            return identity(value)

    first = build_graph(IdentityModule(), (ValueMetadata((8, 128)),))
    second = build_graph(IdentityModule(), (ValueMetadata((32, 128)),))

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
    left = ValueMetadata((2, 3), requires_grad=left_requires_grad)
    right = ValueMetadata((2, 3), requires_grad=right_requires_grad)
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
    left = ValueMetadata((2, 3), requires_grad=left_requires_grad)
    right = ValueMetadata((3, 5), requires_grad=right_requires_grad)
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
        operation.infer_result((ValueMetadata((2, 3)), ValueMetadata((4, 3))))


def test_multiply_broadcast_metadata_matches_add():
    left = ValueMetadata((2, 1, 3), requires_grad=True)
    right = ValueMetadata((3,), requires_grad=False)

    add_result = Add().infer_result((left, right))
    multiply_result = Multiply().infer_result((left, right))

    assert multiply_result.outputs == add_result.outputs


@pytest.mark.parametrize(
    ("operation", "inputs"),
    (
        (Add(), (ValueMetadata((2,), requires_grad=True),) * 2),
        (Identity(), (ValueMetadata((2,), requires_grad=True),)),
        (Reshape((4,)), (ValueMetadata((2, 2), requires_grad=True),)),
        (Transpose((1, 0)), (ValueMetadata((2, 3), requires_grad=True),)),
        (Split((1, 1)), (ValueMetadata((2,), requires_grad=True),)),
    ),
)
def test_operations_without_backward_state_save_nothing(operation, inputs):
    result = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, (), result.outputs) == ()
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
        ValueMetadata((3,), requires_grad=left_requires_grad),
        ValueMetadata((3,), requires_grad=right_requires_grad),
    )
    result = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, (), result.outputs) == saved
    assert result.saved_for_backward == saved


@pytest.mark.parametrize("operation", (Multiply(), MatMul()))
def test_saved_selection_uses_only_the_opposite_operand(operation):
    left = ValueMetadata((2, 2), requires_grad=True)
    right = ValueMetadata((2, 2), requires_grad=False)
    result = operation.infer_result((left, right))

    assert operation.saved_for_backward((left, right), (), result.outputs) == ("right",)


def test_hand_built_result_cannot_bypass_the_saved_selection():
    left = ValueMetadata((2, 2), requires_grad=True)
    right = ValueMetadata((2, 2), requires_grad=False)
    operation = MatMul()
    result = operation.infer_result((left, right))
    tampered = replace(result, saved_for_backward=("left", "right"))

    with pytest.raises(OperationError, match="saved_for_backward selection"):
        operation.validate_result((left, right), (), tampered)


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

        def infer_outputs(self, inputs, parameters=()):
            return inputs

        def saved_for_backward(self, inputs, parameters=(), outputs=()):
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
        parameter_ports=(),
        output_ports=(PortSpec("output"),),
        backward=BackwardSpec(
            supported=True,
            saved_for_backward=("missing",),
        ),
        saved_for_backward=lambda inputs, parameters=(), outputs=(): ("missing",),
    )
    inputs = (ValueMetadata((2,)),)
    result = OperationResult(
        outputs=inputs,
        saved_for_backward=("missing",),
    )
    with pytest.raises(OperationError, match="unknown operation port"):
        InvocationValidator().validate_saved_state(operation, inputs, (), result)


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

        def infer_outputs(self, inputs, parameters=()):
            return (ValueMetadata((4,)),)

        def output_aliases(self):
            return (AliasSpec("input"),)

        def saved_for_backward(self, inputs, parameters=(), outputs=()):
            return ()

        @property
        def backward(self):
            return BackwardSpec()

        def forward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            return ()

    operation = ViewMismatch()
    inputs = (ValueMetadata((2, 3)),)
    result = OperationResult(
        outputs=(ValueMetadata((2, 4)),),
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

        def infer_outputs(self, inputs, parameters=()):
            return inputs

        def saved_for_backward(self, inputs, parameters=(), outputs=()):
            return ()

        @property
        def backward(self):
            return BackwardSpec()

        def forward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            raise RuntimeError("resource_events must not run during validation")

    operation = RaisingEvents()
    inputs = (ValueMetadata((2,)),)
    result = OperationResult(outputs=inputs)
    operation.validate_result(inputs, (), result)


def test_subtract_family_and_backward_contract() -> None:
    operation = Subtract()
    assert operation.family == "subtract"
    assert operation.backward.gradient_outputs == ("left", "right")
    result = operation.infer_result(
        (ValueMetadata((2,)), ValueMetadata((2,)))
    )
    assert result.saved_for_backward == ()


def test_gradient_accumulation_without_gradient_event_kind() -> None:
    metadata = ValueMetadata((4,), role=TensorRole.GRADIENT, requires_grad=False)
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


class TestBroadcastReductionFlops:
    def test_no_reduction_when_shapes_match(self) -> None:
        operand = ValueMetadata((32, 128, 512))
        output = ValueMetadata((32, 128, 512))
        assert broadcast_reduction_flops(operand, output) == 0

    def test_reduction_for_broadcast_bias(self) -> None:
        operand = ValueMetadata((512,))
        output = ValueMetadata((32, 128, 512))
        assert broadcast_reduction_flops(operand, output) == 32 * 128 * 512 - 512


class TestBroadcastBackwardFlops:
    def test_add_reduction_only_for_broadcast_operand(self) -> None:
        x = ValueMetadata((32, 128, 512), requires_grad=True)
        bias = ValueMetadata((512,), requires_grad=True)
        operation = Add()
        result = operation.infer_result((x, bias))
        context = EstimationContext(
            port_metadata=(
                ("left", x),
                ("right", bias),
                ("output", result.outputs[0]),
            )
        )
        assert operation.backward_flops(context) == 32 * 128 * 512 - 512

    def test_subtract_includes_negation(self) -> None:
        x = ValueMetadata((32, 128, 512), requires_grad=True)
        bias = ValueMetadata((512,), requires_grad=True)
        operation = Subtract()
        result = operation.infer_result((x, bias))
        context = EstimationContext(
            port_metadata=(
                ("left", x),
                ("right", bias),
                ("output", result.outputs[0]),
            )
        )
        assert operation.backward_flops(context) == (32 * 128 * 512 - 512) + 512

    def test_multiply_vjp_and_reduction(self) -> None:
        x = ValueMetadata((32, 128, 512), requires_grad=True)
        bias = ValueMetadata((512,), requires_grad=True)
        operation = Multiply()
        result = operation.infer_result((x, bias))
        context = EstimationContext(
            port_metadata=(
                ("left", x),
                ("right", bias),
                ("output", result.outputs[0]),
            )
        )
        output_size = 32 * 128 * 512
        reduction = output_size - 512
        assert operation.backward_flops(context) == output_size + output_size + reduction


class TestMatMulBatchBroadcast:
    def test_linear_layer_shape(self) -> None:
        left = ValueMetadata((32, 128, 512))
        right = ValueMetadata((512, 64))
        result = MatMul().infer_result((left, right))
        assert result.outputs[0].shape == (32, 128, 64)

    def test_singleton_batch_dims(self) -> None:
        left = ValueMetadata((8, 1, 4, 4))
        right = ValueMetadata((1, 5, 4, 4))
        result = MatMul().infer_result((left, right))
        assert result.outputs[0].shape == (8, 5, 4, 4)

    def test_incompatible_batch_rejected(self) -> None:
        with pytest.raises(ValueError, match="matmul shapes"):
            MatMul().infer_result(
                (ValueMetadata((2, 4, 4)), ValueMetadata((3, 4, 4)))
            )

    def test_incompatible_contraction_rejected(self) -> None:
        with pytest.raises(ValueError, match="contracting"):
            MatMul().infer_result(
                (ValueMetadata((4, 3)), ValueMetadata((5, 4)))
            )

    def test_backward_flops_includes_batch_reduction(self) -> None:
        left = ValueMetadata((32, 128, 512), requires_grad=True)
        right = ValueMetadata((512, 64), requires_grad=True)
        operation = MatMul()
        result = operation.infer_result((left, right))
        context = EstimationContext(
            port_metadata=(
                ("left", left),
                ("right", right),
                ("output", result.outputs[0]),
            )
        )
        batch = 32
        gemm = 2 * batch * 128 * 64 * 512
        reduction = 32 * 512 * 64 - 512 * 64
        assert operation.backward_flops(context) == gemm + reduction + gemm + 0


class TestAuxiliaryPorts:
    def test_multiply_declares_four_aux_ports(self) -> None:
        operation = Multiply()
        assert len(operation.auxiliary_ports()) == 4

    def test_active_aux_ports_omit_unreduced_without_broadcast(self) -> None:
        left = ValueMetadata((2, 3), requires_grad=True)
        right = ValueMetadata((2, 3), requires_grad=True)
        result = Multiply().infer_result((left, right))
        assert GRAD_LEFT in result.active_auxiliary_ports
        assert GRAD_RIGHT in result.active_auxiliary_ports
        assert GRAD_RIGHT_UNREDUCED not in result.active_auxiliary_ports

    def test_active_aux_includes_unreduced_for_multiply_bias(self) -> None:
        x = ValueMetadata((32, 128, 512), requires_grad=True)
        bias = ValueMetadata((512,), requires_grad=True)
        result = Multiply().infer_result((x, bias))
        assert GRAD_RIGHT in result.active_auxiliary_ports
        assert GRAD_RIGHT_UNREDUCED in result.active_auxiliary_ports
        assert GRAD_LEFT not in result.active_auxiliary_ports

    def test_infer_result_validates_auxiliary_arity(self) -> None:
        class BadAux(Operation):
            @property
            def family(self):
                return "bad_aux"

            @property
            def input_ports(self):
                return (PortSpec("left"), PortSpec("right"))

            @property
            def output_ports(self):
                return (PortSpec("output"),)

            def auxiliary_ports(self):
                return (PortSpec("grad_left"),)

            def infer_auxiliary_outputs(self, inputs, parameters=(), outputs=()):
                return ()

            def infer_outputs(self, inputs, parameters=()):
                return (ValueMetadata((2,)),)

            def saved_for_backward(self, inputs, parameters=(), outputs=()):
                return ()

            @property
            def backward(self):
                return BackwardSpec()

            def forward_flops(self, context):
                return 0

            def resource_events(self, context, result):
                return ()

        with pytest.raises(OperationError, match="auxiliary outputs"):
            BadAux().infer_result((ValueMetadata((2,)), ValueMetadata((2,))))

    def test_identity_has_no_aux_ports(self) -> None:
        assert Identity().auxiliary_ports() == ()


def test_gradient_aux_metadata_matches_operand_shape() -> None:
    x = ValueMetadata((32, 128, 512), requires_grad=True)
    bias = ValueMetadata((512,), requires_grad=True)
    result = Multiply().infer_result((x, bias))
    grad_right_index = next(
        i
        for i, port in enumerate(Multiply().auxiliary_ports())
        if port.name == GRAD_RIGHT
    )
    assert result.auxiliary_outputs[grad_right_index].shape == (512,)


def test_active_aux_must_be_declared_port() -> None:
    class BadActive(Operation):
        @property
        def family(self):
            return "bad_active"

        @property
        def input_ports(self):
            return (PortSpec("input"),)

        @property
        def output_ports(self):
            return (PortSpec("output"),)

        def auxiliary_ports(self):
            return (PortSpec("grad_input"),)

        def infer_auxiliary_outputs(self, inputs, parameters=(), outputs=()):
            return (ValueMetadata((2,)),)

        def active_auxiliary_ports(self, inputs, parameters=(), outputs=()):
            return ("missing",)

        def infer_outputs(self, inputs, parameters=()):
            return inputs

        def saved_for_backward(self, inputs, parameters=(), outputs=()):
            return ()

        @property
        def backward(self):
            return BackwardSpec()

        def forward_flops(self, context):
            return 0

        def resource_events(self, context, result):
            return ()

    with pytest.raises(OperationError, match="undeclared auxiliary port"):
        BadActive().infer_result((ValueMetadata((2,)),))
