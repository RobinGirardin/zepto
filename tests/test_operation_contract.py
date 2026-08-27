from dataclasses import replace

import pytest

from zepto.analysis.resolved import ResolvedValue
from zepto.graph import ComposeError, GraphId, EdgeId
from zepto.compose import (
    Module,
    Tensor,
    compose_graph,
)
from zepto.semantic import (
    AliasSpec,
    Add,
    BackwardSpec,
    DeclarationValidator,
    EstimationContext,
    Identity,
    InvocationValidator,
    MatMul,
    Maximum,
    Minimum,
    Multiply,
    Operation,
    OperationError,
    OperationResult,
    Port,
    Reshape,
    ResourceEvent,
    ResourceEventKind,
    Split,
    Subtract,
    TensorRole,
    Transpose,
    DType,
)
from zepto.semantic.operations.helpers import (
    GRAD_LEFT,
    GRAD_RIGHT,
    GRAD_RIGHT_UNREDUCED,
    broadcast_reduction_flops,
)


def _resolved(tensor: Tensor, role: TensorRole = TensorRole.ACTIVATION) -> ResolvedValue:
    return ResolvedValue(tensor=tensor, role=role, dtype=tensor.dtype or DType.FP32)


def _estimation(**ports: Tensor) -> EstimationContext:
    return EstimationContext(
        port_values=tuple((name, _resolved(tensor)) for name, tensor in ports.items())
    )


class _IdentityOperation(Operation):
    @property
    def family(self):
        return "custom_identity"

    @property
    def input_ports(self):
        return (Port("input"),)

    @property
    def output_ports(self):
        return (Port("output"),)

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
            return operation(value)

    graph = compose_graph(Custom(), (Tensor(shape=(4,)),))
    node = graph.node(graph.node_order[0])
    assert node.declaration is operation
    assert node.result.aliases == (AliasSpec("input"),)


def test_operation_call_requires_active_composition_context():
    value = Tensor(shape=(1,), _edge_id=EdgeId(GraphId.new(), 0))
    with pytest.raises(ComposeError):
        Add()(value)


def test_incomplete_operation_is_rejected_at_construction():
    class Incomplete(Operation):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_matmul_uses_two_flops_per_multiply_add():
    class MatmulModule(Module):
        def forward(self, left, right):
            return MatMul()(left, right)

    graph = compose_graph(
        MatmulModule(), (Tensor(shape=(2, 3)), Tensor(shape=(3, 5)))
    )
    operation = graph.node(graph.node_order[0])
    result = operation.result
    output = graph.edge(operation.output_edges[0]).tensor
    assert result is not None
    assert operation.declaration is not None
    assert (
        operation.declaration.forward_flops(
            _estimation(
                left=Tensor(shape=(2, 3)),
                right=Tensor(shape=(3, 5)),
                output=output,
            ),
        )
        == 2 * 2 * 3 * 5
    )


def test_identity_is_an_alias_and_not_a_composite_module():
    class IdentityModule(Module):
        def forward(self, value):
            return Identity()(value)

    graph = compose_graph(IdentityModule(), (Tensor(shape=(2,)),))
    operation = graph.node(graph.node_order[0])
    assert operation.result.aliases == (AliasSpec("input"),)
    assert graph.edge(operation.input_edges[0]).storage_id == graph.edge(
        operation.output_edges[0]
    ).storage_id


def test_saved_backward_names_are_resolved_to_graph_port_references():
    class MaximumModule(Module):
        def forward(self, left, right):
            return Maximum()(left, right)

    graph = compose_graph(
        MaximumModule(),
        (
            Tensor(shape=(2,), requires_grad=True),
            Tensor(shape=(2,), requires_grad=True),
        ),
    )
    operation = graph.node(graph.node_order[0])

    assert operation.result.saved_for_backward == ("left", "right")
    assert len(operation.saved_for_backward) == 2
    for saved, port_name in zip(
        operation.saved_for_backward, ("left", "right"), strict=True
    ):
        assert saved.node_id == operation.id
        assert saved.port_name == port_name
        assert saved.role == "input"


def test_tensor_requires_concrete_non_negative_dimensions():
    with pytest.raises(ValueError, match="non-negative integers"):
        Tensor(shape=(2.5,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative integers"):
        Tensor(shape=(-1,))


def test_shape_scenarios_are_represented_by_rebuilt_graphs():
    class IdentityModule(Module):
        def forward(self, value):
            return Identity()(value)

    first = compose_graph(IdentityModule(), (Tensor(shape=(8, 128)),))
    second = compose_graph(IdentityModule(), (Tensor(shape=(32, 128)),))

    assert first.edge(first.inputs[0]).tensor.shape == (8, 128)
    assert second.edge(second.inputs[0]).tensor.shape == (32, 128)
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
    left = Tensor(shape=(2, 3), requires_grad=left_requires_grad)
    right = Tensor(shape=(2, 3), requires_grad=right_requires_grad)
    operation = Multiply()
    inferred = operation.infer_result((left, right))
    context = _estimation(left=left, right=right, output=inferred.outputs[0])

    assert not isinstance(operation, Add)
    assert inferred.outputs[0].requires_grad is (
        left_requires_grad or right_requires_grad
    )
    assert inferred.result.saved_for_backward == saved
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
    left = Tensor(shape=(2, 3), requires_grad=left_requires_grad)
    right = Tensor(shape=(3, 5), requires_grad=right_requires_grad)
    operation = MatMul()
    inferred = operation.infer_result((left, right))
    context = _estimation(left=left, right=right, output=inferred.outputs[0])
    requires_grad = left_requires_grad or right_requires_grad

    assert inferred.outputs[0].requires_grad is requires_grad
    assert inferred.result.saved_for_backward == saved
    assert operation.forward_flops(context) == 60
    assert operation.backward_flops(context) == expected_backward


@pytest.mark.parametrize("operation", (Add(), Multiply()))
def test_elementwise_operations_reject_incompatible_broadcast_shapes(operation):
    with pytest.raises(ValueError, match=f"{operation.family} shapes"):
        operation.infer_result((Tensor(shape=(2, 3)), Tensor(shape=(4, 3))))


def test_multiply_broadcast_tensor_matches_add():
    left = Tensor(shape=(2, 1, 3), requires_grad=True)
    right = Tensor(shape=(3,), requires_grad=False)

    add_inferred = Add().infer_result((left, right))
    multiply_inferred = Multiply().infer_result((left, right))

    assert multiply_inferred.outputs[0].shape == add_inferred.outputs[0].shape
    assert multiply_inferred.outputs[0].requires_grad == add_inferred.outputs[0].requires_grad


@pytest.mark.parametrize(
    ("operation", "inputs"),
    (
        (Add(), (Tensor(shape=(2,), requires_grad=True),) * 2),
        (Identity(), (Tensor(shape=(2,), requires_grad=True),)),
        (Reshape((4,)), (Tensor(shape=(2, 2), requires_grad=True),)),
        (Transpose((1, 0)), (Tensor(shape=(2, 3), requires_grad=True),)),
        (Split((1, 1)), (Tensor(shape=(2,), requires_grad=True),)),
    ),
)
def test_operations_without_backward_state_save_nothing(operation, inputs):
    inferred = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, (), inferred.outputs) == ()
    assert inferred.result.saved_for_backward == ()


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
        Tensor(shape=(3,), requires_grad=left_requires_grad),
        Tensor(shape=(3,), requires_grad=right_requires_grad),
    )
    inferred = operation.infer_result(inputs)

    assert operation.saved_for_backward(inputs, (), inferred.outputs) == saved
    assert inferred.result.saved_for_backward == saved


@pytest.mark.parametrize("operation", (Multiply(), MatMul()))
def test_saved_selection_uses_only_the_opposite_operand(operation):
    left = Tensor(shape=(2, 2), requires_grad=True)
    right = Tensor(shape=(2, 2), requires_grad=False)
    inferred = operation.infer_result((left, right))

    assert operation.saved_for_backward((left, right), (), inferred.outputs) == ("right",)


def test_hand_built_result_cannot_bypass_the_saved_selection():
    left = Tensor(shape=(2, 2), requires_grad=True)
    right = Tensor(shape=(2, 2), requires_grad=False)
    operation = MatMul()
    inferred = operation.infer_result((left, right))
    tampered = replace(inferred.result, saved_for_backward=("left", "right"))

    with pytest.raises(OperationError, match="saved_for_backward selection"):
        operation.validate_result(
            (left, right), (), inferred.outputs, inferred.auxiliary_outputs, tampered
        )


def test_declaration_validator_reports_bad_family() -> None:
    class BadFamily(Operation):
        @property
        def family(self):
            return ""

        @property
        def input_ports(self):
            return (Port("input"),)

        @property
        def output_ports(self):
            return (Port("output"),)

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
        input_ports=(Port("input"),),
        parameter_ports=(),
        output_ports=(Port("output"),),
        backward=BackwardSpec(
            supported=True,
            saved_for_backward=("missing",),
        ),
        saved_for_backward=lambda inputs, parameters=(), outputs=(): ("missing",),
    )
    inputs = (Tensor(shape=(2,)),)
    result = OperationResult(saved_for_backward=("missing",))
    with pytest.raises(OperationError, match="unknown operation port"):
        InvocationValidator().validate_saved_state(
            operation, inputs, (), inputs, result
        )


def test_invocation_validator_reports_bad_view_alias() -> None:
    class ViewMismatch(Operation):
        @property
        def family(self):
            return "view_mismatch"

        @property
        def input_ports(self):
            return (Port("input"),)

        @property
        def output_ports(self):
            return (Port("output"),)

        def infer_outputs(self, inputs, parameters=()):
            return (Tensor(shape=(4,)),)

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
    inputs = (Tensor(shape=(2, 3)),)
    outputs = (Tensor(shape=(2, 4)),)
    result = OperationResult(aliases=(AliasSpec("input"),))
    with pytest.raises(OperationError, match="View aliases must preserve"):
        InvocationValidator().validate_aliases(operation, inputs, outputs, result)


def test_validate_result_does_not_invoke_resource_events() -> None:
    class RaisingEvents(Operation):
        @property
        def family(self):
            return "raising_events"

        @property
        def input_ports(self):
            return (Port("input"),)

        @property
        def output_ports(self):
            return (Port("output"),)

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
    inputs = (Tensor(shape=(2,)),)
    result = OperationResult()
    operation.validate_result(inputs, (), inputs, (), result)


def test_subtract_family_and_backward_contract() -> None:
    operation = Subtract()
    assert operation.family == "subtract"
    assert operation.backward.gradient_outputs == ("left", "right")
    inferred = operation.infer_result(
        (Tensor(shape=(2,)), Tensor(shape=(2,)))
    )
    assert inferred.result.saved_for_backward == ()


def test_gradient_accumulation_without_gradient_event_kind() -> None:
    events = (
        ResourceEvent(ResourceEventKind.ALLOCATE, "grad:0"),
        ResourceEvent(ResourceEventKind.PERSIST, "grad:0"),
        ResourceEvent(ResourceEventKind.RELEASE, "grad:0"),
    )
    assert not hasattr(ResourceEventKind, "GRADIENT")
    assert TensorRole.GRADIENT is TensorRole.GRADIENT
    assert {event.kind for event in events} == {
        ResourceEventKind.ALLOCATE,
        ResourceEventKind.PERSIST,
        ResourceEventKind.RELEASE,
    }


class TestBroadcastReductionFlops:
    def test_no_reduction_when_shapes_match(self) -> None:
        operand = Tensor(shape=(32, 128, 512))
        output = Tensor(shape=(32, 128, 512))
        assert broadcast_reduction_flops(operand, output) == 0

    def test_reduction_for_broadcast_bias(self) -> None:
        operand = Tensor(shape=(512,))
        output = Tensor(shape=(32, 128, 512))
        assert broadcast_reduction_flops(operand, output) == 32 * 128 * 512 - 512


class TestBroadcastBackwardFlops:
    def test_add_reduction_only_for_broadcast_operand(self) -> None:
        x = Tensor(shape=(32, 128, 512), requires_grad=True)
        bias = Tensor(shape=(512,), requires_grad=True)
        operation = Add()
        inferred = operation.infer_result((x, bias))
        context = _estimation(left=x, right=bias, output=inferred.outputs[0])
        assert operation.backward_flops(context) == 32 * 128 * 512 - 512

    def test_subtract_includes_negation(self) -> None:
        x = Tensor(shape=(32, 128, 512), requires_grad=True)
        bias = Tensor(shape=(512,), requires_grad=True)
        operation = Subtract()
        inferred = operation.infer_result((x, bias))
        context = _estimation(left=x, right=bias, output=inferred.outputs[0])
        assert operation.backward_flops(context) == (32 * 128 * 512 - 512) + 512

    def test_multiply_vjp_and_reduction(self) -> None:
        x = Tensor(shape=(32, 128, 512), requires_grad=True)
        bias = Tensor(shape=(512,), requires_grad=True)
        operation = Multiply()
        inferred = operation.infer_result((x, bias))
        context = _estimation(left=x, right=bias, output=inferred.outputs[0])
        output_size = 32 * 128 * 512
        reduction = output_size - 512
        assert operation.backward_flops(context) == output_size + output_size + reduction


class TestMatMulBatchBroadcast:
    def test_linear_layer_shape(self) -> None:
        left = Tensor(shape=(32, 128, 512))
        right = Tensor(shape=(512, 64))
        inferred = MatMul().infer_result((left, right))
        assert inferred.outputs[0].shape == (32, 128, 64)

    def test_singleton_batch_dims(self) -> None:
        left = Tensor(shape=(8, 1, 4, 4))
        right = Tensor(shape=(1, 5, 4, 4))
        inferred = MatMul().infer_result((left, right))
        assert inferred.outputs[0].shape == (8, 5, 4, 4)

    def test_incompatible_batch_rejected(self) -> None:
        with pytest.raises(ValueError, match="matmul shapes"):
            MatMul().infer_result(
                (Tensor(shape=(2, 4, 4)), Tensor(shape=(3, 4, 4)))
            )

    def test_incompatible_contraction_rejected(self) -> None:
        with pytest.raises(ValueError, match="contracting"):
            MatMul().infer_result(
                (Tensor(shape=(4, 3)), Tensor(shape=(5, 4)))
            )

    def test_backward_flops_includes_batch_reduction(self) -> None:
        left = Tensor(shape=(32, 128, 512), requires_grad=True)
        right = Tensor(shape=(512, 64), requires_grad=True)
        operation = MatMul()
        inferred = operation.infer_result((left, right))
        context = _estimation(left=left, right=right, output=inferred.outputs[0])
        batch = 32
        gemm = 2 * batch * 128 * 64 * 512
        reduction = 32 * 512 * 64 - 512 * 64
        assert operation.backward_flops(context) == gemm + reduction + gemm + 0


class TestAuxiliaryPorts:
    def test_multiply_declares_four_aux_ports(self) -> None:
        operation = Multiply()
        assert len(operation.auxiliary_ports()) == 4

    def test_active_aux_ports_omit_unreduced_without_broadcast(self) -> None:
        left = Tensor(shape=(2, 3), requires_grad=True)
        right = Tensor(shape=(2, 3), requires_grad=True)
        inferred = Multiply().infer_result((left, right))
        assert GRAD_LEFT in inferred.result.active_auxiliary_ports
        assert GRAD_RIGHT in inferred.result.active_auxiliary_ports
        assert GRAD_RIGHT_UNREDUCED not in inferred.result.active_auxiliary_ports

    def test_active_aux_includes_unreduced_for_multiply_bias(self) -> None:
        x = Tensor(shape=(32, 128, 512), requires_grad=True)
        bias = Tensor(shape=(512,), requires_grad=True)
        inferred = Multiply().infer_result((x, bias))
        assert GRAD_RIGHT in inferred.result.active_auxiliary_ports
        assert GRAD_RIGHT_UNREDUCED in inferred.result.active_auxiliary_ports
        assert GRAD_LEFT not in inferred.result.active_auxiliary_ports

    def test_infer_result_validates_auxiliary_arity(self) -> None:
        class BadAux(Operation):
            @property
            def family(self):
                return "bad_aux"

            @property
            def input_ports(self):
                return (Port("left"), Port("right"))

            @property
            def output_ports(self):
                return (Port("output"),)

            def auxiliary_ports(self):
                return (Port("grad_left"),)

            def infer_auxiliary_outputs(self, inputs, parameters=(), outputs=()):
                return ()

            def infer_outputs(self, inputs, parameters=()):
                return (Tensor(shape=(2,)),)

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
            BadAux().infer_result((Tensor(shape=(2,)), Tensor(shape=(2,))))

    def test_identity_has_no_aux_ports(self) -> None:
        assert Identity().auxiliary_ports() == ()


def test_gradient_aux_tensor_matches_operand_shape() -> None:
    x = Tensor(shape=(32, 128, 512), requires_grad=True)
    bias = Tensor(shape=(512,), requires_grad=True)
    inferred = Multiply().infer_result((x, bias))
    grad_right_index = next(
        i
        for i, port in enumerate(Multiply().auxiliary_ports())
        if port.name == GRAD_RIGHT
    )
    assert inferred.auxiliary_outputs[grad_right_index].shape == (512,)


def test_active_aux_must_be_declared_port() -> None:
    class BadActive(Operation):
        @property
        def family(self):
            return "bad_active"

        @property
        def input_ports(self):
            return (Port("input"),)

        @property
        def output_ports(self):
            return (Port("output"),)

        def auxiliary_ports(self):
            return (Port("grad_input"),)

        def infer_auxiliary_outputs(self, inputs, parameters=(), outputs=()):
            return (Tensor(shape=(2,)),)

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
        BadActive().infer_result((Tensor(shape=(2,)),))
