import pytest

from zepto.analysis import AccountingPolicy, PrecisionPolicy, ResolvedValue
from zepto.compose.values import Parameter, Tensor
from zepto.graph.ids import GraphId, ParameterId
from zepto.semantic import (
    DType,
    PortContract,
    TensorRole,
    contract_satisfied,
)


def test_dtype_itemsize_widths() -> None:
    assert DType.UNKNOWN.itemsize is None
    assert DType.BOOL.itemsize == 1
    assert DType.INT32.itemsize == 4
    assert DType.FP16.itemsize == 2
    assert DType.BF16.itemsize == 2
    assert DType.FP32.itemsize == 4
    assert DType.FP64.itemsize == 8


def test_tensor_accepts_accounting_fields() -> None:
    tensor = Tensor(
        shape=(4, 8),
        dtype=DType.FP32,
        persistent=True,
        requires_grad=True,
    )
    assert tensor.dtype is DType.FP32
    assert tensor.persistent is True
    assert tensor.requires_grad is True
    assert not hasattr(tensor, "role")


def test_tensor_rejects_invalid_accounting_fields() -> None:
    with pytest.raises(ValueError, match="dtype must be a DType"):
        Tensor(shape=(2,), dtype="fp32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="persistent must be an explicit boolean"):
        Tensor(shape=(2,), persistent=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires_grad must be an explicit boolean"):
        Tensor(shape=(2,), requires_grad=1)  # type: ignore[arg-type]


def test_port_contract_wildcards() -> None:
    contract = PortContract()
    actual = Tensor(
        shape=(2, 3),
        semantic_type="tensor",
        dtype=DType.FP16,
    )
    assert contract_satisfied(contract, actual)

    specified = PortContract(shape=(2, 3), dtype=DType.FP32)
    assert not contract_satisfied(specified, actual)

    empty_shape = PortContract(shape=())
    assert contract_satisfied(empty_shape, actual)

    semantic = PortContract(semantic_type="weight")
    assert not contract_satisfied(semantic, actual)
    assert contract_satisfied(semantic, Tensor(shape=(2, 3), semantic_type="weight"))


def test_parameter_trainable_defaults_true() -> None:
    graph_id = GraphId.new()
    parameter_id = ParameterId(graph_id, 0)
    parameter = Parameter(shape=(8,), trainable=True).registered(parameter_id)
    assert parameter.trainable is True
    assert parameter.as_tensor().requires_grad is True


def test_precision_policy_resolve_precedence() -> None:
    policy = PrecisionPolicy(
        default_dtype=DType.FP32,
        by_semantic_type=(("weight", DType.FP16),),
        by_role=((TensorRole.GRADIENT, DType.FP32),),
    )
    explicit = Tensor(shape=(2,), dtype=DType.FP64)
    by_semantic = Tensor(shape=(2,), semantic_type="weight")
    default = Tensor(shape=(2,))

    assert policy.resolve(explicit, role=TensorRole.ACTIVATION) is DType.FP64
    assert policy.resolve(by_semantic, role=TensorRole.ACTIVATION) is DType.FP16
    assert policy.resolve(default, role=TensorRole.GRADIENT) is DType.FP32
    assert policy.resolve(default, role=TensorRole.ACTIVATION) is DType.FP32


def test_accounting_policy_bytes_for() -> None:
    policy = AccountingPolicy(PrecisionPolicy(default_dtype=DType.FP32))
    tensor = Tensor(shape=(4, 8))
    resolved = ResolvedValue(tensor=tensor, role=TensorRole.ACTIVATION, dtype=DType.FP32)
    assert policy.bytes_for(resolved) == 4 * 8 * 4


def test_accounting_policy_raises_on_unknown_resolved_dtype() -> None:
    with pytest.raises(ValueError, match="Precision policy default cannot be UNKNOWN"):
        PrecisionPolicy(default_dtype=DType.UNKNOWN)

    policy = AccountingPolicy(PrecisionPolicy(default_dtype=DType.FP32))
    tensor = Tensor(shape=(2, 3), dtype=DType.UNKNOWN)
    resolved = ResolvedValue(tensor=tensor, role=TensorRole.ACTIVATION, dtype=DType.UNKNOWN)
    with pytest.raises(ValueError, match="Cannot account bytes"):
        policy.bytes_for(resolved)


def test_accounting_policy_resolve_dtype_on_tensor() -> None:
    policy = AccountingPolicy(
        PrecisionPolicy(
            default_dtype=DType.FP32,
            by_role=((TensorRole.INPUT, DType.FP16),),
        )
    )
    tensor = Tensor(shape=(2,))
    assert policy.resolve_dtype(tensor, role=TensorRole.INPUT) is DType.FP16
    assert policy.resolve_dtype(tensor, role=TensorRole.ACTIVATION) is DType.FP32
