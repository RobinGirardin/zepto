import pytest

from zepto.core import (
    AccountingPolicy,
    DType,
    Parameter,
    PrecisionPolicy,
    ValueMetadata,
    TensorRole,
    metadata_compatible,
)


def test_dtype_itemsize_widths() -> None:
    assert DType.UNKNOWN.itemsize is None
    assert DType.BOOL.itemsize == 1
    assert DType.INT32.itemsize == 4
    assert DType.FP16.itemsize == 2
    assert DType.BF16.itemsize == 2
    assert DType.FP32.itemsize == 4
    assert DType.FP64.itemsize == 8


def test_tensor_metadata_accepts_accounting_fields() -> None:
    metadata = ValueMetadata(
        (4, 8),
        dtype=DType.FP32,
        role=TensorRole.ACTIVATION,
        persistent=True,
        requires_grad=True,
    )
    assert metadata.dtype is DType.FP32
    assert metadata.role is TensorRole.ACTIVATION
    assert metadata.persistent is True
    assert metadata.requires_grad is True


def test_tensor_metadata_rejects_invalid_accounting_fields() -> None:
    with pytest.raises(ValueError, match="dtype must be a DType"):
        ValueMetadata((2,), dtype="fp32")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="role must be a TensorRole"):
        ValueMetadata((2,), role="activation")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="persistent must be an explicit boolean"):
        ValueMetadata((2,), persistent=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires_grad must be an explicit boolean"):
        ValueMetadata((2,), requires_grad=1)  # type: ignore[arg-type]


def test_metadata_compatible_dtype_and_role_wildcards() -> None:
    expected = ValueMetadata((2, 3), semantic_type="tensor")
    actual = ValueMetadata(
        (2, 3),
        semantic_type="tensor",
        dtype=DType.FP16,
        role=TensorRole.ACTIVATION,
    )
    assert metadata_compatible(expected, actual)

    specified = ValueMetadata(
        (2, 3),
        semantic_type="tensor",
        dtype=DType.FP32,
        role=TensorRole.PARAMETER,
    )
    assert not metadata_compatible(specified, actual)


def test_parameter_trainable_defaults_true() -> None:
    from zepto.core.ids import GraphId, ParameterId

    graph_id = GraphId.new()
    parameter_id = ParameterId(graph_id, 0)
    metadata = ValueMetadata((8,), requires_grad=False)
    parameter = Parameter(parameter_id, metadata)
    assert parameter.trainable is True
    assert parameter.metadata.requires_grad is False


def test_precision_policy_resolve_precedence() -> None:
    policy = PrecisionPolicy(
        default_dtype=DType.FP32,
        by_semantic_type=(("weight", DType.FP16),),
        by_role=((TensorRole.GRADIENT, DType.FP32),),
    )
    explicit = ValueMetadata((2,), dtype=DType.FP64)
    by_semantic = ValueMetadata((2,), semantic_type="weight")
    by_role = ValueMetadata((2,), role=TensorRole.GRADIENT)
    default = ValueMetadata((2,))

    assert policy.resolve(explicit) is DType.FP64
    assert policy.resolve(by_semantic) is DType.FP16
    assert policy.resolve(by_role) is DType.FP32
    assert policy.resolve(default) is DType.FP32


def test_accounting_policy_bytes_for() -> None:
    policy = AccountingPolicy(PrecisionPolicy(default_dtype=DType.FP32))
    metadata = ValueMetadata((4, 8))
    assert policy.bytes_for(metadata) == 4 * 8 * 4


def test_accounting_policy_raises_on_unknown_resolved_dtype() -> None:
    with pytest.raises(ValueError, match="Precision policy default cannot be UNKNOWN"):
        PrecisionPolicy(default_dtype=DType.UNKNOWN)

    policy = AccountingPolicy(PrecisionPolicy(default_dtype=DType.FP32))
    metadata = ValueMetadata((2, 3), dtype=DType.UNKNOWN)
    with pytest.raises(ValueError, match="Cannot account bytes"):
        policy.bytes_for(metadata)
