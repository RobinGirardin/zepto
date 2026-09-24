"""Empirical parity context: uniform fp32/fp16 only."""

from __future__ import annotations

import pytest

from zepto.compose import Tensor
from zepto.empirical.parity_ctx import build_invocation_context
from zepto.semantic.metadata import DType, TensorRole

_CUDA = (8, 0)


def test_fp32_context() -> None:
    ctx = build_invocation_context("fp32", cuda_capability=_CUDA)
    assert ctx.precision.default_dtype is DType.FP32
    activation = Tensor(shape=(4, 8))
    assert ctx.precision.resolve(activation, role=TensorRole.GRADIENT) is DType.FP32


def test_fp16_context_is_uniform() -> None:
    ctx = build_invocation_context("fp16", cuda_capability=_CUDA)
    assert ctx.precision.default_dtype is DType.FP16
    weight = Tensor(shape=(4, 8), semantic_type="weight")
    activation = Tensor(shape=(4, 8))
    assert ctx.precision.resolve(weight, role=TensorRole.PARAMETER) is DType.FP16
    assert ctx.precision.resolve(activation, role=TensorRole.GRADIENT) is DType.FP16


def test_parity_ctx_fp16_optim_prec_is_2() -> None:
    fp16 = build_invocation_context("fp16", cuda_capability=_CUDA)
    fp32 = build_invocation_context("fp32", cuda_capability=_CUDA)
    assert fp16.optim_prec == 2
    assert fp32.optim_prec == 4


def test_mixed_precision_raises() -> None:
    with pytest.raises(ValueError, match="unknown precision"):
        build_invocation_context("mixed", cuda_capability=_CUDA)
