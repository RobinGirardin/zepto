"""Shared helpers for Zepto module primitive composition."""

from __future__ import annotations

from zepto.compose import Parameter, Tensor
from zepto.semantic import ParameterBias, ParameterScale
from zepto.semantic.metadata import DType

# HF Llama/Apertus: variance path runs in fp32 regardless of activation storage.
RMSNORM_COMPUTE_DTYPE = DType.FP32


def norm_axis(value: Tensor) -> int:
    """Return the last-dimension axis used for normalization reductions."""
    rank = len(value.shape)
    if rank < 1:
        raise ValueError("normalization expects rank ≥ 1 input")
    return rank - 1


def feature_size(value: Tensor, *, axis: int | None = None) -> int:
    """Return the normalized feature dimension size."""
    resolved = norm_axis(value) if axis is None else axis
    return value.shape[resolved]


def scale_by_parameter(value: Tensor, weight: Parameter) -> Tensor:
    """Broadcast-multiply ``value`` by a 1-D ``weight`` parameter."""
    return ParameterScale()(value, parameters=(weight,))  # type: ignore[return-value]


def add_bias_parameter(value: Tensor, bias: Parameter) -> Tensor:
    """Broadcast-add a 1-D ``bias`` parameter to ``value``."""
    return ParameterBias()(value, parameters=(bias,))  # type: ignore[return-value]


def activation_dtype(value: Tensor, *, default: DType = DType.BF16) -> DType:
    """Return the dtype RMSNorm restores to after fp32 normalize."""
    return value.dtype if value.dtype is not None else default
