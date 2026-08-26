"""Shared helpers for Zepto module primitive composition."""

from __future__ import annotations

from ..core.composition import GraphCompositionContext, GraphParameter, GraphTensor
from ..core.operation import ParameterBias, ParameterScale


def require_context() -> GraphCompositionContext:
    """Return the active graph composition context."""
    context = GraphCompositionContext.current()
    if context is None:
        raise RuntimeError("Module construction requires an active GraphCompositionContext")
    return context


def norm_axis(value: GraphTensor) -> int:
    """Return the last-dimension axis used for normalization reductions."""
    rank = len(value.metadata.shape)
    if rank < 1:
        raise ValueError("normalization expects rank ≥ 1 input")
    return rank - 1


def feature_size(value: GraphTensor, *, axis: int | None = None) -> int:
    """Return the normalized feature dimension size."""
    resolved = norm_axis(value) if axis is None else axis
    return value.metadata.shape[resolved]


def scale_by_parameter(value: GraphTensor, weight: GraphParameter) -> GraphTensor:
    """Broadcast-multiply ``value`` by a 1-D ``weight`` parameter."""
    return require_context().apply(  # type: ignore[return-value]
        ParameterScale(),
        value,
        parameters=(weight,),
    )


def add_bias_parameter(value: GraphTensor, bias: GraphParameter) -> GraphTensor:
    """Broadcast-add a 1-D ``bias`` parameter to ``value``."""
    return require_context().apply(  # type: ignore[return-value]
        ParameterBias(),
        value,
        parameters=(bias,),
    )
