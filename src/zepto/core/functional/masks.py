"""Functional wrappers for mask and constant tensor allocation."""

from ..composition import GraphTensor
from ..operation import MaterializedCausalMask
from ._common import context


def materialized_causal_mask(seq_len: int) -> GraphTensor:
    """Record allocation of a persistent additive causal mask."""
    return context().apply(MaterializedCausalMask(seq_len))  # type: ignore[return-value]
