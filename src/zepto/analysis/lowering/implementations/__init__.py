"""Default lowering implementation registrations."""

from __future__ import annotations

from zepto.semantic.operations import (
    Add,
    Divide,
    Identity,
    LinearMatMul,
    MatMul,
    Maximum,
    Minimum,
    Multiply,
    Reshape,
    Split,
    SquareRoot,
    Subtract,
    Transpose,
)
from ..registry import ImplementationDescriptor, LoweringRegistry
from .identity import IdentityImplementation
from .maximum import RELU_MASK


def register_identity_defaults(registry: LoweringRegistry) -> None:
    """Register one identity implementation per shipped structural family."""
    for operation in (
        Add(),
        Subtract(),
        Multiply(),
        Divide(),
        MatMul(),
        LinearMatMul(),
        Identity(),
        Reshape(shape=(1,)),
        Transpose(permutation=(0,)),
        Split(sizes=(1,)),
        Maximum(),
        Minimum(),
        SquareRoot(),
    ):
        registry.register(
            IdentityImplementation(
                operation=operation,
                descriptor=ImplementationDescriptor(
                    id=f"{operation.family}/identity",
                    family=operation.family,
                    priority=0,
                ),
            )
        )


def register_specialized(registry: LoweringRegistry) -> None:
    """Register non-identity implementations that override identity defaults."""
    registry.register(RELU_MASK)


def register_defaults(registry: LoweringRegistry) -> None:
    """Populate a registry with all built-in implementations."""
    register_identity_defaults(registry)
    register_specialized(registry)
