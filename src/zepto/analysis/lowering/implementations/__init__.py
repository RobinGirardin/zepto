"""Default lowering implementation registrations."""

from __future__ import annotations

from zepto.semantic.operations import (
    Add,
    Concat,
    Cos,
    Divide,
    EmbeddingLookup,
    Exp,
    Gather,
    GreaterThan,
    Identity,
    LinearMatMul,
    Log,
    MatMul,
    MaterializedCausalMask,
    Maximum,
    Minimum,
    Multiply,
    ParameterBias,
    ParameterScale,
    Pow,
    ReduceSum,
    RepeatKV,
    Reshape,
    Sin,
    Split,
    SquareRoot,
    Subtract,
    Transpose,
    Where,
)
from ..registry import ImplementationDescriptor, LoweringRegistry
from .identity import IdentityImplementation
from .maximum import RELU_MASK
from .regions import FUSED_LAYERNORM_REGION, FUSED_LINEAR_REGION, RELU_REGION


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
        Concat(axis=0, input_count=2),
        Cos(),
        EmbeddingLookup(),
        Exp(),
        Gather(axis=0),
        GreaterThan(),
        Log(),
        MaterializedCausalMask(seq_len=1),
        ParameterBias(),
        ParameterScale(),
        Pow(),
        ReduceSum(axis=0, keepdim=True),
        RepeatKV(n_rep=1),
        Sin(),
        Where(),
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


def register_regions(registry: LoweringRegistry) -> None:
    """Register fused region implementations."""
    registry.register_region(RELU_REGION)
    registry.register_region(FUSED_LINEAR_REGION)
    registry.register_region(FUSED_LAYERNORM_REGION)


def register_defaults(registry: LoweringRegistry) -> None:
    """Populate a registry with all built-in implementations."""
    register_identity_defaults(registry)
    register_specialized(registry)
    register_regions(registry)
