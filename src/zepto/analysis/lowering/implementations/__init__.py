"""Default lowering implementation registrations."""

from __future__ import annotations

from zepto.semantic.metadata import DType
from zepto.semantic.operations import (
    Add,
    AttentionSoftmaxWithSink,
    BilinearResize2D,
    Cast,
    Concat,
    Cos,
    Divide,
    EmbeddingLookup,
    Erf,
    Exp,
    Gather,
    GeluErfGate,
    GeluTanhGate,
    GreaterThan,
    Identity,
    LinearMatMul,
    Log,
    MatMul,
    MaterializedCausalMask,
    MaterializedSlidingWindowCausalMask,
    Maximum,
    Minimum,
    Multiply,
    ParameterBias,
    ParameterScale,
    Pow,
    ReduceSum,
    RepeatKV,
    Reshape,
    Sigmoid,
    Sin,
    Split,
    SquareRoot,
    Subtract,
    Tanh,
    Transpose,
    Where,
)
from ..registry import ImplementationDescriptor, LoweringRegistry
from .identity import IdentityImplementation
from .maximum import RELU_MASK
from .regions import (
    FUSED_LAYERNORM_REGION,
    FUSED_LINEAR_REGION,
    GELU_REGIONS,
    GQA_REGIONS,
    GQA_SINK_REGIONS,
    LINEAR_CE_REGIONS,
    MASKED_SOFTMAX_REGIONS,
    RELU_REGIONS,
    RMSNORM_REGIONS,
    SILU_REGIONS,
    SQUARED_RELU_REGIONS,
    SOFTMAX_ONE_REGIONS,
    SOFTMAX_REGIONS,
    SOFTPLUS_REGIONS,
    GEGLU_REGIONS,
    SWIGLU_REGIONS,
    XIELU_REGIONS,
)


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
        Cast(to_dtype=DType.FP32),
        Concat(axis=0, input_count=2),
        Cos(),
        EmbeddingLookup(),
        Exp(),
        Gather(axis=0),
        GreaterThan(),
        Log(),
        MaterializedCausalMask(seq_len=1),
        MaterializedSlidingWindowCausalMask(seq_len=1, window_size=1),
        AttentionSoftmaxWithSink(),
        BilinearResize2D(out_h=1, out_w=1),
        ParameterBias(),
        ParameterScale(),
        Pow(),
        ReduceSum(axis=0, keepdim=True),
        RepeatKV(n_rep=1),
        Sigmoid(),
        Sin(),
        Tanh(),
        Erf(),
        GeluTanhGate(),
        GeluErfGate(),
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
    for impl in RELU_REGIONS:
        registry.register_region(impl)
    for impl in SILU_REGIONS:
        registry.register_region(impl)
    for impl in SQUARED_RELU_REGIONS:
        registry.register_region(impl)
    for impl in SOFTPLUS_REGIONS:
        registry.register_region(impl)
    for impl in GELU_REGIONS:
        registry.register_region(impl)
    registry.register_region(FUSED_LINEAR_REGION)
    registry.register_region(FUSED_LAYERNORM_REGION)
    for impl in RMSNORM_REGIONS:
        registry.register_region(impl)
    for impl in XIELU_REGIONS:
        registry.register_region(impl)
    for impl in SOFTMAX_REGIONS:
        registry.register_region(impl)
    for impl in SOFTMAX_ONE_REGIONS:
        registry.register_region(impl)
    for impl in LINEAR_CE_REGIONS:
        registry.register_region(impl)
    for impl in MASKED_SOFTMAX_REGIONS:
        registry.register_region(impl)
    for impl in GQA_REGIONS:
        registry.register_region(impl)
    for impl in GQA_SINK_REGIONS:
        registry.register_region(impl)
    for impl in SWIGLU_REGIONS:
        registry.register_region(impl)
    for impl in GEGLU_REGIONS:
        registry.register_region(impl)


def register_defaults(registry: LoweringRegistry) -> None:
    """Populate a registry with all built-in implementations."""
    register_identity_defaults(registry)
    register_specialized(registry)
    register_regions(registry)
