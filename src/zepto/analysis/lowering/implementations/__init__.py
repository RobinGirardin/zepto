"""Default lowering implementation registrations."""

from __future__ import annotations

from zepto.semantic.metadata import DType
from zepto.semantic.operations import (
    Add,
    AttentionSoftmaxWithSink,
    BilinearResize2D,
    Cast,
    Concat,
    Conv2d,
    Conv3d,
    Cos,
    Divide,
    EmbeddingLookup,
    Erf,
    Equal,
    Exp,
    Gather,
    GeluErfGate,
    GeluTanhGate,
    GreaterThan,
    Identity,
    LinearMatMul,
    Log,
    MatMul,
    MaterializedBidirectionalMask,
    MaterializedCausalMask,
    MaterializedSlidingWindowBidirectionalMask,
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
    ScatterAdd,
    ScatterUpdate,
    Sigmoid,
    Sin,
    Split,
    SquareRoot,
    Subtract,
    Tanh,
    TopK,
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
    DEPTHWISE_CAUSAL_CONV1D_REGIONS,
    GATED_DELTA_SCAN_REGIONS,
    GATED_DELTA_NET_REGIONS,
    MAMBA2_SCAN_REGIONS,
    GATED_GROUPED_RMS_NORM_REGIONS,
    GATED_RMS_NORM_REGIONS,
    L2_NORMALIZE_REGIONS,
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
    MAMBA2_MIXER_REGIONS,
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
        Equal(),
        Exp(),
        Gather(axis=0),
        GreaterThan(),
        Log(),
        Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            stride=1,
        ),
        Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
        ),
        MaterializedBidirectionalMask(seq_len=1),
        MaterializedCausalMask(seq_len=1),
        MaterializedSlidingWindowBidirectionalMask(seq_len=1, window_size=1),
        MaterializedSlidingWindowCausalMask(seq_len=1, window_size=1),
        ScatterUpdate(axis=0),
        AttentionSoftmaxWithSink(),
        BilinearResize2D(out_h=1, out_w=1),
        ParameterBias(),
        ParameterScale(),
        Pow(),
        ReduceSum(axis=0, keepdim=True),
        RepeatKV(n_rep=1),
        ScatterAdd(axis=0),
        Sigmoid(),
        Sin(),
        Tanh(),
        TopK(k=1),
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
    for impl in L2_NORMALIZE_REGIONS:
        registry.register_region(impl)
    for impl in DEPTHWISE_CAUSAL_CONV1D_REGIONS:
        registry.register_region(impl)
    for impl in GATED_DELTA_SCAN_REGIONS:
        registry.register_region(impl)
    for impl in GATED_DELTA_NET_REGIONS:
        registry.register_region(impl)
    for impl in MAMBA2_SCAN_REGIONS:
        registry.register_region(impl)
    for impl in GATED_GROUPED_RMS_NORM_REGIONS:
        registry.register_region(impl)
    for impl in GATED_RMS_NORM_REGIONS:
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
    for impl in MAMBA2_MIXER_REGIONS:
        registry.register_region(impl)


def register_defaults(registry: LoweringRegistry) -> None:
    """Populate a registry with all built-in implementations."""
    register_identity_defaults(registry)
    register_specialized(registry)
    register_regions(registry)
