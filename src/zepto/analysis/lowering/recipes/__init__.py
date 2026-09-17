"""Closed-form cost recipes for fused region implementations."""

from .geglu import (
    DEFAULT_GEGLU_RECIPE,
    GEGLU_ERF_RECIPE,
    LIGER_GEGLU_RECIPE,
    GeGLURecipe,
)
from .gelu import (
    GELU_ERF_RECIPE,
    GELU_QUICK_RECIPE,
    GELU_TANH_RECIPE,
    GeluRecipe,
)
from .gqa import (
    DEFAULT_GQA_RECIPE,
    DEFAULT_SDPA_MATH_RECIPE,
    GQARecipe,
    GQASDPAMathRecipe,
)
from .depthwise_causal_conv1d import (
    DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE,
    DepthwiseCausalConv1dRecipe,
)
from .gated_delta_scan import (
    DECODE_GATED_DELTA_SCAN_RECIPE,
    DEFAULT_GATED_DELTA_SCAN_RECIPE,
    GatedDeltaScanRecipe,
)
from .mamba2_scan import (
    DECODE_MAMBA2_SCAN_RECIPE,
    DEFAULT_MAMBA2_SCAN_RECIPE,
    Mamba2ScanRecipe,
)
from .gated_grouped_rms_norm import (
    DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE,
    EAGER_HF_GATED_GROUPED_RMS_NORM_RECIPE,
    GatedGroupedRMSNormRecipe,
)
from .gated_rms_norm import (
    DEFAULT_GATED_RMS_NORM_RECIPE,
    GatedRMSNormRecipe,
)
from .l2_normalize import DEFAULT_L2_NORMALIZE_RECIPE, L2NormalizeRecipe
from .linear_ce import DEFAULT_LINEAR_CE_RECIPE, LinearCERecipe
from .masked_softmax import (
    DEFAULT_MASKED_SOFTMAX_RECIPE,
    DEFAULT_MASKED_SOFTMAX_SINK_RECIPE,
    MaskedSoftmaxRecipe,
    MaskedSoftmaxSinkRecipe,
)
from .softmax_one import (
    DEFAULT_SOFTMAX_ONE_RECIPE,
    MEGATRON_SOFTMAX_ONE_RECIPE,
    SoftmaxOneRecipe,
)
from .rmsnorm import RMSNormRecipe
from .softmax import DEFAULT_SOFTMAX_RECIPE, SoftmaxRecipe
from .relu import DEFAULT_RELU_RECIPE, SAVED_INPUT_RELU_RECIPE, ReLURecipe
from .silu import DEFAULT_SILU_RECIPE, SiLURecipe
from .squared_relu import DEFAULT_SQUARED_RELU_RECIPE, SquaredReLURecipe
from .softplus import (
    DEFAULT_SOFTPLUS_RECIPE,
    XIELU_SCALAR_SOFTPLUS_RECIPE,
    SoftplusRecipe,
)
from .swiglu import (
    DEFAULT_SWIGLU_RECIPE,
    LIGER_FUSED_GATE_UP_SWIGLU_RECIPE,
    LIGER_SWIGLU_RECIPE,
    SwiGLURecipe,
)
from .xielu import XIELURecipe

__all__ = [
    "DEFAULT_RELU_RECIPE",
    "SAVED_INPUT_RELU_RECIPE",
    "GELU_ERF_RECIPE",
    "GELU_QUICK_RECIPE",
    "GELU_TANH_RECIPE",
    "GeluRecipe",
    "DEFAULT_GEGLU_RECIPE",
    "GEGLU_ERF_RECIPE",
    "LIGER_GEGLU_RECIPE",
    "GeGLURecipe",
    "DEFAULT_GQA_RECIPE",
    "DEFAULT_DEPTHWISE_CAUSAL_CONV1D_RECIPE",
    "DECODE_GATED_DELTA_SCAN_RECIPE",
    "DEFAULT_GATED_DELTA_SCAN_RECIPE",
    "DEFAULT_L2_NORMALIZE_RECIPE",
    "DepthwiseCausalConv1dRecipe",
    "GatedDeltaScanRecipe",
    "DECODE_MAMBA2_SCAN_RECIPE",
    "DEFAULT_MAMBA2_SCAN_RECIPE",
    "Mamba2ScanRecipe",
    "DEFAULT_GATED_GROUPED_RMS_NORM_RECIPE",
    "EAGER_HF_GATED_GROUPED_RMS_NORM_RECIPE",
    "GatedGroupedRMSNormRecipe",
    "DEFAULT_GATED_RMS_NORM_RECIPE",
    "GatedRMSNormRecipe",
    "DEFAULT_LINEAR_CE_RECIPE",
    "LinearCERecipe",
    "DEFAULT_MASKED_SOFTMAX_RECIPE",
    "DEFAULT_MASKED_SOFTMAX_SINK_RECIPE",
    "DEFAULT_SOFTMAX_ONE_RECIPE",
    "MEGATRON_SOFTMAX_ONE_RECIPE",
    "DEFAULT_SDPA_MATH_RECIPE",
    "DEFAULT_SOFTMAX_RECIPE",
    "GQARecipe",
    "GQASDPAMathRecipe",
    "L2NormalizeRecipe",
    "MaskedSoftmaxRecipe",
    "MaskedSoftmaxSinkRecipe",
    "SoftmaxOneRecipe",
    "DEFAULT_SILU_RECIPE",
    "DEFAULT_SQUARED_RELU_RECIPE",
    "ReLURecipe",
    "RMSNormRecipe",
    "DEFAULT_SOFTPLUS_RECIPE",
    "SiLURecipe",
    "SquaredReLURecipe",
    "SoftmaxRecipe",
    "SoftplusRecipe",
    "DEFAULT_SWIGLU_RECIPE",
    "LIGER_FUSED_GATE_UP_SWIGLU_RECIPE",
    "LIGER_SWIGLU_RECIPE",
    "SwiGLURecipe",
    "XIELU_SCALAR_SOFTPLUS_RECIPE",
    "XIELURecipe",
]
