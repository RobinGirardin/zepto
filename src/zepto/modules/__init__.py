"""Reusable user-level module compositions built on Zepto operations."""

from .affine_linear import AffineLinear
from .apertus import APERTUS_70B, APERTUS_8B, Apertus, ApertusConfig
from .apertus_decoder_block import ApertusDecoderBlock
from .attention import FlexibleAttention
from .attention_config import (
    AttentionConfig,
    AttentionLayerTemplate,
    gated_gqa,
    gemma4_text_attention_layer,
    gpt_oss_attention_layer,
    granite_attention_layer,
    laguna_xs_attention_layer,
    llama_gqa,
    muse_glimmer_text_attention_layer,
    repeat_pattern,
)
from .attention_softmax_with_sink import AttentionSoftmaxWithSink
from .embedding import Embedding
from .ffn import FFN
from .geglu import GeGLU
from .fused_linear_cross_entropy import FusedLinearCrossEntropy
from .gqa import GroupedQueryAttention
from .layer_norm import LayerNorm
from .linear import Linear
from .lm_head import LMHead
from .materialized_causal_mask import MaterializedCausalMask
from .sliding_window_causal_mask import SlidingWindowCausalMask
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .relu import ReLU
from .silu import SiLU
from .softplus import Softplus
from .rope_apply import RoPE, RoPEApply
from .rope_materialize import RoPEConfig, RoPEMaterialize
from .softmax import Softmax, attention_softmax_scale
from .swiglu import SwiGLU
from .xielu import XIELU

__all__ = [
    "APERTUS_70B",
    "APERTUS_8B",
    "AffineLinear",
    "Apertus",
    "ApertusConfig",
    "ApertusDecoderBlock",
    "AttentionConfig",
    "AttentionLayerTemplate",
    "AttentionSoftmaxWithSink",
    "Embedding",
    "FlexibleAttention",
    "FFN",
    "GeGLU",
    "FusedLinearCrossEntropy",
    "GroupedQueryAttention",
    "SlidingWindowCausalMask",
    "gated_gqa",
    "gemma4_text_attention_layer",
    "gpt_oss_attention_layer",
    "granite_attention_layer",
    "laguna_xs_attention_layer",
    "llama_gqa",
    "muse_glimmer_text_attention_layer",
    "repeat_pattern",
    "LayerNorm",
    "LMHead",
    "Linear",
    "MaterializedCausalMask",
    "QKNormRMSNorm",
    "RMSNorm",
    "ReLU",
    "SiLU",
    "Softplus",
    "RoPE",
    "RoPEApply",
    "RoPEConfig",
    "RoPEMaterialize",
    "Softmax",
    "SwiGLU",
    "XIELU",
    "attention_softmax_scale",
]
