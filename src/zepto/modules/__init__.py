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
from .gpt_oss_expert import GptOssExpert
from .gpt_oss_expert_pool import GptOssExpertPool
from .gpt_oss_moe_block import GptOssMoEBlock
from .gpt_oss_moe_router import GptOssMoERouter
from .fused_linear_cross_entropy import FusedLinearCrossEntropy
from .gqa import GroupedQueryAttention
from .laguna_expert_pool import LagunaExpertPool
from .laguna_moe_router import LagunaMoERouter
from .laguna_sparse_moe_block import LagunaSparseMoEBlock
from .layer_norm import LayerNorm
from .linear import Linear
from .lm_head import LMHead
from .moe_presets import gpt_oss_moe_block, laguna_sparse_moe_block, nemotron_moe_block
from .moe_routing import UniformRoutingProfile
from .materialized_causal_mask import MaterializedCausalMask
from .sliding_window_causal_mask import SlidingWindowCausalMask
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .relu import ReLU
from .silu import SiLU
from .softplus import Softplus
from .axial_rope_materialize import AxialRoPEMaterialize
from .interpolated_position_grid import InterpolatedPositionGrid
from .learned_position_embedding_2d import LearnedPositionEmbedding2D
from .multimodal_rope_materialize import MultimodalRoPEMaterialize
from .nemotron_expert_pool import NemotronExpertPool
from .nemotron_moe_block import NemotronMoEBlock
from .nemotron_moe_router import NemotronMoERouter
from .rope_apply import RoPE, RoPEApply
from .rope_config import (
    LayerRoPEBinding,
    RoPEConfig,
    apertus_rope,
    bind_rope,
    gemma4_layer_binding,
    gemma4_rope_layer,
    gpt_oss_layer_binding,
    gpt_oss_rope,
    granite_rope,
    laguna_layer_binding,
    laguna_rope_layer,
    llama_rope,
    muse_glimmer_layer_binding,
    muse_glimmer_rope_layer,
    qwen3_vl_mrope,
)
from .rope_materialize import RoPEMaterialize
from .softmax import Softmax, attention_softmax_scale
from .squared_relu import SquaredReLU
from .squared_relu_ffn import SquaredReluFFN
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
    "AxialRoPEMaterialize",
    "Embedding",
    "FlexibleAttention",
    "FFN",
    "GeGLU",
    "GptOssExpert",
    "GptOssExpertPool",
    "GptOssMoEBlock",
    "GptOssMoERouter",
    "FusedLinearCrossEntropy",
    "GroupedQueryAttention",
    "gpt_oss_moe_block",
    "laguna_sparse_moe_block",
    "nemotron_moe_block",
    "InterpolatedPositionGrid",
    "LagunaExpertPool",
    "LagunaMoERouter",
    "LagunaSparseMoEBlock",
    "LayerRoPEBinding",
    "LearnedPositionEmbedding2D",
    "MultimodalRoPEMaterialize",
    "SlidingWindowCausalMask",
    "apertus_rope",
    "bind_rope",
    "gemma4_layer_binding",
    "gemma4_rope_layer",
    "gpt_oss_layer_binding",
    "gpt_oss_rope",
    "granite_rope",
    "laguna_layer_binding",
    "laguna_rope_layer",
    "llama_rope",
    "muse_glimmer_layer_binding",
    "muse_glimmer_rope_layer",
    "qwen3_vl_mrope",
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
    "NemotronExpertPool",
    "NemotronMoEBlock",
    "NemotronMoERouter",
    "UniformRoutingProfile",
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
    "SquaredReLU",
    "SquaredReluFFN",
    "SwiGLU",
    "XIELU",
    "attention_softmax_scale",
]
