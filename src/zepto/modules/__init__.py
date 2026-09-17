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
from .depthwise_causal_conv1d import DepthwiseCausalConv1d
from .embedding import Embedding
from .gated_delta_net import GatedDeltaNet
from .gated_delta_scan import GatedDeltaScan
from .gated_grouped_rms_norm import GatedGroupedRMSNorm
from .gated_rms_norm import GatedRMSNorm
from .hybrid_decoder_block import HybridDecoderBlock
from .l2_normalize import L2Normalize
from .layer_spec import LayerSpec, MixerKind
from .mamba2_mixer import Mamba2Mixer
from .mixer_config import GatedDeltaNetConfig, Mamba2MixerConfig
from .mixer_presets import nemotron_h_layer_specs, qwen35_language_layer_specs
from .qwen35_language_decoder_block import Qwen35LanguageDecoderBlock
from .selective_ssm_scan import SelectiveSSMScan
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
from .multimodal_projector import (
    MultimodalProjector,
    ProjectorConfig,
    gemma4_vision_projector,
    muse_glimmer_perception_adapter,
)
from .multimodal_rope_materialize import MultimodalRoPEMaterialize
from .multimodal_sequence_builder import MultimodalSequenceBuilder
from .patch_embed import (
    Conv2dPatchEmbed,
    Conv3dPatchEmbed,
    GemmaLinearPatchEmbed,
    LinearPatchEmbed,
    qwen3_vl_patch_embed,
)
from .patch_merger import PatchMerger2x2
from .pixel_shuffle import PixelShuffle2x2
from .position_aware_pool import PositionAwareAveragePool2x2
from .vision_attention_config import (
    VisionAttentionConfig,
    gemma4_vision_attention,
    muse_glimmer_vision_attention,
    muse_glimmer_vision_schedule,
    qwen3_vl_vision_attention,
)
from .vision_encoder_block import VisionEncoderBlock
from .vision_masks import (
    MaterializedBidirectionalMask,
    SlidingWindowBidirectionalMask,
    vision_mask_for_config,
)
from .vision_presets import Gemma4VisionPath, MuseGlimmerVisionTower, Qwen3VLVisionTower, VisionConfig
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
    "DepthwiseCausalConv1d",
    "Embedding",
    "FlexibleAttention",
    "GatedDeltaNet",
    "GatedDeltaNetConfig",
    "GatedDeltaScan",
    "GatedGroupedRMSNorm",
    "GatedRMSNorm",
    "HybridDecoderBlock",
    "L2Normalize",
    "LayerSpec",
    "Mamba2Mixer",
    "Mamba2MixerConfig",
    "MixerKind",
    "Qwen35LanguageDecoderBlock",
    "SelectiveSSMScan",
    "nemotron_h_layer_specs",
    "qwen35_language_layer_specs",
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
    "Conv2dPatchEmbed",
    "Conv3dPatchEmbed",
    "Gemma4VisionPath",
    "GemmaLinearPatchEmbed",
    "LinearPatchEmbed",
    "MaterializedBidirectionalMask",
    "MultimodalProjector",
    "MultimodalRoPEMaterialize",
    "MultimodalSequenceBuilder",
    "MuseGlimmerVisionTower",
    "PatchMerger2x2",
    "PixelShuffle2x2",
    "PositionAwareAveragePool2x2",
    "ProjectorConfig",
    "Qwen3VLVisionTower",
    "SlidingWindowBidirectionalMask",
    "VisionAttentionConfig",
    "VisionConfig",
    "VisionEncoderBlock",
    "gemma4_vision_attention",
    "gemma4_vision_projector",
    "muse_glimmer_perception_adapter",
    "muse_glimmer_vision_attention",
    "muse_glimmer_vision_schedule",
    "qwen3_vl_patch_embed",
    "qwen3_vl_vision_attention",
    "vision_mask_for_config",
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
