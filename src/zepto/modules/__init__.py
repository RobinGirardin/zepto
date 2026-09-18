"""Reusable user-level module compositions built on Zepto operations."""

from .layers.affine_linear import AffineLinear
from .models.apertus import APERTUS_70B, APERTUS_8B, Apertus, ApertusConfig
from .blocks.apertus_decoder_block import ApertusDecoderBlock
from .attention.attention import FlexibleAttention
from .attention.attention_config import (
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
from .attention.attention_softmax_with_sink import AttentionSoftmaxWithSink
from .mixers.depthwise_causal_conv1d import DepthwiseCausalConv1d
from .layers.embedding import Embedding
from .mixers.gated_delta_net import GatedDeltaNet
from .mixers.gated_delta_scan import GatedDeltaScan
from .layers.gated_grouped_rms_norm import GatedGroupedRMSNorm
from .layers.gated_rms_norm import GatedRMSNorm
from .blocks.attention_moe_decoder_block import AttentionMoEDecoderBlock
from .attention.decoder_attention_context import DecoderAttentionContext, LayerAttentionInputs
from .models.gemma4 import GEMMA4_31B_IT, Gemma4, Gemma4Config
from .models.gpt_oss import GPT_OSS_20B, GptOss, GptOssConfig
from .models.granite import GRANITE_42_30B, Granite, GraniteConfig
from .blocks.hybrid_decoder_block import HybridDecoderBlock
from .blocks.laguna_decoder_block import LagunaDecoderBlock
from .models.laguna_xs import LAGUNA_XS_21, LagunaXs, LagunaXsConfig
from .multimodal.multimodal_language_model import embed_multimodal_sequence
from .models.muse_glimmer import MUSE_GLIMMER_30B, MuseGlimmer, MuseGlimmerConfig
from .models.nemotron_h import NEMOTRON_H_30B_A3B, NemotronH, NemotronHConfig
from .models.qwen38 import QWEN38_27B, Qwen38, Qwen38Config
from .blocks.sandwich_norm_swiglu_decoder_block import (
    GEMMA4_LANGUAGE_SANDWICH_NORM,
    MUSE_LANGUAGE_SANDWICH_NORM,
    SandwichNormStyle,
    SandwichNormSwiGLUDecoderBlock,
)
from .blocks.swiglu_decoder_block import SwiGLUDecoderBlock
from .layers.l2_normalize import L2Normalize
from .mixers.layer_spec import LayerSpec, MixerKind
from .mixers.mamba2_mixer import Mamba2Mixer
from .mixers.mixer_config import GatedDeltaNetConfig, Mamba2MixerConfig
from .mixers.mixer_presets import nemotron_h_layer_specs, qwen35_language_layer_specs
from .blocks.qwen35_language_decoder_block import Qwen35LanguageDecoderBlock
from .mixers.selective_ssm_scan import SelectiveSSMScan
from .ffn.ffn import FFN
from .ffn.geglu import GeGLU
from .moe.gpt_oss.gpt_oss_expert import GptOssExpert
from .moe.gpt_oss.gpt_oss_expert_pool import GptOssExpertPool
from .moe.gpt_oss.gpt_oss_moe_block import GptOssMoEBlock
from .moe.gpt_oss.gpt_oss_moe_router import GptOssMoERouter
from .output.capped_fused_linear_cross_entropy import CappedFusedLinearCrossEntropy
from .output.fused_linear_cross_entropy import FusedLinearCrossEntropy
from .output.language_model_output import LanguageModelOutput, LanguageModelOutputConfig
from .output.logit_soft_cap import LogitSoftCap, LogitSoftCapConfig
from .mtp.mtp_config import MtpStageConfig
from .mtp.mtp_input_fusion import MtpInputFusion
from .mtp.mtp_mlp_block import MtpMlpBlock
from .mtp.mtp_presets import nemotron_h_mtp_stage, qwen38_mtp_head
from .mtp.mtp_stage import MtpStage
from .output.output_presets import (
    gemma4_lm_output,
    gpt_oss_lm_output,
    granite_lm_output,
    laguna_xs_lm_output,
    muse_glimmer_lm_output,
    nemotron_h_lm_output,
    qwen38_lm_output,
)
from .attention.gqa import GroupedQueryAttention
from .moe.laguna.laguna_expert_pool import LagunaExpertPool
from .moe.laguna.laguna_moe_router import LagunaMoERouter
from .moe.laguna.laguna_sparse_moe_block import LagunaSparseMoEBlock
from .layers.layer_norm import LayerNorm
from .layers.linear import Linear
from .layers.lm_head import LMHead
from .moe.moe_presets import gpt_oss_moe_block, laguna_sparse_moe_block, nemotron_moe_block
from .moe.moe_routing import UniformRoutingProfile
from .attention.materialized_causal_mask import MaterializedCausalMask
from .attention.sliding_window_causal_mask import SlidingWindowCausalMask
from .layers.qk_norm import QKNormRMSNorm
from .layers.rms_norm import RMSNorm
from .layers.relu import ReLU
from .layers.silu import SiLU
from .layers.softplus import Softplus
from .position.axial_rope_materialize import AxialRoPEMaterialize
from .position.interpolated_position_grid import InterpolatedPositionGrid
from .position.learned_position_embedding_2d import LearnedPositionEmbedding2D
from .multimodal.multimodal_projector import (
    MultimodalProjector,
    ProjectorConfig,
    gemma4_vision_projector,
    muse_glimmer_perception_adapter,
)
from .position.multimodal_rope_materialize import MultimodalRoPEMaterialize
from .multimodal.multimodal_sequence_builder import MultimodalSequenceBuilder
from .vision.patch_embed import (
    Conv2dPatchEmbed,
    Conv3dPatchEmbed,
    GemmaLinearPatchEmbed,
    LinearPatchEmbed,
    qwen3_vl_patch_embed,
)
from .vision.patch_merger import PatchMerger2x2
from .vision.pixel_shuffle import PixelShuffle2x2
from .vision.position_aware_pool import PositionAwareAveragePool2x2
from .vision.vision_attention_config import (
    VisionAttentionConfig,
    gemma4_vision_attention,
    muse_glimmer_vision_attention,
    muse_glimmer_vision_schedule,
    qwen3_vl_vision_attention,
)
from .vision.vision_encoder_block import VisionEncoderBlock
from .vision.vision_masks import (
    MaterializedBidirectionalMask,
    SlidingWindowBidirectionalMask,
    vision_mask_for_config,
)
from .vision.vision_presets import Gemma4VisionPath, MuseGlimmerVisionTower, Qwen3VLVisionTower, VisionConfig
from .moe.nemotron.nemotron_expert_pool import NemotronExpertPool
from .moe.nemotron.nemotron_moe_block import NemotronMoEBlock
from .moe.nemotron.nemotron_moe_router import NemotronMoERouter
from .position.rope_apply import RoPE, RoPEApply
from .position.rope_config import (
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
from .position.rope_materialize import RoPEMaterialize
from .attention.softmax import Softmax, attention_softmax_scale
from .layers.squared_relu import SquaredReLU
from .ffn.squared_relu_ffn import SquaredReluFFN
from .ffn.swiglu import SwiGLU
from .layers.xielu import XIELU

__all__ = [
    "APERTUS_70B",
    "APERTUS_8B",
    "AffineLinear",
    "Apertus",
    "ApertusConfig",
    "ApertusForCausalLM",
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
    "AttentionMoEDecoderBlock",
    "DecoderAttentionContext",
    "GEMMA4_31B_IT",
    "GPT_OSS_20B",
    "GRANITE_42_30B",
    "Gemma4",
    "Gemma4Config",
    "GptOss",
    "GptOssConfig",
    "Granite",
    "GraniteConfig",
    "HybridDecoderBlock",
    "LAGUNA_XS_21",
    "LagunaDecoderBlock",
    "LagunaXs",
    "LagunaXsConfig",
    "LayerAttentionInputs",
    "MUSE_GLIMMER_30B",
    "MuseGlimmer",
    "MuseGlimmerConfig",
    "NEMOTRON_H_30B_A3B",
    "NemotronH",
    "NemotronHConfig",
    "QWEN38_27B",
    "Qwen38",
    "Qwen38Config",
    "GEMMA4_LANGUAGE_SANDWICH_NORM",
    "MUSE_LANGUAGE_SANDWICH_NORM",
    "SandwichNormStyle",
    "SandwichNormSwiGLUDecoderBlock",
    "SwiGLUDecoderBlock",
    "embed_multimodal_sequence",
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
    "CappedFusedLinearCrossEntropy",
    "FusedLinearCrossEntropy",
    "LanguageModelOutput",
    "LanguageModelOutputConfig",
    "LogitSoftCap",
    "LogitSoftCapConfig",
    "MtpInputFusion",
    "MtpMlpBlock",
    "MtpStage",
    "MtpStageConfig",
    "gemma4_lm_output",
    "gpt_oss_lm_output",
    "granite_lm_output",
    "laguna_xs_lm_output",
    "muse_glimmer_lm_output",
    "nemotron_h_lm_output",
    "nemotron_h_mtp_stage",
    "qwen38_lm_output",
    "qwen38_mtp_head",
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
