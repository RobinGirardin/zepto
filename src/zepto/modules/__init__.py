"""Reusable user-level module compositions built on Zepto operations."""

from .apertus import APERTUS_70B, APERTUS_8B, Apertus, ApertusConfig
from .apertus_decoder_block import ApertusDecoderBlock
from .embedding import Embedding
from .ffn import FFN
from .fused_linear_cross_entropy import FusedLinearCrossEntropy
from .gqa import GroupedQueryAttention
from .layer_norm import LayerNorm
from .linear import Linear
from .lm_head import LMHead
from .materialized_causal_mask import MaterializedCausalMask
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .relu import ReLU
from .silu import SiLU
from .softplus import Softplus
from .rope_apply import RoPE, RoPEApply
from .rope_materialize import RoPEConfig, RoPEMaterialize
from .softmax import Softmax, attention_softmax_scale
from .xielu import XIELU

__all__ = [
    "APERTUS_70B",
    "APERTUS_8B",
    "Apertus",
    "ApertusConfig",
    "ApertusDecoderBlock",
    "Embedding",
    "FFN",
    "FusedLinearCrossEntropy",
    "GroupedQueryAttention",
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
    "XIELU",
    "attention_softmax_scale",
]
