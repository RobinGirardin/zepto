"""Checkpoint-faithful MTP stage factories."""

from __future__ import annotations

from collections.abc import Callable

from zepto.compose import Module, Parameter

from .attention_config import AttentionConfig
from .embedding import Embedding
from .language_model_output import LanguageModelOutput, LanguageModelOutputConfig
from .layer_spec import LayerSpec
from .mtp_config import MtpStageConfig
from .mtp_stage import MtpStage
from .output_presets import qwen38_lm_output


_QWEN38_HIDDEN = 5120
_QWEN38_VOCAB = 248320
_QWEN38_MTP_FFN = 17408
_QWEN38_MTP_SOURCE_LAYER = 47

_NEMOTRON_HIDDEN = 2688
_NEMOTRON_VOCAB = 131072


def _nemotron_mtp_attention(hidden_size: int) -> AttentionConfig:
    if hidden_size == _NEMOTRON_HIDDEN:
        return AttentionConfig(
            hidden_size=hidden_size,
            num_q_heads=32,
            num_kv_heads=2,
            head_dim=128,
            position="none",
        )
    head_dim = max(8, hidden_size // 4)
    num_q_heads = max(1, hidden_size // head_dim)
    return AttentionConfig(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=min(2, num_q_heads),
        head_dim=head_dim,
        position="none",
    )


def qwen38_mtp_head(
    *,
    hidden_size: int = _QWEN38_HIDDEN,
    vocab_size: int = _QWEN38_VOCAB,
    mlp_intermediate: int = _QWEN38_MTP_FFN,
    source_hidden_layer_index: int = _QWEN38_MTP_SOURCE_LAYER,
    num_prediction_steps: int = 1,
    embed: Embedding | None = None,
    lm_output: LanguageModelOutput | None = None,
    weight: Parameter | None = None,
) -> MtpStage:
    """Qwen3.8 MTP: fusion + one SwiGLU hidden layer + untied vocab projection."""
    resolved_embed = embed or Embedding(hidden_size, vocab_size)
    resolved_lm = lm_output or qwen38_lm_output(
        hidden_size=hidden_size, vocab_size=vocab_size, weight=weight
    )
    config = MtpStageConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_prediction_steps=num_prediction_steps,
        source_hidden_layer_index=source_hidden_layer_index,
        mlp_intermediate=mlp_intermediate,
        fusion_mode="concat_linear",
        tie_weights=False,
    )
    return MtpStage(config, embed=resolved_embed, lm_output=resolved_lm)


def nemotron_h_mtp_stage(
    *,
    hidden_size: int = _NEMOTRON_HIDDEN,
    vocab_size: int = _NEMOTRON_VOCAB,
    num_nextn_predict_layers: int = 2,
    source_hidden_layer_index: int = 51,
    num_prediction_steps: int = 1,
    embed: Embedding | None = None,
    lm_output: LanguageModelOutput | None = None,
    moe_factory: Callable[..., Module] | None = None,
    seq_len: int = 128,
) -> MtpStage:
    """Nemotron MTP: attention then MoE hybrid blocks (default block types)."""
    if num_nextn_predict_layers < 1:
        raise ValueError("num_nextn_predict_layers must be positive")
    layer_specs: list[LayerSpec] = []
    attn = _nemotron_mtp_attention(hidden_size)
    for index in range(num_nextn_predict_layers):
        if index == 0:
            layer_specs.append(LayerSpec(mixer="attention", attention=attn))
        else:
            layer_specs.append(
                LayerSpec(mixer="moe", moe_hidden_size=hidden_size)
            )
    resolved_embed = embed or Embedding(hidden_size, vocab_size)
    resolved_lm = lm_output or LanguageModelOutput(
        LanguageModelOutputConfig(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
    )
    config = MtpStageConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_prediction_steps=num_prediction_steps,
        source_hidden_layer_index=source_hidden_layer_index,
        layer_specs=tuple(layer_specs),
        fusion_mode="concat_linear",
    )
    factory = moe_factory
    if factory is None:
        from .moe_presets import nemotron_moe_block

        factory = lambda: nemotron_moe_block(  # noqa: E731
            hidden_size=hidden_size, seq_len=seq_len
        )
    return MtpStage(
        config,
        embed=resolved_embed,
        lm_output=resolved_lm,
        moe_factory=factory,
    )


__all__ = ["nemotron_h_mtp_stage", "qwen38_mtp_head"]
