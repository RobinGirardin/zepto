"""Checkpoint-faithful language-model output head factories."""

from __future__ import annotations

from zepto.compose import Parameter

from .language_model_output import LanguageModelOutput, LanguageModelOutputConfig
from .logit_soft_cap import LogitSoftCapConfig


def _untied_output(
    *,
    hidden_size: int,
    vocab_size: int,
    soft_cap: LogitSoftCapConfig | None = None,
    weight: Parameter | None = None,
) -> LanguageModelOutput:
    return LanguageModelOutput(
        LanguageModelOutputConfig(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tie_weights=False,
            soft_cap=soft_cap,
        ),
        weight=weight,
    )


def granite_lm_output(
    *,
    hidden_size: int = 4096,
    vocab_size: int = 100352,
    weight: Parameter | None = None,
) -> LanguageModelOutput:
    """Granite 4.2 30B untied LM head."""
    return _untied_output(
        hidden_size=hidden_size, vocab_size=vocab_size, weight=weight
    )


def gpt_oss_lm_output(
    *,
    hidden_size: int = 2880,
    vocab_size: int = 201088,
    weight: Parameter | None = None,
) -> LanguageModelOutput:
    """GPT-OSS 20B untied LM head."""
    return _untied_output(
        hidden_size=hidden_size, vocab_size=vocab_size, weight=weight
    )


def laguna_xs_lm_output(
    *,
    hidden_size: int = 2048,
    vocab_size: int = 100352,
    weight: Parameter | None = None,
) -> LanguageModelOutput:
    """Laguna XS 2.1 untied LM head."""
    return _untied_output(
        hidden_size=hidden_size, vocab_size=vocab_size, weight=weight
    )


def nemotron_h_lm_output(
    *,
    hidden_size: int = 2688,
    vocab_size: int = 131072,
    weight: Parameter | None = None,
) -> LanguageModelOutput:
    """Nemotron 3.5 Lightning untied LM head."""
    return _untied_output(
        hidden_size=hidden_size, vocab_size=vocab_size, weight=weight
    )


def muse_glimmer_lm_output(
    *,
    hidden_size: int = 6656,
    vocab_size: int = 202048,
    output_scale: float = 1.0,
    weight: Parameter | None = None,
) -> LanguageModelOutput:
    """Muse Glimmer 30B untied head with logit soft-cap 20."""
    return _untied_output(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        soft_cap=LogitSoftCapConfig(cap=20.0, output_scale=output_scale),
        weight=weight,
    )


def gemma4_lm_output(
    *,
    tied_weight: Parameter,
    hidden_size: int = 5376,
    vocab_size: int = 262144,
) -> LanguageModelOutput:
    """Gemma 4 31B tied head with logit soft-cap 30."""
    return LanguageModelOutput(
        LanguageModelOutputConfig(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            tie_weights=True,
            soft_cap=LogitSoftCapConfig(cap=30.0),
        ),
        weight=tied_weight,
    )


def qwen38_lm_output(
    *,
    hidden_size: int = 5120,
    vocab_size: int = 248320,
    weight: Parameter | None = None,
) -> LanguageModelOutput:
    """Qwen3.8-27B untied LM head."""
    return _untied_output(
        hidden_size=hidden_size, vocab_size=vocab_size, weight=weight
    )


__all__ = [
    "gemma4_lm_output",
    "gpt_oss_lm_output",
    "granite_lm_output",
    "laguna_xs_lm_output",
    "muse_glimmer_lm_output",
    "nemotron_h_lm_output",
    "qwen38_lm_output",
]
