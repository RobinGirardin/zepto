"""Hybrid identities for the Qwen3.8 text-only catalog.

Residual width is free. Do not import validity_common.identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from zepto.empirical.models.qwen38_core import (
    OPTION_KEYS,
    Qwen38FamilyCore,
    round_to_multiple,
)


@dataclass(frozen=True, slots=True)
class FreeKnobs:
    """Quantities that may be drawn. They never appear on the twin."""

    num_cycles: int
    hidden_size: int
    delta_qk_heads: int
    delta_v_group: int
    delta_head_dim: int
    attn_kv_heads: int
    attn_gqa_group: int
    attn_head_dim: int
    ffn_mult: float
    vocab_size: int


@dataclass(frozen=True, slots=True)
class Architecture:
    """Option dict handed to Zepto and HuggingFace."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    vocab_size: int
    delta_num_qk_heads: int
    delta_num_v_heads: int
    delta_head_dim: int
    attn_num_q_heads: int
    attn_num_kv_heads: int
    attn_head_dim: int

    def as_options(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in OPTION_KEYS}


def derive_architecture(knobs: FreeKnobs) -> Architecture:
    """Turn free knobs into a valid hybrid option dict (§5.4)."""
    opts = Qwen38FamilyCore().derive_options(asdict(knobs))
    architecture = Architecture(**opts)
    assert_identities(architecture)
    return architecture


def assert_identities(architecture: Architecture) -> None:
    """§5.5. After derivation these are identities, not rejection rules."""
    Qwen38FamilyCore().validate_options(architecture.as_options())


__all__ = [
    "Architecture",
    "FreeKnobs",
    "assert_identities",
    "derive_architecture",
    "round_to_multiple",
]
