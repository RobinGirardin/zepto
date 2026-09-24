"""Architectural identities: sample generators, compute dependents.

Framework §7.1–7.2. Independent integer draws on hidden_size, num_heads,
head_dim, and intermediate_size almost never form a valid GQA-dense graph.
A free knob is a quantity we may draw. Everything else is a function of
those draws, so a valid option dict is produced without rejection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FreeKnobs:
    """Quantities that may be drawn. They never appear on the twin."""

    num_kv_heads: int
    gqa_group: int
    head_dim: int
    ffn_mult: float
    num_layers: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class Architecture:
    """Option dict handed to Zepto and HuggingFace."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int
    head_dim: int

    def as_options(self) -> dict[str, int]:
        return asdict(self)


def round_to_multiple(value: float, multiple: int = 64) -> int:
    """Nearest multiple. Used for intermediate_size (§7.1 identity 4)."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return int(round(value / multiple) * multiple)


def derive_architecture(knobs: FreeKnobs) -> Architecture:
    """Turn free knobs into a valid GQA-dense option dict.

    Worked example (§7.2): kv=4, gqa_group=2, head_dim=128, ffn_mult=5.25
    → num_heads=8, hidden_size=1024, intermediate_size=5376.
    """
    num_heads = knobs.num_kv_heads * knobs.gqa_group
    hidden_size = num_heads * knobs.head_dim
    intermediate_size = round_to_multiple(knobs.ffn_mult * hidden_size)
    architecture = Architecture(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=knobs.num_kv_heads,
        num_layers=knobs.num_layers,
        vocab_size=knobs.vocab_size,
        head_dim=knobs.head_dim,
    )
    assert_identities(architecture)
    return architecture


def assert_identities(architecture: Architecture) -> None:
    """§7.1. After derivation these are identities, not rejection rules."""
    if architecture.head_dim % 2 != 0:
        raise ValueError("head_dim must be even (RoPE pairs dimensions)")
    if architecture.num_heads % architecture.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if architecture.hidden_size != architecture.num_heads * architecture.head_dim:
        raise ValueError("hidden_size must equal num_heads * head_dim")
    for key in ("hidden_size", "intermediate_size", "vocab_size", "num_layers"):
        if getattr(architecture, key) <= 0:
            raise ValueError(f"{key} must be positive")
