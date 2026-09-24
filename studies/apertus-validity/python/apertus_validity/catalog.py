"""Locked free-knob and workload catalogs. Framework §7.3–7.4.

T_max (max_tokens) and the optional 131072 vocabulary stay off until the
protocol-matched pilot locks the executable frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from apertus_validity.identities import Architecture, FreeKnobs, derive_architecture
from apertus_validity.protocol import MIN_SEQ_LEN, OPTIONAL_LARGE_VOCAB


@dataclass(frozen=True, slots=True)
class KnobCatalog:
    """Finite discrete supports for the generators in §7.3."""

    gqa_group: tuple[int, ...] = (1, 2, 4)
    num_kv_heads: tuple[int, ...] = (2, 4, 8)
    head_dim: tuple[int, ...] = (64, 128)
    num_layers: tuple[int, ...] = (2, 4, 8)
    ffn_mult: tuple[float, ...] = (4.0, 5.25)
    vocab_size: tuple[int, ...] = (1024, 4096)
    include_large_vocab: bool = False

    def __post_init__(self) -> None:
        for name in (
            "gqa_group",
            "num_kv_heads",
            "head_dim",
            "num_layers",
            "ffn_mult",
            "vocab_size",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")

    def vocab_values(self) -> tuple[int, ...]:
        if self.include_large_vocab:
            return (*self.vocab_size, OPTIONAL_LARGE_VOCAB)
        return self.vocab_size

    def knobs(self) -> list[FreeKnobs]:
        """Cartesian product, documented loop order for a reproducible frame."""
        return [
            FreeKnobs(
                num_kv_heads=num_kv_heads,
                gqa_group=gqa_group,
                head_dim=head_dim,
                ffn_mult=ffn_mult,
                num_layers=num_layers,
                vocab_size=vocab_size,
            )
            for gqa_group, num_kv_heads, head_dim, num_layers, ffn_mult, vocab_size in product(
                self.gqa_group,
                self.num_kv_heads,
                self.head_dim,
                self.num_layers,
                self.ffn_mult,
                self.vocab_values(),
            )
        ]


@dataclass(frozen=True, slots=True)
class WorkloadCatalog:
    """Discrete (B, S) support. S ≥ 32 matches the windowed RoPE of §1."""

    seq_len: tuple[int, ...] = (32, 64, 128)
    batch_size: tuple[int, ...] = (1, 2, 4)
    min_seq_len: int = MIN_SEQ_LEN
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.seq_len:
            raise ValueError("seq_len must not be empty")
        if not self.batch_size:
            raise ValueError("batch_size must not be empty")
        if self.min_seq_len <= 0:
            raise ValueError("min_seq_len must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def feasible_pairs(self) -> list[tuple[int, int]]:
        """Pairs `(batch_size, seq_len)` that pass the RoPE window and T_max."""
        pairs = [
            (batch, seq)
            for batch in self.batch_size
            for seq in self.seq_len
            if seq >= self.min_seq_len
            and (self.max_tokens is None or batch * seq <= self.max_tokens)
        ]
        if not pairs:
            raise ValueError(
                "workload catalog has no feasible (batch_size, seq_len) pairs"
            )
        return pairs


@dataclass(frozen=True, slots=True)
class Catalog:
    knobs: KnobCatalog = field(default_factory=KnobCatalog)
    workload: WorkloadCatalog = field(default_factory=WorkloadCatalog)

    def architectures(self) -> list[tuple[FreeKnobs, Architecture]]:
        return [(knobs, derive_architecture(knobs)) for knobs in self.knobs.knobs()]

    def as_meta(self) -> dict[str, object]:
        return {
            "gqa_group": list(self.knobs.gqa_group),
            "num_kv_heads": list(self.knobs.num_kv_heads),
            "head_dim": list(self.knobs.head_dim),
            "num_layers": list(self.knobs.num_layers),
            "ffn_mult": list(self.knobs.ffn_mult),
            "vocab_size": list(self.knobs.vocab_values()),
            "include_large_vocab": self.knobs.include_large_vocab,
            "seq_len": list(self.workload.seq_len),
            "batch_size": list(self.workload.batch_size),
            "min_seq_len": self.workload.min_seq_len,
            "max_tokens": self.workload.max_tokens,
        }
