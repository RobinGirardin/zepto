"""Locked free-knob catalog for the Qwen3.8 hybrid text-only frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from qwen38_validity.identities import Architecture, FreeKnobs, derive_architecture
from validity_common.catalog import WorkloadCatalog


@dataclass(frozen=True, slots=True)
class KnobCatalog:
    """Finite discrete supports for the generators in §5.3."""

    num_cycles: tuple[int, ...] = (1, 2)
    hidden_size: tuple[int, ...] = (256, 512)
    delta_qk_heads: tuple[int, ...] = (2, 4)
    delta_v_group: tuple[int, ...] = (3,)
    delta_head_dim: tuple[int, ...] = (32, 64)
    attn_kv_heads: tuple[int, ...] = (2, 4)
    attn_gqa_group: tuple[int, ...] = (6,)
    attn_head_dim: tuple[int, ...] = (64,)
    ffn_mult: tuple[float, ...] = (4.0, 5.25)
    vocab_size: tuple[int, ...] = (1024, 4096)
    include_large_vocab: bool = False
    large_vocab: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "num_cycles",
            "hidden_size",
            "delta_qk_heads",
            "delta_v_group",
            "delta_head_dim",
            "attn_kv_heads",
            "attn_gqa_group",
            "attn_head_dim",
            "ffn_mult",
            "vocab_size",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")

    def vocab_values(self) -> tuple[int, ...]:
        if self.include_large_vocab:
            if self.large_vocab is None:
                raise ValueError("include_large_vocab requires large_vocab")
            return (*self.vocab_size, self.large_vocab)
        return self.vocab_size

    def knobs(self) -> list[FreeKnobs]:
        """Cartesian product, documented loop order for a reproducible frame."""
        return [
            FreeKnobs(
                num_cycles=num_cycles,
                hidden_size=hidden_size,
                delta_qk_heads=delta_qk_heads,
                delta_v_group=delta_v_group,
                delta_head_dim=delta_head_dim,
                attn_kv_heads=attn_kv_heads,
                attn_gqa_group=attn_gqa_group,
                attn_head_dim=attn_head_dim,
                ffn_mult=ffn_mult,
                vocab_size=vocab_size,
            )
            for (
                num_cycles,
                hidden_size,
                delta_qk_heads,
                delta_v_group,
                delta_head_dim,
                attn_kv_heads,
                attn_gqa_group,
                attn_head_dim,
                ffn_mult,
                vocab_size,
            ) in product(
                self.num_cycles,
                self.hidden_size,
                self.delta_qk_heads,
                self.delta_v_group,
                self.delta_head_dim,
                self.attn_kv_heads,
                self.attn_gqa_group,
                self.attn_head_dim,
                self.ffn_mult,
                self.vocab_values(),
            )
        ]


@dataclass(frozen=True, slots=True)
class Catalog:
    knobs: KnobCatalog = field(default_factory=KnobCatalog)
    workload: WorkloadCatalog = field(default_factory=WorkloadCatalog)

    def architectures(self) -> list[tuple[FreeKnobs, Architecture]]:
        return [(knobs, derive_architecture(knobs)) for knobs in self.knobs.knobs()]

    def as_meta(self) -> dict[str, object]:
        return {
            "num_cycles": list(self.knobs.num_cycles),
            "hidden_size": list(self.knobs.hidden_size),
            "delta_qk_heads": list(self.knobs.delta_qk_heads),
            "delta_v_group": list(self.knobs.delta_v_group),
            "delta_head_dim": list(self.knobs.delta_head_dim),
            "attn_kv_heads": list(self.knobs.attn_kv_heads),
            "attn_gqa_group": list(self.knobs.attn_gqa_group),
            "attn_head_dim": list(self.knobs.attn_head_dim),
            "ffn_mult": list(self.knobs.ffn_mult),
            "vocab_size": list(self.knobs.vocab_values()),
            "include_large_vocab": self.knobs.include_large_vocab,
            "include_vision": False,
            "include_mtp": False,
            "mixer_schedule": "3delta+1gqa",
            "seq_len": list(self.workload.seq_len),
            "batch_size": list(self.workload.batch_size),
            "min_seq_len": self.workload.min_seq_len,
            "max_tokens": self.workload.max_tokens,
        }


__all__ = ["Catalog", "KnobCatalog", "WorkloadCatalog"]
