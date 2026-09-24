"""Apertus 1.5 architecture options (no torch / heavy Zepto imports)."""

from __future__ import annotations

from zepto.empirical.models.gqa_dense_core import (
    OPTION_KEYS as GQA_OPTION_KEYS,
    GqaDenseFamilyCore,
    round_to_multiple,
)

OPTION_KEYS: tuple[str, ...] = (*GQA_OPTION_KEYS, "output_vocab_size")


class Apertus15FamilyCore(GqaDenseFamilyCore):
    model_id = "apertus15"

    def option_keys(self) -> tuple[str, ...]:
        return OPTION_KEYS

    def derive_options(self, sampled: dict[str, int | float]) -> dict[str, int]:
        opts = super().derive_options(sampled)
        if "output_vocab_size" in sampled:
            opts["output_vocab_size"] = int(sampled["output_vocab_size"])
        else:
            opts["output_vocab_size"] = opts["vocab_size"]
        return opts

    def validate_options(self, opts: dict[str, int]) -> None:
        super().validate_options(opts)
        output_vocab_size = opts["output_vocab_size"]
        if not (1 <= output_vocab_size <= opts["vocab_size"]):
            raise ValueError(
                "output_vocab_size must satisfy 1 <= output_vocab_size <= vocab_size"
            )


__all__ = [
    "OPTION_KEYS",
    "Apertus15FamilyCore",
    "round_to_multiple",
]
