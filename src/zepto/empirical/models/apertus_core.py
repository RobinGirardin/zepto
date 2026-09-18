"""Apertus architecture options (no torch / heavy Zepto imports)."""

from __future__ import annotations

OPTION_KEYS: tuple[str, ...] = (
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "num_heads",
    "num_kv_heads",
    "num_layers",
    "vocab_size",
)


class ApertusFamilyCore:
    model_id = "apertus"

    def option_keys(self) -> tuple[str, ...]:
        return OPTION_KEYS

    def derive_options(self, sampled: dict[str, int]) -> dict[str, int]:
        hidden_size = sampled["num_heads"] * sampled["head_dim"]
        return {
            "hidden_size": hidden_size,
            "head_dim": sampled["head_dim"],
            "intermediate_size": sampled["intermediate_size"],
            "num_heads": sampled["num_heads"],
            "num_kv_heads": sampled["num_kv_heads"],
            "num_layers": sampled["num_layers"],
            "vocab_size": sampled["vocab_size"],
        }

    def validate_options(self, opts: dict[str, int]) -> None:
        if opts["num_heads"] % opts["num_kv_heads"] != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if opts["hidden_size"] != opts["num_heads"] * opts["head_dim"]:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        for key in ("hidden_size", "intermediate_size", "vocab_size", "num_layers"):
            if opts[key] <= 0:
                raise ValueError(f"{key} must be positive")
