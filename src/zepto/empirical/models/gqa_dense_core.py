"""Shared GQA-dense architecture options (no torch / heavy Zepto imports)."""

from __future__ import annotations


def round_to_multiple(value: float, multiple: int = 64) -> int:
    """Round ``value`` to the nearest multiple of ``multiple``."""
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return int(round(value / multiple) * multiple)


OPTION_KEYS: tuple[str, ...] = (
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "num_heads",
    "num_kv_heads",
    "num_layers",
    "vocab_size",
)


class GqaDenseFamilyCore:
    model_id: str

    def option_keys(self) -> tuple[str, ...]:
        return OPTION_KEYS

    def derive_options(self, sampled: dict[str, int | float]) -> dict[str, int]:
        if "gqa_group" in sampled:
            num_kv_heads = int(sampled["num_kv_heads"])
            num_heads = num_kv_heads * int(sampled["gqa_group"])
            head_dim = int(sampled["head_dim"])
            hidden_size = num_heads * head_dim
            intermediate_size = round_to_multiple(float(sampled["ffn_mult"]) * hidden_size)
            return {
                "hidden_size": hidden_size,
                "head_dim": head_dim,
                "intermediate_size": intermediate_size,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "num_layers": int(sampled["num_layers"]),
                "vocab_size": int(sampled["vocab_size"]),
            }
        hidden_size = int(sampled["num_heads"]) * int(sampled["head_dim"])
        return {
            "hidden_size": hidden_size,
            "head_dim": int(sampled["head_dim"]),
            "intermediate_size": int(sampled["intermediate_size"]),
            "num_heads": int(sampled["num_heads"]),
            "num_kv_heads": int(sampled["num_kv_heads"]),
            "num_layers": int(sampled["num_layers"]),
            "vocab_size": int(sampled["vocab_size"]),
        }

    def validate_options(self, opts: dict[str, int]) -> None:
        if opts["head_dim"] % 2 != 0:
            raise ValueError("head_dim must be even (RoPE pairs dimensions)")
        if opts["num_heads"] % opts["num_kv_heads"] != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if opts["hidden_size"] != opts["num_heads"] * opts["head_dim"]:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        for key in ("hidden_size", "intermediate_size", "vocab_size", "num_layers"):
            if opts[key] <= 0:
                raise ValueError(f"{key} must be positive")
