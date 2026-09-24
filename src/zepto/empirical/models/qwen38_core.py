"""Qwen3.8 hybrid architecture options (no torch / heavy Zepto imports)."""

from __future__ import annotations

from zepto.empirical.models.gqa_dense_core import round_to_multiple

OPTION_KEYS: tuple[str, ...] = (
    "hidden_size",
    "intermediate_size",
    "num_layers",
    "vocab_size",
    "delta_num_qk_heads",
    "delta_num_v_heads",
    "delta_head_dim",
    "attn_num_q_heads",
    "attn_num_kv_heads",
    "attn_head_dim",
)

ROTARY_DIM = 64
MROPE_SECTION = (8, 12, 12)
CYCLE = ("gated_delta_net", "gated_delta_net", "gated_delta_net", "attention")

_GENERATOR_KEYS = ("num_cycles", "delta_v_group", "attn_gqa_group", "ffn_mult")


class Qwen38FamilyCore:
    model_id = "qwen38"

    def option_keys(self) -> tuple[str, ...]:
        return OPTION_KEYS

    def derive_options(self, sampled: dict[str, int | float]) -> dict[str, int]:
        if any(key in sampled for key in _GENERATOR_KEYS):
            hidden_size = int(sampled["hidden_size"])
            opts = {
                "hidden_size": hidden_size,
                "intermediate_size": round_to_multiple(
                    float(sampled["ffn_mult"]) * hidden_size
                ),
                "num_layers": 4 * int(sampled["num_cycles"]),
                "vocab_size": int(sampled["vocab_size"]),
                "delta_num_qk_heads": int(sampled["delta_qk_heads"]),
                "delta_num_v_heads": int(sampled["delta_qk_heads"])
                * int(sampled["delta_v_group"]),
                "delta_head_dim": int(sampled["delta_head_dim"]),
                "attn_num_q_heads": int(sampled["attn_kv_heads"])
                * int(sampled["attn_gqa_group"]),
                "attn_num_kv_heads": int(sampled["attn_kv_heads"]),
                "attn_head_dim": int(sampled["attn_head_dim"]),
            }
        else:
            opts = {key: int(sampled[key]) for key in OPTION_KEYS}
        self.validate_options(opts)
        return opts

    def validate_options(self, opts: dict[str, int]) -> None:
        if opts["num_layers"] <= 0 or opts["num_layers"] % 4 != 0:
            raise ValueError("num_layers must form complete 3:1 cycles")
        if opts["delta_num_v_heads"] % opts["delta_num_qk_heads"] != 0:
            raise ValueError("DeltaNet V grouping requires v heads divisible by qk heads")
        if opts["attn_num_q_heads"] % opts["attn_num_kv_heads"] != 0:
            raise ValueError("GQA grouping requires q heads divisible by kv heads")
        if opts["delta_head_dim"] % 2 != 0 or opts["attn_head_dim"] % 2 != 0:
            raise ValueError("even head_dim required (RoPE pairs dimensions)")
        if opts["attn_head_dim"] < ROTARY_DIM:
            raise ValueError(f"attn_head_dim must be >= rotary_dim {ROTARY_DIM}")
        for key in ("hidden_size", "intermediate_size", "vocab_size"):
            if opts[key] <= 0:
                raise ValueError(f"{key} must be positive")


def layer_specs_from_options(opts: dict[str, int]):
    from zepto.modules.attention.attention_config import gated_gqa
    from zepto.modules.mixers.layer_spec import LayerSpec
    from zepto.modules.mixers.mixer_config import GatedDeltaNetConfig

    delta = GatedDeltaNetConfig(
        hidden_size=opts["hidden_size"],
        num_qk_heads=opts["delta_num_qk_heads"],
        num_v_heads=opts["delta_num_v_heads"],
        head_dim=opts["delta_head_dim"],
    )
    attn = gated_gqa(
        opts["hidden_size"],
        opts["attn_num_q_heads"],
        opts["attn_num_kv_heads"],
        head_dim=opts["attn_head_dim"],
        gate="sigmoid",
    )
    cycle = (
        LayerSpec(
            mixer="gated_delta_net",
            gated_delta=delta,
            ffn="swiglu",
            swiglu_intermediate=opts["intermediate_size"],
        ),
        LayerSpec(
            mixer="gated_delta_net",
            gated_delta=delta,
            ffn="swiglu",
            swiglu_intermediate=opts["intermediate_size"],
        ),
        LayerSpec(
            mixer="gated_delta_net",
            gated_delta=delta,
            ffn="swiglu",
            swiglu_intermediate=opts["intermediate_size"],
        ),
        LayerSpec(
            mixer="attention",
            attention=attn,
            ffn="swiglu",
            swiglu_intermediate=opts["intermediate_size"],
        ),
    )
    n_cycles = opts["num_layers"] // 4
    return cycle * n_cycles


def rope_config_from_options(opts: dict[str, int]):
    from zepto.modules.position.rope_config import qwen3_vl_mrope

    return qwen3_vl_mrope(head_dim=opts["attn_head_dim"], rotary_dim=ROTARY_DIM)


__all__ = [
    "CYCLE",
    "MROPE_SECTION",
    "OPTION_KEYS",
    "ROTARY_DIM",
    "Qwen38FamilyCore",
    "layer_specs_from_options",
    "rope_config_from_options",
    "round_to_multiple",
]
