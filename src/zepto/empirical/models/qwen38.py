"""Qwen3.8 model family for empirical Zepto vs HF twin studies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zepto.empirical.models.qwen38_core import (
    MROPE_SECTION,
    OPTION_KEYS,
    ROTARY_DIM,
    Qwen38FamilyCore,
    layer_specs_from_options,
    rope_config_from_options,
)

try:
    from transformers import Qwen3_5TextConfig as HFQwen38Config
    from transformers import Qwen3_5ForCausalLM as HFQwen38ForCausalLM
except ImportError:  # pragma: no cover - optional notebook dependency
    HFQwen38Config = None  # type: ignore[misc, assignment]
    HFQwen38ForCausalLM = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    import torch
    from torch import nn

    from zepto.analysis.horizon.spec import HorizonStep

TwinMode = Literal["scored_window", "transformers_defaults"]

SCORED_WINDOW_MIN_MPE = 32


def qwen38_hf_max_position_embeddings(seq_len: int, *, twin_mode: TwinMode) -> int:
    """HF ``max_position_embeddings`` for a twin mode that overrides MPE.

    ``scored_window`` matches smoke (``max(seq_len, 32)``).
    ``transformers_defaults`` does not override MPE.
    """
    if twin_mode == "scored_window":
        return max(seq_len, SCORED_WINDOW_MIN_MPE)
    raise ValueError(f"twin_mode {twin_mode!r} does not override MPE")


def _hf_config_accepts(name: str) -> bool:
    if HFQwen38Config is None:
        return False
    try:
        import inspect

        signature = inspect.signature(HFQwen38Config.__init__)
    except (TypeError, ValueError):
        return hasattr(HFQwen38Config, name)
    return name in signature.parameters


def _zepto_config(opts: dict[str, int]):
    from zepto.modules.models.qwen38 import Qwen38Config

    return Qwen38Config(
        hidden_size=opts["hidden_size"],
        num_layers=opts["num_layers"],
        vocab_size=opts["vocab_size"],
    )


class Qwen38Family(Qwen38FamilyCore):
    def build_zepto_infer_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ):
        from zepto.modules import Qwen38

        cfg = _zepto_config(opts)
        specs = layer_specs_from_options(opts)
        rope = rope_config_from_options(opts)

        def factory(_ctx):
            return Qwen38(
                config=cfg,
                seq_len=seq_len,
                include_vision=False,
                include_mtp=False,
                layer_specs=specs,
                rope_config=rope,
            )

        return factory

    def build_zepto_train_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ):
        from zepto.modules import Qwen38ForCausalLM

        cfg = _zepto_config(opts)
        specs = layer_specs_from_options(opts)
        rope = rope_config_from_options(opts)

        def factory(_ctx):
            return Qwen38ForCausalLM(
                config=cfg,
                seq_len=seq_len,
                layer_specs=specs,
                rope_config=rope,
            )

        return factory

    def zepto_infer_inputs(self, step: "HorizonStep", _ctx, _state) -> tuple:
        from zepto.compose import Tensor

        return (
            Tensor(
                shape=(step.batch, step.seq_len),
                semantic_type="token_ids",
                requires_grad=False,
            ),
        )

    def zepto_train_inputs(self, step: "HorizonStep", _ctx, _state) -> tuple:
        from zepto.analysis import inputs_from_token_ids_and_labels

        return inputs_from_token_ids_and_labels()(step, _ctx, _state)

    def build_hf_model(
        self,
        opts: dict[str, int],
        *,
        seq_len: int,
        precision: str,
        device: "torch.device",
        twin_mode: TwinMode = "scored_window",
    ) -> "nn.Module":
        import torch

        if HFQwen38ForCausalLM is None or HFQwen38Config is None:
            raise ImportError(
                "transformers with Qwen3.5 text CausalLM support is required for HF twin builds"
            )
        layer_types = (
            ["linear_attention"] * 3 + ["full_attention"]
        ) * (opts["num_layers"] // 4)
        cfg_kwargs: dict = {
            "vocab_size": opts["vocab_size"],
            "hidden_size": opts["hidden_size"],
            "intermediate_size": opts["intermediate_size"],
            "num_hidden_layers": opts["num_layers"],
            "num_attention_heads": opts["attn_num_q_heads"],
            "num_key_value_heads": opts["attn_num_kv_heads"],
            "head_dim": opts["attn_head_dim"],
            "linear_num_key_heads": opts["delta_num_qk_heads"],
            "linear_num_value_heads": opts["delta_num_v_heads"],
            "linear_key_head_dim": opts["delta_head_dim"],
            "linear_value_head_dim": opts["delta_head_dim"],
            "linear_conv_kernel_dim": 4,
            "layer_types": layer_types,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "use_cache": False,
            "attention_bias": False,
        }
        if _hf_config_accepts("rope_parameters"):
            cfg_kwargs["rope_parameters"] = {
                "rope_type": "default",
                "mrope_section": MROPE_SECTION,
                "partial_rotary_factor": ROTARY_DIM / opts["attn_head_dim"],
            }
        if _hf_config_accepts("mrope_section"):
            cfg_kwargs["mrope_section"] = MROPE_SECTION
        if twin_mode == "scored_window":
            cfg_kwargs["max_position_embeddings"] = qwen38_hf_max_position_embeddings(
                seq_len, twin_mode=twin_mode
            )
        elif twin_mode != "transformers_defaults":
            raise ValueError(f"unknown twin_mode: {twin_mode!r}")
        cfg = HFQwen38Config(**cfg_kwargs)
        if precision == "fp32":
            dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"unknown precision: {precision!r}")
        model = HFQwen38ForCausalLM(cfg)
        model.config.use_cache = False
        model.gradient_checkpointing_disable()
        model = model.to(device=device, dtype=dtype)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        return model

    def hf_forward_infer(self, model: "nn.Module", input_ids: "torch.Tensor"):
        return model(input_ids=input_ids, use_cache=False)

    def hf_forward_train(
        self,
        model: "nn.Module",
        input_ids: "torch.Tensor",
        labels: "torch.Tensor",
    ):
        return model(input_ids=input_ids, labels=labels, use_cache=False)


__all__ = [
    "SCORED_WINDOW_MIN_MPE",
    "Qwen38Family",
    "OPTION_KEYS",
    "TwinMode",
    "qwen38_hf_max_position_embeddings",
]
