"""Granite model family for empirical Zepto vs HF twin studies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zepto.empirical.models.granite_core import GraniteFamilyCore, OPTION_KEYS

try:
    from transformers import GraniteConfig as HFGraniteConfig
    from transformers import GraniteForCausalLM as HFGraniteForCausalLM
except ImportError:  # pragma: no cover - optional notebook dependency
    HFGraniteConfig = None  # type: ignore[misc, assignment]
    HFGraniteForCausalLM = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    import torch
    from torch import nn

    from zepto.analysis.horizon.spec import HorizonStep

TwinMode = Literal["scored_window", "transformers_defaults"]

SCORED_WINDOW_MIN_MPE = 32
GRANITE_ROPE_THETA = 50_000_000.0


def granite_hf_max_position_embeddings(seq_len: int, *, twin_mode: TwinMode) -> int:
    """HF ``max_position_embeddings`` for a twin mode that overrides MPE.

    ``scored_window`` matches smoke (``max(seq_len, 32)``).
    ``transformers_defaults`` does not override MPE.
    """
    if twin_mode == "scored_window":
        return max(seq_len, SCORED_WINDOW_MIN_MPE)
    raise ValueError(f"twin_mode {twin_mode!r} does not override MPE")


def _zepto_config(opts: dict[str, int]):
    from zepto.modules.models.granite import GraniteConfig

    return GraniteConfig(
        hidden_size=opts["hidden_size"],
        intermediate_size=opts["intermediate_size"],
        num_q_heads=opts["num_heads"],
        num_kv_heads=opts["num_kv_heads"],
        head_dim=opts["head_dim"],
        num_layers=opts["num_layers"],
        vocab_size=opts["vocab_size"],
    )


def _hf_config_accepts(name: str) -> bool:
    if HFGraniteConfig is None:
        return False
    try:
        import inspect

        signature = inspect.signature(HFGraniteConfig.__init__)
    except (TypeError, ValueError):
        return hasattr(HFGraniteConfig, name)
    return name in signature.parameters


class GraniteFamily(GraniteFamilyCore):
    def build_zepto_infer_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ):
        from zepto.modules import Granite

        cfg = _zepto_config(opts)

        def factory(_ctx):
            return Granite(config=cfg, seq_len=seq_len)

        return factory

    def build_zepto_train_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ):
        from zepto.modules import GraniteForCausalLM

        cfg = _zepto_config(opts)

        def factory(_ctx):
            return GraniteForCausalLM(config=cfg, seq_len=seq_len)

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

        if HFGraniteForCausalLM is None or HFGraniteConfig is None:
            raise ImportError(
                "transformers with Granite support is required for HF twin builds"
            )
        cfg_kwargs: dict = {
            "vocab_size": opts["vocab_size"],
            "hidden_size": opts["hidden_size"],
            "intermediate_size": opts["intermediate_size"],
            "num_hidden_layers": opts["num_layers"],
            "num_attention_heads": opts["num_heads"],
            "num_key_value_heads": opts["num_kv_heads"],
            "tie_word_embeddings": False,
            "use_cache": False,
            "attention_bias": False,
            "hidden_act": "silu",
            "rope_theta": GRANITE_ROPE_THETA,
        }
        for name, value in (
            ("residual_multiplier", 1.0),
            ("embedding_multiplier", 1.0),
            ("attention_multiplier", 1.0),
            ("logits_scaling", 1.0),
            ("head_dim", opts["head_dim"]),
        ):
            if _hf_config_accepts(name):
                cfg_kwargs[name] = value
        if twin_mode == "scored_window":
            cfg_kwargs["max_position_embeddings"] = granite_hf_max_position_embeddings(
                seq_len, twin_mode=twin_mode
            )
        elif twin_mode != "transformers_defaults":
            raise ValueError(f"unknown twin_mode: {twin_mode!r}")
        cfg = HFGraniteConfig(**cfg_kwargs)
        if precision == "fp32":
            dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"unknown precision: {precision!r}")
        model = HFGraniteForCausalLM(cfg)
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
    "GRANITE_ROPE_THETA",
    "SCORED_WINDOW_MIN_MPE",
    "GraniteFamily",
    "OPTION_KEYS",
    "TwinMode",
    "granite_hf_max_position_embeddings",
]
