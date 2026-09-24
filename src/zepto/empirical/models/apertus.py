"""Apertus model family for empirical Zepto vs HF twin studies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zepto.empirical.models.apertus_core import ApertusFamilyCore, OPTION_KEYS

try:
    from transformers import ApertusConfig, ApertusForCausalLM as HFApertusForCausalLM
except ImportError:  # pragma: no cover - optional notebook dependency
    ApertusConfig = None  # type: ignore[misc, assignment]
    HFApertusForCausalLM = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    import torch
    from torch import nn

    from zepto.analysis.horizon.spec import HorizonStep

TwinMode = Literal["scored_window", "apertus_parity", "transformers_defaults"]

APERTUS_ORIGINAL_MPE = 8192
SCORED_WINDOW_MIN_MPE = 32


def apertus_hf_max_position_embeddings(seq_len: int, *, twin_mode: TwinMode) -> int:
    """HF ``max_position_embeddings`` for a twin mode that overrides MPE.

    ``scored_window`` matches smoke (``max(seq_len, 32)``). ``apertus_parity``
    keeps llama3 valid without warning (``max(seq_len, 8193)``).
    ``transformers_defaults`` does not override MPE.
    """
    if twin_mode == "scored_window":
        return max(seq_len, SCORED_WINDOW_MIN_MPE)
    if twin_mode == "apertus_parity":
        return max(seq_len, APERTUS_ORIGINAL_MPE + 1)
    raise ValueError(f"twin_mode {twin_mode!r} does not override MPE")


def _module_kwargs(opts: dict[str, int], *, seq_len: int) -> dict[str, int]:
    return {
        "hidden_size": opts["hidden_size"],
        "intermediate_size": opts["intermediate_size"],
        "num_heads": opts["num_heads"],
        "num_kv_heads": opts["num_kv_heads"],
        "num_layers": opts["num_layers"],
        "vocab_size": opts["vocab_size"],
        "head_dim": opts["head_dim"],
        "seq_len": seq_len,
    }


class ApertusFamily(ApertusFamilyCore):
    def build_zepto_infer_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ):
        from zepto.modules import Apertus

        kwargs = _module_kwargs(opts, seq_len=seq_len)

        def factory(_ctx):
            return Apertus(**kwargs)

        return factory

    def build_zepto_train_module_factory(
        self, opts: dict[str, int], *, seq_len: int
    ):
        from zepto.modules import ApertusForCausalLM

        kwargs = _module_kwargs(opts, seq_len=seq_len)

        def factory(_ctx):
            return ApertusForCausalLM(**kwargs)

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

        if HFApertusForCausalLM is None or ApertusConfig is None:
            raise ImportError(
                "transformers with Apertus support is required for HF twin builds"
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
            "hidden_act": "xielu",
        }
        if twin_mode in ("scored_window", "apertus_parity"):
            cfg_kwargs["max_position_embeddings"] = apertus_hf_max_position_embeddings(
                seq_len, twin_mode=twin_mode
            )
        elif twin_mode != "transformers_defaults":
            raise ValueError(f"unknown twin_mode: {twin_mode!r}")
        cfg = ApertusConfig(**cfg_kwargs)
        if precision == "fp32":
            dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"unknown precision: {precision!r}")
        model = HFApertusForCausalLM(cfg)
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
    "APERTUS_ORIGINAL_MPE",
    "SCORED_WINDOW_MIN_MPE",
    "ApertusFamily",
    "OPTION_KEYS",
    "TwinMode",
    "apertus_hf_max_position_embeddings",
]
