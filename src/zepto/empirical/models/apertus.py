"""Apertus model family for empirical Zepto vs HF twin studies."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

        return (Tensor(shape=(step.seq_len,)),)

    def zepto_train_inputs(self, step: "HorizonStep", _ctx, _state) -> tuple:
        from zepto.compose import Tensor

        seq = step.seq_len
        return (
            Tensor(shape=(seq,)),
            Tensor(shape=(seq,), semantic_type="labels", requires_grad=False),
        )

    def build_hf_model(
        self, opts: dict[str, int], *, precision: str, device: "torch.device"
    ) -> "nn.Module":
        import torch

        if HFApertusForCausalLM is None or ApertusConfig is None:
            raise ImportError(
                "transformers with Apertus support is required for HF twin builds"
            )
        # HF twin: use transformers Apertus defaults (max_position_embeddings,
        # rope_parameters / YaRN) — do not override with a short context window.
        cfg = ApertusConfig(
            vocab_size=opts["vocab_size"],
            hidden_size=opts["hidden_size"],
            intermediate_size=opts["intermediate_size"],
            num_hidden_layers=opts["num_layers"],
            num_attention_heads=opts["num_heads"],
            num_key_value_heads=opts["num_kv_heads"],
            tie_word_embeddings=False,
            use_cache=False,
            attention_bias=False,
            hidden_act="xielu",
        )
        if precision == "fp32":
            dtype = torch.float32
        elif precision in ("fp16", "mixed"):
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


__all__ = ["ApertusFamily", "OPTION_KEYS"]
