"""Nemotron 3.5 Lightning hybrid full model with optional MTP."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor

from .embedding import Embedding
from .hybrid_decoder_block import HybridDecoderBlock
from .layer_spec import LayerSpec
from .materialized_causal_mask import MaterializedCausalMask
from .mixer_presets import nemotron_h_layer_specs
from .moe_presets import nemotron_moe_block
from .mtp_presets import nemotron_h_mtp_stage
from .mtp_stage import MtpStage
from .output_presets import nemotron_h_lm_output
from .rms_norm import RMSNorm


@dataclass(frozen=True, slots=True)
class NemotronHConfig:
    """Architecture constants for Nemotron 3.5 Lightning."""

    hidden_size: int = 2688
    num_layers: int = 52
    vocab_size: int = 131072


NEMOTRON_H_30B_A3B = NemotronHConfig()


class NemotronH(Module):
    """Mamba / attention / MoE hybrid trunk with optional MTP."""

    module_kind = "NemotronH"

    def __init__(
        self,
        *,
        config: NemotronHConfig = NEMOTRON_H_30B_A3B,
        seq_len: int,
        num_layers: int | None = None,
        include_mtp: bool = True,
        layer_specs: tuple[LayerSpec, ...] | None = None,
    ) -> None:
        super().__init__()
        cfg = config
        specs = layer_specs if layer_specs is not None else nemotron_h_layer_specs()
        layers = num_layers if num_layers is not None else cfg.num_layers
        if layers <= 0 or seq_len <= 0:
            raise ValueError("num_layers and seq_len must be positive")
        if layers > len(specs):
            raise ValueError("num_layers exceeds available layer specs")
        self.seq_len = seq_len
        self.config = cfg
        self._layer_specs = specs[:layers]
        self.embedding = Embedding(cfg.hidden_size, cfg.vocab_size)
        self.causal_mask = MaterializedCausalMask(seq_len)
        self.blocks: list[HybridDecoderBlock] = []
        for index, spec in enumerate(self._layer_specs):
            moe_factory = None
            if spec.mixer == "moe":
                moe_factory = lambda s=seq_len, h=cfg.hidden_size: nemotron_moe_block(
                    seq_len=s, hidden_size=h
                )
            block = HybridDecoderBlock(spec, moe_factory=moe_factory)
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_output = nemotron_h_lm_output(
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size
        )
        self.mtp: MtpStage | None = None
        if include_mtp:
            self.mtp = nemotron_h_mtp_stage(
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
                embed=self.embedding,
                lm_output=self.lm_output,
                seq_len=seq_len,
            )

    @classmethod
    def nemotron_h_30b_a3b(cls, seq_len: int) -> NemotronH:
        return cls(config=NEMOTRON_H_30B_A3B, seq_len=seq_len)

    def _run_block(
        self,
        block: HybridDecoderBlock,
        spec_index: int,
        hidden: Tensor,
        *,
        conv_in: Tensor | None = None,
        scan_in: Tensor | None = None,
    ) -> Tensor:
        spec = self._layer_specs[spec_index]
        if spec.mixer in ("mamba2", "gated_delta_net"):
            out = block(hidden, conv_in, scan_in)
            if isinstance(out, tuple):
                return out[0]
            return out  # type: ignore[return-value]
        if spec.mixer == "attention":
            mask = self.causal_mask()
            out = block(hidden, mask, None, None)
            if isinstance(out, tuple):
                return out[0]
            return out  # type: ignore[return-value]
        out = block(hidden)
        if isinstance(out, tuple):
            return out[0]
        return out  # type: ignore[return-value]

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.embedding(token_ids)
        for index, block in enumerate(self.blocks):
            hidden = self._run_block(block, index, hidden)
        return self.lm_output(self.final_norm(hidden))  # type: ignore[return-value]

    def mtp_forward(self, trunk_hidden: Tensor, mtp_token_ids: Tensor) -> Tensor:
        if self.mtp is None:
            raise ValueError("mtp_forward requires include_mtp=True")
        return self.mtp(trunk_hidden, mtp_token_ids)  # type: ignore[return-value]

    def forward_with_mtp(
        self,
        token_ids: Tensor,
        mtp_token_ids: Tensor,
        trunk_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        primary = self.forward(token_ids)
        assert self.mtp is not None
        aux = self.mtp_forward(trunk_hidden, mtp_token_ids)
        return primary, aux

    def forward_with_mixer_state(
        self,
        token_ids: Tensor,
        *,
        conv_states: tuple[Tensor | None, ...] | None = None,
        scan_states: tuple[Tensor | None, ...] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        hidden = self.embedding(token_ids)
        conv_states = conv_states or (None,) * len(self.blocks)
        scan_states = scan_states or (None,) * len(self.blocks)
        state_out: dict[str, Tensor] = {}
        mask = self.causal_mask()
        for index, block in enumerate(self.blocks):
            spec = self._layer_specs[index]
            if spec.mixer in ("mamba2", "gated_delta_net"):
                out = block(
                    hidden,
                    conv_states[index],
                    scan_states[index],
                )
                if isinstance(out, tuple):
                    hidden, conv_out, scan_out = out
                    state_out[f"conv.{index}"] = conv_out
                    state_out[f"scan.{index}"] = scan_out
                else:
                    hidden = out
            elif spec.mixer == "attention":
                out = block(hidden, mask, None, None)
                hidden = out[0] if isinstance(out, tuple) else out
            else:
                out = block(hidden)
                hidden = out[0] if isinstance(out, tuple) else out
        logits = self.lm_output(self.final_norm(hidden))  # type: ignore[assignment]
        return logits, state_out


__all__ = ["NEMOTRON_H_30B_A3B", "NemotronH", "NemotronHConfig"]
