"""Qwen3.8-27B vision-language full model with optional MTP."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor

from zepto.modules.layers.embedding import Embedding
from zepto.modules.attention.materialized_causal_mask import MaterializedCausalMask
from zepto.modules.mixers.mixer_presets import qwen35_language_layer_specs
from zepto.modules.mtp.mtp_presets import qwen38_mtp_head
from zepto.modules.mtp.mtp_stage import MtpStage
from zepto.modules.multimodal.multimodal_language_model import embed_multimodal_sequence
from zepto.modules.position.multimodal_rope_materialize import MultimodalRoPEMaterialize
from zepto.modules.multimodal.multimodal_sequence_builder import MultimodalSequenceBuilder
from zepto.modules.output.output_presets import qwen38_lm_output
from zepto.modules.blocks.qwen35_language_decoder_block import Qwen35LanguageDecoderBlock
from zepto.modules.layers.rms_norm import RMSNorm
from zepto.modules.position.rope_config import qwen3_vl_mrope
from zepto.modules.vision.vision_presets import Qwen3VLVisionTower


@dataclass(frozen=True, slots=True)
class Qwen38Config:
    """Architecture constants for Qwen3.8-27B."""

    hidden_size: int = 5120
    num_layers: int = 64
    vocab_size: int = 248320


QWEN38_27B = Qwen38Config()


class Qwen38(Module):
    """Language hybrid trunk + optional vision tower and MTP auxiliary head."""

    module_kind = "Qwen38"

    def __init__(
        self,
        *,
        config: Qwen38Config = QWEN38_27B,
        seq_len: int,
        num_layers: int | None = None,
        include_vision: bool = True,
        include_mtp: bool = True,
        mtp_source_hidden_layer_index: int | None = None,
        vision_num_layers: int = 2,
        grid_t: int = 1,
        grid_h: int = 8,
        grid_w: int = 8,
        layer_specs: tuple | None = None,
    ) -> None:
        super().__init__()
        cfg = config
        specs = layer_specs if layer_specs is not None else qwen35_language_layer_specs()
        layers = num_layers if num_layers is not None else cfg.num_layers
        if layers <= 0 or seq_len <= 0:
            raise ValueError("num_layers and seq_len must be positive")
        if layers > len(specs):
            raise ValueError("num_layers exceeds available layer specs")
        self.seq_len = seq_len
        self.config = cfg
        self._layer_specs = specs[:layers]
        self.vision: Qwen3VLVisionTower | None = None
        self.sequence_builder: MultimodalSequenceBuilder | None = None
        if include_vision:
            self.vision = Qwen3VLVisionTower(
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                num_layers=vision_num_layers,
            )
            self.sequence_builder = MultimodalSequenceBuilder(
                cfg.hidden_size, cfg.vocab_size
            )
            self.embedding = self.sequence_builder.embedding
        else:
            self.embedding = Embedding(cfg.hidden_size, cfg.vocab_size)
        self.causal_mask = MaterializedCausalMask(seq_len)
        self.mrope = MultimodalRoPEMaterialize(seq_len, qwen3_vl_mrope())
        self.blocks: list[Qwen35LanguageDecoderBlock] = []
        for index, spec in enumerate(self._layer_specs):
            block = Qwen35LanguageDecoderBlock(spec)
            self.register_module(f"layers.{index}", block)
            self.blocks.append(block)
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.lm_output = qwen38_lm_output(
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size
        )
        self.mtp: MtpStage | None = None
        if include_mtp:
            tap_index = (
                mtp_source_hidden_layer_index
                if mtp_source_hidden_layer_index is not None
                else 47
            )
            if tap_index < 0 or tap_index >= layers:
                raise ValueError(
                    "mtp_source_hidden_layer_index must be in [0, num_layers)"
                )
            self.mtp = qwen38_mtp_head(
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
                embed=self.embedding,
                lm_output=self.lm_output,
                source_hidden_layer_index=tap_index,
            )

    @classmethod
    def qwen38_27b(cls, seq_len: int) -> Qwen38:
        return cls(config=QWEN38_27B, seq_len=seq_len)

    def _run_block(
        self,
        block: Qwen35LanguageDecoderBlock,
        spec_index: int,
        hidden: Tensor,
    ) -> Tensor:
        spec = self._layer_specs[spec_index]
        if spec.mixer == "gated_delta_net":
            out = block(hidden)
            if isinstance(out, tuple):
                return out[0]
            return out  # type: ignore[return-value]
        cos, sin = self.mrope()
        mask = self.causal_mask()
        out = block(hidden, mask, cos, sin)
        if isinstance(out, tuple):
            return out[0]
        return out  # type: ignore[return-value]

    def _language_trunk(
        self,
        hidden: Tensor,
        *,
        capture_layer_index: int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        captured: Tensor | None = None
        for index, block in enumerate(self.blocks):
            hidden = self._run_block(block, index, hidden)
            if capture_layer_index is not None and index == capture_layer_index:
                captured = hidden
        return hidden, captured

    def forward(
        self,
        token_ids: Tensor,
        placeholder_indices: Tensor | None = None,
        vision_pixels: Tensor | None = None,
    ) -> Tensor:
        hidden = embed_multimodal_sequence(
            self,
            token_ids,
            placeholder_indices=placeholder_indices,
            vision_pixels=vision_pixels,
        )
        hidden, _ = self._language_trunk(hidden)
        return self.lm_output(self.final_norm(hidden))  # type: ignore[return-value]

    def mtp_forward(self, trunk_hidden: Tensor, mtp_token_ids: Tensor) -> Tensor:
        if self.mtp is None:
            raise ValueError("mtp_forward requires include_mtp=True")
        return self.mtp(trunk_hidden, mtp_token_ids)  # type: ignore[return-value]

    def forward_with_mtp(
        self,
        token_ids: Tensor,
        mtp_token_ids: Tensor,
        *,
        placeholder_indices: Tensor | None = None,
        vision_pixels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.mtp is None:
            raise ValueError("forward_with_mtp requires include_mtp=True")
        hidden = embed_multimodal_sequence(
            self,
            token_ids,
            placeholder_indices=placeholder_indices,
            vision_pixels=vision_pixels,
        )
        tap = self.mtp.config.source_hidden_layer_index
        hidden, mtp_hidden = self._language_trunk(
            hidden, capture_layer_index=tap
        )
        if mtp_hidden is None:
            raise RuntimeError(
                f"MTP tap missing at layer {tap}; check num_layers and "
                "mtp_source_hidden_layer_index"
            )
        primary = self.lm_output(self.final_norm(hidden))  # type: ignore[assignment]
        aux = self.mtp(mtp_hidden, mtp_token_ids)  # type: ignore[assignment]
        return primary, aux

    def vision_forward(self, vision_pixels: Tensor) -> Tensor:
        if self.vision is None:
            raise ValueError("vision_forward requires include_vision=True")
        return self.vision(vision_pixels)  # type: ignore[return-value]

    def forward_with_mixer_state(
        self,
        token_ids: Tensor,
        *,
        conv_states: tuple[Tensor | None, ...] | None = None,
        scan_states: tuple[Tensor | None, ...] | None = None,
        placeholder_indices: Tensor | None = None,
        vision_pixels: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Prefill/decode path with conv and scan carry for stateful mixers."""
        hidden = embed_multimodal_sequence(
            self,
            token_ids,
            placeholder_indices=placeholder_indices,
            vision_pixels=vision_pixels,
        )
        conv_states = conv_states or (None,) * len(self.blocks)
        scan_states = scan_states or (None,) * len(self.blocks)
        state_out: dict[str, Tensor] = {}
        cos, sin = self.mrope()
        mask = self.causal_mask()
        for index, block in enumerate(self.blocks):
            spec = self._layer_specs[index]
            if spec.mixer == "gated_delta_net":
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
            else:
                out = block(hidden, mask, cos, sin)
                hidden = out[0] if isinstance(out, tuple) else out
        logits = self.lm_output(self.final_norm(hidden))  # type: ignore[assignment]
        return logits, state_out


__all__ = ["QWEN38_27B", "Qwen38", "Qwen38Config"]
