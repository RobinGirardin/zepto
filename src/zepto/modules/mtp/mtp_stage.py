"""Auxiliary multi-token prediction stage orchestration."""

from __future__ import annotations

from collections.abc import Callable

from zepto.compose import Module, Tensor

from zepto.modules.layers.embedding import Embedding
from zepto.modules.blocks.hybrid_decoder_block import HybridDecoderBlock
from zepto.modules.output.language_model_output import LanguageModelOutput
from .mtp_config import MtpStageConfig
from .mtp_input_fusion import MtpInputFusion
from .mtp_mlp_block import MtpMlpBlock


class MtpStage(Module):
    """Fusion → body (SwiGLU or hybrid blocks) → language-model output."""

    module_kind = "MtpStage"

    def __init__(
        self,
        config: MtpStageConfig,
        *,
        embed: Embedding,
        lm_output: LanguageModelOutput,
        moe_factory: Callable[..., Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed = embed
        self.lm_output = lm_output
        self.fusion = MtpInputFusion(
            config.hidden_size, mode=config.fusion_mode
        )
        if config.layer_specs:
            self.blocks = tuple(
                HybridDecoderBlock(spec, moe_factory=moe_factory)
                for spec in config.layer_specs
            )
            self.mlp = None
        else:
            assert config.mlp_intermediate is not None
            self.mlp = MtpMlpBlock(config.hidden_size, config.mlp_intermediate)
            self.blocks = ()

    def forward(
        self,
        trunk_hidden: Tensor,
        token_ids: Tensor,
        *mixer_inputs: Tensor,
    ) -> Tensor:
        token_embed = self.embed(token_ids)
        u = self.fusion(trunk_hidden, token_embed)
        x = u
        if self.blocks:
            for block in self.blocks:
                x = block(x, *mixer_inputs)  # type: ignore[misc,assignment]
        else:
            x = self.mlp(x)  # type: ignore[misc,assignment]
        return self.lm_output(x)  # type: ignore[return-value]


__all__ = ["MtpStage"]
