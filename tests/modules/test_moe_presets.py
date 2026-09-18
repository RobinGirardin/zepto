"""Preset factory dimension tests for MoE blocks."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.moe.moe_presets import (
    gpt_oss_moe_block,
    laguna_sparse_moe_block,
    nemotron_moe_block,
)


def test_gpt_oss_preset_dims() -> None:
    def factory(_ctx):
        block = gpt_oss_moe_block(seq_len=64)
        assert block.hidden_size == 2880
        assert block.router.num_experts == 32
        assert block.router.top_k == 4
        return block

    compose_graph(
        factory,
        (Tensor(shape=(8, 2880), requires_grad=True),),
    )


def test_laguna_preset_dims() -> None:
    def factory(_ctx):
        block = laguna_sparse_moe_block(seq_len=64)
        assert block.hidden_size == 2048
        assert block.router.num_experts == 256
        assert block.router.top_k == 8
        return block

    compose_graph(
        factory,
        (Tensor(shape=(8, 2048), requires_grad=True),),
    )


def test_nemotron_preset_dims() -> None:
    def factory(_ctx):
        block = nemotron_moe_block(seq_len=64)
        assert block.hidden_size == 2688
        assert block.router.num_experts == 128
        assert block.router.top_k == 6
        return block

    compose_graph(
        factory,
        (Tensor(shape=(8, 2688), requires_grad=True),),
    )
