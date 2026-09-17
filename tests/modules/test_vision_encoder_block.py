"""Vision encoder block compose tests."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.vision_attention_config import (
    gemma4_vision_attention,
    muse_glimmer_vision_attention,
    qwen3_vl_vision_attention,
)
from zepto.modules.vision_encoder_block import VisionEncoderBlock
from zepto.modules.vision_masks import (
    MaterializedBidirectionalMask,
    SlidingWindowBidirectionalMask,
)


def _families(graph) -> tuple[str, ...]:
    return tuple(graph.node(node_id).operation_family for node_id in graph.nodes)


def test_global_vision_block_no_causal_mask() -> None:
    seq, d = 16, 1152
    head_dim = 72
    cfg = qwen3_vl_vision_attention()
    graph = compose_graph(
        lambda _ctx: VisionEncoderBlock(
            cfg, ffn_intermediate=4304, ffn_activation="gelu_tanh"
        ),
        (
            Tensor(shape=(seq, d), requires_grad=True),
            Tensor(shape=(1, seq, seq), requires_grad=False),
            Tensor(shape=(seq, head_dim), requires_grad=False),
            Tensor(shape=(seq, head_dim), requires_grad=False),
        ),
    )
    assert "matmul" in _families(graph)
    assert "materialized_causal_mask" not in _families(graph)


def test_sliding_vision_mask_op_in_graph() -> None:
    seq = 64
    mask_graph = compose_graph(
        lambda _ctx: SlidingWindowBidirectionalMask(seq, window_size=8),
        (),
    )
    assert "materialized_sliding_window_bidirectional_mask" in _families(mask_graph)


def test_gemma_preset_qkv_norm_modules() -> None:
    cfg = gemma4_vision_attention()
    assert cfg.value_norm == "rms"
    seq, d, head_dim = 8, 1152, 72

    def _factory(_ctx):
        block = VisionEncoderBlock(
            cfg, ffn_intermediate=4608, norm_kind="rms", ffn_activation="geglu"
        )
        assert hasattr(block.attn, "qk_norm")
        assert hasattr(block.attn, "v_norm")
        return block

    compose_graph(
        _factory,
        (
            Tensor(shape=(seq, d), requires_grad=True),
            Tensor(shape=(1, seq, seq), requires_grad=False),
            Tensor(shape=(seq, head_dim), requires_grad=False),
            Tensor(shape=(seq, head_dim), requires_grad=False),
        ),
    )


def test_bidirectional_mask_module() -> None:
    graph = compose_graph(
        lambda _ctx: MaterializedBidirectionalMask(8),
        (),
    )
    assert "materialized_bidirectional_mask" in _families(graph)


def test_muse_sliding_layer_config() -> None:
    cfg = muse_glimmer_vision_attention(layer_index=0)
    assert cfg.mask == "sliding_1d"
    assert cfg.window_size == 112
