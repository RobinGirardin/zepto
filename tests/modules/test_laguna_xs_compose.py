"""Laguna XS full-model compose tests."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.blocks.laguna_decoder_block import LagunaDecoderBlock
from zepto.modules.models.laguna_xs import LagunaXs, LagunaXsConfig
from zepto.modules.moe.moe_presets import laguna_sparse_moe_block
from zepto.modules.position.rope_config import laguna_layer_binding
from zepto.semantic.metadata import DType


def _rope_apply(layer_index: int):
    from zepto.modules.models.laguna_xs import _rope_apply_for_binding

    return _rope_apply_for_binding(layer_index)


def test_laguna_layer1_flops_exceed_layer0() -> None:
    seq_len, hidden = 512, 128
    ctx = reference_invocation(default_dtype=DType.FP16)

    def layer0(_c):
        binding = laguna_layer_binding(layer_index=0)
        attn = binding.attention.__class__(
            hidden_size=hidden,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=32,
            mask="full",
        )
        return LagunaDecoderBlock(
            0,
            attention=attn,
            hidden_size=hidden,
            rope=None,
            dense_swiglu_intermediate=256,
        )

    def layer1(_c):
        binding = laguna_layer_binding(layer_index=1)
        attn = binding.attention.__class__(
            hidden_size=hidden,
            num_q_heads=4,
            num_kv_heads=2,
            head_dim=32,
            mask="sliding",
            window_size=16,
            output_gate="softplus",
            position="none",
        )
        return LagunaDecoderBlock(
            1,
            attention=attn,
            hidden_size=hidden,
            rope=None,
            moe_factory=lambda: laguna_sparse_moe_block(
                hidden_size=hidden,
                num_experts=4,
                num_experts_per_tok=2,
                seq_len=seq_len,
            ),
        )

    hidden_in = Tensor(shape=(seq_len, hidden), requires_grad=True)
    mask = Tensor(shape=(1, seq_len, seq_len), requires_grad=False)
    class _Run(Module):
        module_kind = "LagunaRun"

        def __init__(self, block: Module) -> None:
            super().__init__()
            self.block = block

        def forward(self, h: Tensor, m: Tensor) -> Tensor:
            return self.block(h, m, None, None)  # type: ignore[misc,return-value]

    g0 = compose_graph(lambda c: _Run(layer0(c)), (hidden_in, mask))
    g1 = compose_graph(lambda c: _Run(layer1(c)), (hidden_in, mask))
    assert estimate(g1, ctx).flops.total_flops > estimate(g0, ctx).flops.total_flops


def test_laguna_xs_tiny_compose() -> None:
    seq_len = 512
    graph = compose_graph(
        lambda _ctx: LagunaXs(
            config=LagunaXsConfig(
                hidden_size=2048,
                dense_swiglu_intermediate=8192,
                num_layers=2,
                vocab_size=512,
            ),
            seq_len=seq_len,
        ),
        (Tensor(shape=(seq_len,)),),
    )
    kinds = [graph.node(n).provenance.component_type for n in graph.nodes]
    assert "LagunaSparseMoEBlock" in kinds
