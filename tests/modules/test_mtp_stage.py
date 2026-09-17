"""Compose tests for MTP stage modules."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_identity_defaults
from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.materialized_causal_mask import MaterializedCausalMask
from zepto.modules.moe_presets import nemotron_moe_block
from zepto.modules.mtp_presets import nemotron_h_mtp_stage, qwen38_mtp_head


def test_qwen_mtp_graph_no_attention() -> None:
    d, v, s = 64, 128, 8
    graph = compose_graph(
        lambda _ctx: qwen38_mtp_head(
            hidden_size=d,
            vocab_size=v,
            mlp_intermediate=256,
        ),
        (
            Tensor(shape=(s, d), requires_grad=True),
            Tensor(shape=(s,), semantic_type="token_ids", requires_grad=False),
        ),
    )
    kinds = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert "MtpInputFusion" in kinds
    assert "MtpMlpBlock" in kinds
    assert "LanguageModelOutput" in kinds
    families = {graph.node(n).operation_family for n in graph.nodes}
    assert "flexible_attention" not in families


def test_nemotron_mtp_hybrid_blocks() -> None:
    d, v, s = 64, 128, 4
    mask_mod = MaterializedCausalMask(s)

    class NemotronMtpProbe(Module):
        module_kind = "NemotronMtpProbe"

        def __init__(self, ctx) -> None:
            super().__init__()
            self.stage = nemotron_h_mtp_stage(
                hidden_size=d,
                vocab_size=v,
                num_nextn_predict_layers=2,
                seq_len=s,
                moe_factory=lambda: nemotron_moe_block(hidden_size=d, seq_len=s),
            )
            self.trunk = ctx.input(Tensor(shape=(s, d), requires_grad=True))
            self.token_ids = ctx.input(
                Tensor(shape=(s,), semantic_type="token_ids", requires_grad=False)
            )
            self.mask = mask_mod()

        def forward(self) -> Tensor:
            return self.stage(self.trunk, self.token_ids, self.mask)  # type: ignore[return-value]

    graph = compose_graph(lambda ctx: NemotronMtpProbe(ctx), ())
    kinds = {graph.node(n).provenance.component_type for n in graph.nodes}
    assert "HybridDecoderBlock" in kinds


def test_mtp_output_shape() -> None:
    d, v, s = 32, 64, 4
    graph = compose_graph(
        lambda _ctx: qwen38_mtp_head(
            hidden_size=d, vocab_size=v, mlp_intermediate=128
        ),
        (
            Tensor(shape=(s, d), requires_grad=True),
            Tensor(shape=(s,), semantic_type="token_ids", requires_grad=False),
        ),
    )
    out_node = graph.node(graph.node_order[-1])
    shape = graph.edge(out_node.output_edges[0]).tensor.shape
    assert shape == (s, v)


def test_qwen_mtp_flop_floor() -> None:
    s, d, i = 8, 64, 256
    graph = compose_graph(
        lambda _ctx: qwen38_mtp_head(
            hidden_size=d, vocab_size=128, mlp_intermediate=i
        ),
        (
            Tensor(shape=(s, d), requires_grad=True),
            Tensor(shape=(s,), semantic_type="token_ids", requires_grad=False),
        ),
    )
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    flops = sum(
        n.forward_flops
        for n in lower(graph, reference_invocation(phase="forward"), registry=registry).nodes
    )
    fusion_floor = 2 * s * d * (2 * d)
    assert flops >= fusion_floor
