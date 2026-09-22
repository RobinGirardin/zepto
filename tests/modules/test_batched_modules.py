"""Compose smoke tests for parallel-batch module ranks (Workstream 5.2 Tier A)."""

from __future__ import annotations

from zepto.compose import Tensor, compose_graph
from zepto.modules.layers.embedding import Embedding
from zepto.modules.layers.lm_head import LMHead
from zepto.modules.layers.xielu import XIELU
from zepto.modules.layers.gated_grouped_rms_norm import GatedGroupedRMSNorm
from zepto.modules.models.apertus import Apertus
from zepto.modules.output.capped_fused_linear_cross_entropy import (
    CappedFusedLinearCrossEntropy,
)
from zepto.modules.output.fused_linear_cross_entropy import FusedLinearCrossEntropy
from zepto.modules.output.language_model_output import (
    LanguageModelOutput,
    LanguageModelOutputConfig,
)
from zepto.modules.output.logit_soft_cap import LogitSoftCapConfig
_B, _S, _D, _V = 2, 8, 32, 64


def _out_shape(graph) -> tuple[int, ...]:
    return graph.edge(graph.outputs[0]).tensor.shape


def test_embedding_batched_compose() -> None:
    graph = compose_graph(
        lambda _ctx: Embedding(_D, _V),
        (Tensor(shape=(_B, _S), semantic_type="token_ids", requires_grad=False),),
    )
    assert _out_shape(graph) == (_B, _S, _D)


def test_lm_head_batched_compose() -> None:
    graph = compose_graph(
        lambda _ctx: LMHead(_D, _V),
        (Tensor(shape=(_B, _S, _D), requires_grad=True),),
    )
    assert _out_shape(graph) == (_B, _S, _V)


def test_language_model_output_batched_compose() -> None:
    graph = compose_graph(
        lambda _ctx: LanguageModelOutput(
            LanguageModelOutputConfig(hidden_size=_D, vocab_size=_V)
        ),
        (Tensor(shape=(_B, _S, _D), requires_grad=True),),
    )
    assert _out_shape(graph) == (_B, _S, _V)


def test_fused_linear_ce_batched_compose() -> None:
    graph = compose_graph(
        lambda _ctx: FusedLinearCrossEntropy(_D, _V),
        (
            Tensor(shape=(_B, _S, _D), requires_grad=True),
            Tensor(shape=(_B, _S), semantic_type="labels", requires_grad=False),
        ),
    )
    assert _out_shape(graph) == (_B * _S,)


def test_capped_fused_linear_ce_batched_compose() -> None:
    graph = compose_graph(
        lambda _ctx: CappedFusedLinearCrossEntropy(
            _D, _V, soft_cap=LogitSoftCapConfig(cap=20.0)
        ),
        (
            Tensor(shape=(_B, _S, _D), requires_grad=True),
            Tensor(shape=(_B, _S), semantic_type="labels", requires_grad=False),
        ),
    )
    assert _out_shape(graph) == (_B * _S,)


def test_xielu_batched_compose() -> None:
    graph = compose_graph(
        lambda _ctx: XIELU(),
        (Tensor(shape=(_B, _S, _D), requires_grad=True),),
    )
    assert _out_shape(graph) == (_B, _S, _D)


def test_gated_grouped_rms_norm_batched_compose() -> None:
    groups = 4
    graph = compose_graph(
        lambda _ctx: GatedGroupedRMSNorm(_D, groups),
        (
            Tensor(shape=(_B, _S, _D), requires_grad=True),
            Tensor(shape=(_B, _S, _D), requires_grad=True),
        ),
    )
    assert _out_shape(graph) == (_B, _S, _D)


def _tiny_apertus(seq_len: int) -> Apertus:
    return Apertus(
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        num_kv_heads=2,
        num_layers=1,
        vocab_size=512,
        seq_len=seq_len,
    )


def test_apertus_batched_compose() -> None:
    graph = compose_graph(
        lambda _ctx: _tiny_apertus(_S),
        (Tensor(shape=(_B, _S), semantic_type="token_ids", requires_grad=False),),
    )
    assert _out_shape(graph) == (_B, _S, 512)

