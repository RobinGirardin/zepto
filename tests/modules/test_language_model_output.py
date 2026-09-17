"""Compose tests for LanguageModelOutput."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor, compose_graph
from zepto.modules.language_model_output import (
    LanguageModelOutput,
    LanguageModelOutputConfig,
)
from zepto.modules.lm_head import LMHead
from zepto.modules.logit_soft_cap import LogitSoftCapConfig
from zepto.semantic import EmbeddingLookup


def test_untied_matches_lm_head_families_without_cap() -> None:
    d, v, s = 64, 128, 8
    lm_graph = compose_graph(
        lambda _ctx: LMHead(d, v),
        (Tensor(shape=(s, d), requires_grad=True),),
    )
    out_graph = compose_graph(
        lambda _ctx: LanguageModelOutput(
            LanguageModelOutputConfig(hidden_size=d, vocab_size=v)
        ),
        (Tensor(shape=(s, d), requires_grad=True),),
    )
    lm_families = {lm_graph.node(n).operation_family for n in lm_graph.nodes}
    out_families = {out_graph.node(n).operation_family for n in out_graph.nodes}
    assert lm_families == out_families


def test_tied_weight_single_parameter_in_graph() -> None:
    d, v, s = 32, 64, 4

    class TiedProbe(Module):
        module_kind = "TiedProbe"

        def __init__(self, ctx) -> None:
            super().__init__()
            self.shared = ctx.parameter(
                Parameter(shape=(d, v), semantic_type="weight")
            )
            self.head = LanguageModelOutput(
                LanguageModelOutputConfig(
                    hidden_size=d, vocab_size=v, tie_weights=True
                ),
                weight=self.shared,
            )
            self.hidden = ctx.input(Tensor(shape=(s, d), requires_grad=True))
            self.token_ids = ctx.input(
                Tensor(shape=(s,), semantic_type="token_ids", requires_grad=False)
            )

        def forward(self) -> Tensor:
            _ = EmbeddingLookup()(self.token_ids, parameters=(self.shared,))
            return self.head(self.hidden)  # type: ignore[return-value]

    graph = compose_graph(lambda ctx: TiedProbe(ctx), ())
    assert len(graph.parameters) == 1


def test_with_soft_cap_includes_tanh() -> None:
    graph = compose_graph(
        lambda _ctx: LanguageModelOutput(
            LanguageModelOutputConfig(
                hidden_size=32,
                vocab_size=64,
                soft_cap=LogitSoftCapConfig(cap=20.0),
            )
        ),
        (Tensor(shape=(4, 32), requires_grad=True),),
    )
    families = {graph.node(n).operation_family for n in graph.nodes}
    assert "tanh" in families
    assert "linear_matmul" in families


def test_rank2_validation() -> None:
    graph = compose_graph(
        lambda _ctx: LanguageModelOutput(
            LanguageModelOutputConfig(hidden_size=16, vocab_size=32)
        ),
        (Tensor(shape=(4, 16), requires_grad=True),),
    )
    assert graph.node_order
