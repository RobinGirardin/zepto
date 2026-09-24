"""Apertus 1.5 training wrapper: pruned fused-CE vocab and horizon smoke."""

from __future__ import annotations

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    account_flops,
    discover_regions,
    estimate_horizon,
    inputs_from_token_ids_and_labels,
    lower,
)
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.linear_ce import DEFAULT_LINEAR_CE_RECIPE
from zepto.compose import Tensor, compose_graph
from zepto.modules.apertus15_for_causal_lm import Apertus15ForCausalLM
from zepto.modules.models.apertus15 import Apertus15

from tests.integration.apertus.shared import golden_apertus_ctx, merge_train_lowered

_TINY = dict(
    hidden_size=32,
    intermediate_size=64,
    num_heads=8,
    num_kv_heads=2,
    num_layers=2,
    vocab_size=200,
    output_vocab_size=100,
    seq_len=8,
    head_dim=4,
)


def _tiny_train(_ctx):
    return Apertus15ForCausalLM(**_TINY)


def _train_inputs():
    return (
        Tensor(shape=(8,)),
        Tensor(shape=(8,), semantic_type="labels", requires_grad=False),
    )


def test_apertus15_ce_shares_pruned_head_weight() -> None:
    captured: dict = {}

    def factory(_ctx):
        model = Apertus15ForCausalLM(**_TINY)
        captured["ce_vocab"] = model.loss_head.vocab_size
        captured["shared"] = model.loss_head.weight is model.backbone.lm_head.weight
        captured["embed_shape"] = model.backbone.embedding.weight.shape
        return model

    compose_graph(factory, _train_inputs())
    assert captured["ce_vocab"] == 100
    assert captured["shared"] is True
    assert captured["embed_shape"] == (32, 200)


def test_apertus15_discovers_one_linear_ce_region() -> None:
    graph = compose_graph(_tiny_train, _train_inputs())
    ctx = golden_apertus_ctx()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, ctx, registry)
    linear_ce = [r for r in regions if r.kind == "region/linear_ce"]
    assert len(linear_ce) == 1
    assert linear_ce[0].anchor.component_type == "FusedLinearCrossEntropy"
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert any(step.kind == "region" for step in plan.steps)


def test_apertus15_train_ce_flops_use_output_vocab() -> None:
    ctx = golden_apertus_ctx()
    train_graph = compose_graph(_tiny_train, _train_inputs())
    infer_graph = compose_graph(
        lambda _ctx: Apertus15(**_TINY),
        (Tensor(shape=(8,)),),
    )

    train_flops = account_flops(merge_train_lowered(train_graph, ctx)).total_flops
    backbone_infer_flops = account_flops(
        lower(infer_graph, golden_apertus_ctx(phase="forward"))
    ).total_flops
    ce_total = (
        DEFAULT_LINEAR_CE_RECIPE.forward_flops(8, 32, 100)
        + DEFAULT_LINEAR_CE_RECIPE.backward_flops(8, 32, 100, requires_grad=True)
    )
    assert train_flops >= backbone_infer_flops + ce_total

    registry = LoweringRegistry()
    register_defaults(registry)
    ce_regions = [
        r
        for r in discover_regions(train_graph, ctx, registry)
        if r.kind == "region/linear_ce"
    ]
    assert len(ce_regions) == 1
    assert ce_regions[0].anchor.component_type == "FusedLinearCrossEntropy"


def test_apertus15_train_horizon_smoke() -> None:
    ctx = golden_apertus_ctx()
    spec = HorizonSpec.training(
        seq_len=8, batch=1, micro_batches=1, optimizer=AdamW
    )
    report = estimate_horizon(
        spec, _tiny_train, inputs_from_token_ids_and_labels(), ctx
    )
    assert report.flops.total_flops > 0
    assert report.memory.breakdown.parameters > 0
