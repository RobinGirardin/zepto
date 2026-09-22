"""Golden integration tests for Apertus training (ApertusForCausalLM)."""

from __future__ import annotations

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    account_flops,
    discover_regions,
    estimate,
    estimate_horizon,
    lower,
)
from zepto.analysis.lowering import LoweringRegistry, build_lowering_plan
from zepto.analysis.lowering.implementations import register_defaults
from zepto.analysis.lowering.recipes.linear_ce import DEFAULT_LINEAR_CE_RECIPE
from zepto.analysis.horizon.state import trainable_parameter_elements
from zepto.compose import Tensor, compose_graph
from zepto.modules import Apertus, ApertusForCausalLM

import pytest

from tests.integration.apertus.shared import (
    GOLDEN,
    golden_apertus_ctx,
    golden_apertus_hf_flop_ctx,
    merge_train_lowered,
)

_S = GOLDEN["seq_len"]
_D = GOLDEN["hidden_size"]
_V = GOLDEN["vocab_size"]


def _training_module(_ctx):
    return ApertusForCausalLM(**GOLDEN)


def _training_inputs(step, _ctx, _state):
    seq = step.seq_len
    return (
        Tensor(shape=(seq,)),
        Tensor(shape=(seq,), semantic_type="labels", requires_grad=False),
    )


def _compose_training():
    return compose_graph(
        _training_module,
        (
            Tensor(shape=(_S,)),
            Tensor(shape=(_S,), semantic_type="labels", requires_grad=False),
        ),
    )


def _compose_inference():
    return compose_graph(
        lambda _ctx: Apertus(**GOLDEN),
        (Tensor(shape=(_S,)),),
    )


def test_apertus_golden_graph_uses_llama3_rope() -> None:
    captured: dict = {}

    def factory(_ctx):
        model = ApertusForCausalLM(**GOLDEN)
        captured["rope"] = model.backbone.rope_materialize.config
        return model

    compose_graph(
        factory,
        (
            Tensor(shape=(_S,)),
            Tensor(shape=(_S,), semantic_type="labels", requires_grad=False),
        ),
    )
    rope = captured["rope"]
    assert rope.rope_type == "llama3"
    assert rope.rope_theta == 12_000_000.0


def test_apertus_for_causal_lm_composes_and_lowers() -> None:
    graph = _compose_training()
    ctx = golden_apertus_ctx()
    registry = LoweringRegistry()
    register_defaults(registry)
    regions = discover_regions(graph, ctx, registry)
    linear_ce = [r for r in regions if r.kind == "region/linear_ce"]
    assert len(linear_ce) == 1
    plan = build_lowering_plan(graph, regions, ctx, registry)
    assert any(step.kind == "region" for step in plan.steps)


def test_apertus_train_flops_match_linear_ce_recipe() -> None:
    ctx = golden_apertus_ctx()
    train_graph = _compose_training()
    infer_graph = _compose_inference()

    train_lowered = merge_train_lowered(train_graph, ctx)
    infer_lowered = lower(infer_graph, golden_apertus_ctx(phase="forward"))

    train_flops = account_flops(train_lowered).total_flops
    backbone_flops = account_flops(infer_lowered).total_flops
    ce_total = (
        DEFAULT_LINEAR_CE_RECIPE.forward_flops(_S, _D, _V)
        + DEFAULT_LINEAR_CE_RECIPE.backward_flops(_S, _D, _V, requires_grad=True)
    )
    assert train_flops >= backbone_flops + ce_total

    registry = LoweringRegistry()
    register_defaults(registry)
    ce_regions = [
        r
        for r in discover_regions(train_graph, ctx, registry)
        if r.kind == "region/linear_ce"
    ]
    assert len(ce_regions) == 1
    assert ce_regions[0].anchor.component_type == "FusedLinearCrossEntropy"
    assert len(ce_regions[0].operation_ids) == 8


def test_apertus_train_horizon_produces_positive_peak_with_optimizer() -> None:
    ctx = golden_apertus_ctx()
    spec = HorizonSpec.training(seq_len=_S, micro_batches=1, optimizer=AdamW)

    report = estimate_horizon(spec, _training_module, _training_inputs, ctx)

    assert len(report.per_step) == 2
    assert report.total_flops > 0
    assert report.total_flops < 1_230_000
    assert report.peak_vram > 0
    assert report.state_final.optimizer is not None
    assert report.state_final.optimizer.bytes > 0

    train_graph = _compose_training()
    train_lowered = lower(train_graph, golden_apertus_ctx(phase="forward"))
    trainable = trainable_parameter_elements(train_lowered)
    expected_opt = 2 * trainable * (ctx.optim_prec or 4)
    assert abs(report.state_final.optimizer.bytes - expected_opt) <= expected_opt * 0.01


def test_apertus_train_includes_cublas_runtime_workspace() -> None:
    ctx = golden_apertus_ctx()
    spec = HorizonSpec.training(seq_len=_S, micro_batches=1, optimizer=AdamW)
    report = estimate_horizon(spec, _training_module, _training_inputs, ctx)
    assert len(report.per_step) == 2
    assert report.memory.breakdown.runtime_workspace > 0
    max_step_runtime = max(
        step.memory.breakdown.runtime_workspace for step in report.per_step
    )
    assert report.peak_vram >= max_step_runtime
    assert report.per_step[1].memory.breakdown.runtime_workspace == 2 * 8_519_680


def test_apertus_inference_prefill_unchanged() -> None:
    graph = _compose_inference()
    ctx = golden_apertus_ctx()
    report = estimate(graph, ctx)

    assert report.flops.total_flops > 0
    assert report.memory.peak_live_bytes > 0
    assert report.memory.breakdown.parameters > 0

    _, lowered = estimate(graph, ctx, return_lowered=True)
    gqa_regions = [
        n for n in lowered.nodes if n.implementation.startswith("region/gqa")
    ]
    assert len(gqa_regions) == GOLDEN["num_layers"]


def _rel_err(measured: float, reference: float) -> float:
    if reference == 0:
        return float("inf") if measured != 0 else 0.0
    return abs(measured - reference) / reference


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="CUDA required",
)
def test_apertus_train_flop_no_opt_cuda_parity() -> None:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from zepto.empirical.measure_torch import measure_train_step_torch
    from zepto.empirical.models.apertus import ApertusFamily

    ctx = golden_apertus_hf_flop_ctx()
    spec_no_opt = HorizonSpec.training(
        seq_len=_S, micro_batches=1, batch=1, optimizer=None
    )
    zepto_no_opt = int(
        estimate_horizon(
            spec_no_opt, _training_module, _training_inputs, ctx
        ).total_flops
    )

    family = ApertusFamily()
    device = torch.device("cuda")
    model = family.build_hf_model(
        GOLDEN,
        seq_len=_S,
        precision="fp32",
        device=device,
        twin_mode="apertus_parity",
    )
    input_ids = torch.arange(_S, device=device, dtype=torch.long)
    labels = input_ids.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    measured = measure_train_step_torch(
        family, model, input_ids, labels, opt
    ).target_flop_no_opt

    assert _rel_err(measured, zepto_no_opt) < 0.10


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="CUDA required",
)
def test_apertus_train_full_flop_cuda_parity() -> None:
    from zepto.empirical.measure_torch import measure_train_step_torch
    from zepto.empirical.models.apertus import ApertusFamily

    import torch

    ctx = golden_apertus_hf_flop_ctx()
    spec = HorizonSpec.training(
        seq_len=_S, micro_batches=1, batch=1, optimizer=AdamW
    )
    zepto_full = int(
        estimate_horizon(spec, _training_module, _training_inputs, ctx).total_flops
    )

    family = ApertusFamily()
    device = torch.device("cuda")
    model = family.build_hf_model(
        GOLDEN,
        seq_len=_S,
        precision="fp32",
        device=device,
        twin_mode="apertus_parity",
    )
    input_ids = torch.arange(_S, device=device, dtype=torch.long)
    labels = input_ids.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    measured = measure_train_step_torch(
        family, model, input_ids, labels, opt
    ).target_flop

    assert _rel_err(measured, zepto_full) < 0.10
