"""Apertus fused-graph invariant: FCM mask sees GEMMs, not region extras."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis import (
    FCM_OPERATION_FAMILIES,
    account_fcm_flops,
    estimate,
)
from zepto.analysis.lowering.helpers import build_estimation_context
from zepto.compose import Tensor, compose_graph
from zepto.modules import Apertus, ApertusForCausalLM

from tests.integration.apertus.shared import GOLDEN, golden_apertus_hf_flop_ctx

_S = GOLDEN["seq_len"]

_CE_EXTRAS = frozenset(
    {"exp", "divide", "log", "gather", "reduce_sum", "multiply", "reshape"}
)


def _fused_hf_flop_ctx():
    """HF GOLDEN flop ctx with SDPA backend so ``region/gqa/sdpa-math`` wins.

    ``golden_apertus_hf_flop_ctx`` keeps ``attention_backend="eager"`` for the
    HF twin; GQA SDPA variants require ``attention_backend="sdpa"``.
    """
    return replace(golden_apertus_hf_flop_ctx(), attention_backend="sdpa")


def _compose_infer():
    return compose_graph(
        lambda _ctx: Apertus(**GOLDEN),
        (Tensor(shape=(_S,)),),
    )


def _compose_train():
    return compose_graph(
        lambda _ctx: ApertusForCausalLM(**GOLDEN),
        (
            Tensor(shape=(_S,)),
            Tensor(shape=(_S,), semantic_type="labels", requires_grad=False),
        ),
    )


def _region_nodes(lowered, prefix: str):
    return [node for node in lowered.nodes if node.implementation.startswith(prefix)]


def _fcm_families(graph):
    return frozenset(
        node.operation_family
        for node in graph.nodes.values()
        if node.declaration is not None
        and node.operation_family in FCM_OPERATION_FAMILIES
    )


def _component_family_forward(
    graph, context, *, family: str, component_type: str
) -> int:
    total = 0
    for node in graph.nodes.values():
        if node.operation_family != family or node.declaration is None:
            continue
        if node.provenance.component_type != component_type:
            continue
        estimation = build_estimation_context(node, graph, context)
        total += node.declaration.forward_flops(estimation)
    return total


def test_apertus_infer_fcm_mask_sees_gqa_gemms() -> None:
    graph = _compose_infer()
    ctx = _fused_hf_flop_ctx()
    report, lowered = estimate(graph, ctx, return_lowered=True)

    assert _region_nodes(lowered, "region/gqa")
    assert report.flops.zepto_forward_flops == sum(
        node.forward_flops for node in lowered.nodes
    )
    assert 0 < report.flops.fcm_forward_flops < report.flops.zepto_forward_flops
    assert report.flops.fcm_forward_flops == account_fcm_flops(graph, ctx)[0]

    contributing = _fcm_families(graph)
    assert contributing <= FCM_OPERATION_FAMILIES
    assert contributing <= {"matmul", "linear_matmul"}
    assert "matmul" in contributing
    assert "linear_matmul" in contributing

    extras = {
        node.operation_family for node in graph.nodes.values()
    } - FCM_OPERATION_FAMILIES
    assert extras


def test_apertus_train_fcm_mask_keeps_lm_head_drops_ce() -> None:
    graph = _compose_train()
    ctx = _fused_hf_flop_ctx()
    report, lowered = estimate(graph, ctx, return_lowered=True)

    assert _region_nodes(lowered, "region/gqa")
    assert report.flops.zepto_forward_flops == sum(
        node.forward_flops for node in lowered.nodes
    )
    assert 0 < report.flops.fcm_forward_flops < report.flops.zepto_forward_flops
    assert report.flops.fcm_forward_flops == account_fcm_flops(graph, ctx)[0]

    contributing = _fcm_families(graph)
    assert contributing <= FCM_OPERATION_FAMILIES
    assert "linear_matmul" in contributing

    ce_families = {
        node.operation_family
        for node in graph.nodes.values()
        if node.provenance.component_type == "FusedLinearCrossEntropy"
    }
    assert "linear_matmul" in ce_families
    assert ce_families & _CE_EXTRAS
    assert not ((ce_families & _CE_EXTRAS) & contributing)

    lm_head = _component_family_forward(
        graph,
        ctx,
        family="linear_matmul",
        component_type="FusedLinearCrossEntropy",
    )
    assert lm_head > 0
    assert report.flops.fcm_forward_flops >= lm_head
