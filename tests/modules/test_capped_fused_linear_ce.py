"""Compose tests for CappedFusedLinearCrossEntropy."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_identity_defaults
from zepto.compose import Tensor, compose_graph
from modules.output.capped_fused_linear_cross_entropy import (
    CappedFusedLinearCrossEntropy,
)
from modules.output.fused_linear_cross_entropy import FusedLinearCrossEntropy
from modules.output.logit_soft_cap import LogitSoftCapConfig

_S, _D, _V = 4, 32, 64


def test_capped_flce_tanh_before_exp() -> None:
    graph = compose_graph(
        lambda _ctx: CappedFusedLinearCrossEntropy(
            _D, _V, soft_cap=LogitSoftCapConfig(cap=20.0)
        ),
        (
            Tensor(shape=(_S, _D), requires_grad=True),
            Tensor(shape=(_S,), semantic_type="labels", requires_grad=False),
        ),
    )
    families = [graph.node(n).operation_family for n in graph.node_order]
    tanh_index = families.index("tanh")
    exp_index = families.index("exp")
    assert tanh_index < exp_index


def test_capped_flce_flops_ge_uncapped() -> None:
    inputs = (
        Tensor(shape=(_S, _D), requires_grad=True),
        Tensor(shape=(_S,), semantic_type="labels", requires_grad=False),
    )
    uncapped_graph = compose_graph(
        lambda _ctx: FusedLinearCrossEntropy(_D, _V),
        inputs,
    )
    capped_graph = compose_graph(
        lambda _ctx: CappedFusedLinearCrossEntropy(
            _D, _V, soft_cap=LogitSoftCapConfig(cap=20.0)
        ),
        inputs,
    )
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    ctx = reference_invocation(phase="forward")
    uncapped = sum(
        n.forward_flops
        for n in lower(uncapped_graph, ctx, registry=registry).nodes
    )
    capped = sum(
        n.forward_flops
        for n in lower(capped_graph, ctx, registry=registry).nodes
    )
    assert capped >= uncapped
