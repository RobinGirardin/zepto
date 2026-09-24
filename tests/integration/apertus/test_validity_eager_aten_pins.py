"""Validity context uses eager ATen reference leaves, not Flash/SDPA/Liger."""

from __future__ import annotations

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    estimate_horizon,
    inputs_from_token_ids_and_labels,
)
from zepto.empirical.parity_ctx import EAGER_ATEN_REGION_PINS, build_invocation_context
from zepto.modules import ApertusForCausalLM

_SUBJECT = dict(
    hidden_size=256,
    intermediate_size=1024,
    num_heads=2,
    num_kv_heads=2,
    num_layers=2,
    vocab_size=1024,
    seq_len=64,
    head_dim=128,
)


def _module(_ctx):
    return ApertusForCausalLM(**_SUBJECT)


def test_validity_ctx_pins_reference_aten_leaves() -> None:
    ctx = build_invocation_context("fp32", cuda_capability=(7, 5))
    assert ctx.requested_capabilities == frozenset({"fused"})
    assert ctx.attention_backend == "eager"
    assert dict(ctx.region_implementation_pins) == dict(EAGER_ATEN_REGION_PINS)


def test_validity_train_graph_uses_reference_aten_not_gqa() -> None:
    ctx = build_invocation_context("fp32", cuda_capability=(7, 5))
    spec = HorizonSpec.training(
        seq_len=_SUBJECT["seq_len"],
        batch=4,
        micro_batches=1,
        optimizer=AdamW,
    )
    _report, sim = estimate_horizon(
        spec,
        _module,
        inputs_from_token_ids_and_labels(),
        ctx,
        return_simulation=True,
    )
    impls = {node.implementation for node in sim.timeline[0].lowered.nodes}
    assert "region/rmsnorm/reference" in impls
    assert "region/xielu/reference" in impls
    assert "region/softmax/reference" in impls
    assert "region/linear_ce/reference" in impls
    assert not any(name.startswith("region/gqa/") for name in impls)
    assert not any(name.endswith("/liger") for name in impls)
    assert not any(name.endswith("/cuda") for name in impls)
