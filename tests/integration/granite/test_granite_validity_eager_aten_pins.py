"""Granite validity context pins eager ATen SwiGLU, not Liger."""

from __future__ import annotations

from zepto.analysis import (
    AdamW,
    HorizonSpec,
    estimate_horizon,
    inputs_from_token_ids_and_labels,
)
from zepto.empirical.parity_ctx import EAGER_ATEN_REGION_PINS, build_invocation_context
from zepto.modules import GraniteForCausalLM
from zepto.modules.models.granite import GraniteConfig

_CFG = GraniteConfig(
    hidden_size=256,
    intermediate_size=1024,
    num_q_heads=4,
    num_kv_heads=2,
    head_dim=64,
    num_layers=2,
    vocab_size=1024,
)
_SEQ = 32


def _module(_ctx):
    return GraniteForCausalLM(config=_CFG, seq_len=_SEQ)


def test_validity_ctx_pins_swiglu_decomposed() -> None:
    ctx = build_invocation_context("fp32", cuda_capability=(7, 5))
    assert ctx.region_implementation_pins["region/swiglu"] == (
        "region/swiglu/decomposed"
    )
    assert EAGER_ATEN_REGION_PINS["region/swiglu"] == "region/swiglu/decomposed"


def test_granite_train_graph_uses_decomposed_swiglu_not_liger() -> None:
    ctx = build_invocation_context("fp32", cuda_capability=(7, 5))
    spec = HorizonSpec.training(
        seq_len=_SEQ,
        batch=1,
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
    assert "region/swiglu/decomposed" in impls
    assert "region/rmsnorm/reference" in impls
    assert "region/softmax/reference" in impls
    assert "region/linear_ce/reference" in impls
    assert not any("xielu" in name for name in impls)
    assert not any(name.startswith("region/gqa/") for name in impls)
    assert "region/swiglu/liger" not in impls
    assert "region/swiglu/liger-fused-gate-up" not in impls
    assert not any(
        n.implementation.startswith("sigmoid") for n in sim.timeline[0].lowered.nodes
    )
