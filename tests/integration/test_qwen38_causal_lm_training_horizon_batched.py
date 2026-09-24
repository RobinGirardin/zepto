"""Batched Qwen38ForCausalLM training-horizon estimate (tokens + labels + CE)."""

from __future__ import annotations

import pytest

from zepto.analysis import (
    AdamW,
    HorizonCostReport,
    HorizonSimulation,
    HorizonSpec,
    estimate_horizon,
    inputs_from_token_ids_and_labels,
    reference_invocation,
)
from zepto.modules import Qwen38ForCausalLM
from zepto.modules.attention.attention_config import gated_gqa
from zepto.modules.mixers.layer_spec import LayerSpec
from zepto.modules.mixers.mixer_config import GatedDeltaNetConfig
from zepto.modules.models.qwen38 import Qwen38Config
from zepto.modules.position.rope_config import qwen3_vl_mrope
from zepto.semantic.metadata import DType

_SEQ_LEN = 8
_BATCH = 2


def _tiny_cycle_specs() -> tuple[LayerSpec, ...]:
    hidden_size = 128
    delta = GatedDeltaNetConfig(
        hidden_size=hidden_size,
        num_qk_heads=4,
        num_v_heads=12,
        head_dim=32,
    )
    attn = gated_gqa(hidden_size, 12, 2, head_dim=64, gate="sigmoid")
    delta_layer = LayerSpec(
        mixer="gated_delta_net",
        gated_delta=delta,
        ffn="swiglu",
        swiglu_intermediate=256,
    )
    attn_layer = LayerSpec(
        mixer="attention",
        attention=attn,
        ffn="swiglu",
        swiglu_intermediate=256,
    )
    return (delta_layer, delta_layer, delta_layer, attn_layer)


def _tiny_clm(_compose):
    return Qwen38ForCausalLM(
        config=Qwen38Config(hidden_size=128, num_layers=4, vocab_size=1024),
        seq_len=_SEQ_LEN,
        layer_specs=_tiny_cycle_specs(),
        rope_config=qwen3_vl_mrope(head_dim=64, rotary_dim=64),
    )


def _ctx():
    return reference_invocation(
        requested_capabilities=frozenset({"fused", "sdpa", "gqa"}),
        attention_backend="eager",
        default_dtype=DType.FP32,
    )


def _training_horizon(
    batch: int,
) -> tuple[HorizonCostReport, HorizonSimulation]:
    spec = HorizonSpec.training(
        seq_len=_SEQ_LEN,
        batch=batch,
        micro_batches=1,
        optimizer=AdamW,
    )
    return estimate_horizon(
        spec,
        _tiny_clm,
        inputs_from_token_ids_and_labels(),
        _ctx(),
        return_simulation=True,
    )


def _ce_node_ids(sim: HorizonSimulation) -> set:
    graph = sim.timeline[0].graph
    return {
        nid
        for nid in graph.node_order
        if graph.node(nid).provenance.component_type == "FusedLinearCrossEntropy"
    }


def _ce_forward_flops(sim: HorizonSimulation) -> int:
    ce_ids = _ce_node_ids(sim)
    return sum(
        n.forward_flops
        for n in sim.timeline[0].lowered.nodes
        if n.node_id in ce_ids
    )


def test_causal_lm_train_horizon_batch_scales_activations() -> None:
    report_1, sim_1 = _training_horizon(1)
    report_b, sim_b = _training_horizon(_BATCH)

    assert len(report_1.per_step) == 2
    assert len(report_b.per_step) == 2
    assert report_1.state_final.grad_accum is None
    assert report_b.state_final.grad_accum is None
    assert report_1.state_final.optimizer is not None
    assert report_b.state_final.optimizer is not None
    assert report_1.state_final.optimizer.bytes == report_b.state_final.optimizer.bytes

    params_1 = report_1.per_step[0].memory.breakdown.parameters
    params_b = report_b.per_step[0].memory.breakdown.parameters
    assert params_1 == params_b

    act_1 = report_1.per_step[0].memory.breakdown.activations
    act_b = report_b.per_step[0].memory.breakdown.activations
    assert act_1 > 0
    assert act_b / act_1 == pytest.approx(_BATCH, rel=0.05)
    assert report_b.peak_vram > report_1.peak_vram

    assert _ce_node_ids(sim_1)
    assert _ce_node_ids(sim_b)
    ce_flops_1 = _ce_forward_flops(sim_1)
    ce_flops_b = _ce_forward_flops(sim_b)
    assert ce_flops_1 > 0
    assert ce_flops_b / ce_flops_1 == pytest.approx(_BATCH, rel=0.05)

    kinds = [
        sim_1.timeline[0].graph.node(n).provenance.component_type
        for n in sim_1.timeline[0].graph.nodes
    ]
    assert "GatedDeltaNet" in kinds
    assert "FusedLinearCrossEntropy" in kinds
