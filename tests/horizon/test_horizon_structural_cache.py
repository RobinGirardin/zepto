"""Structural cache correctness for horizon simulation."""

from __future__ import annotations

from zepto.analysis import (
    HorizonSpec,
    KVConfig,
    StepKind,
    account_flops,
    estimate_horizon,
    reference_invocation,
    simulate_horizon,
)
from zepto.analysis.flops import account_flops as account_step_flops
from zepto.analysis.horizon.cache import HorizonStructuralCacheStats
from zepto.compose import Tensor
from modules.blocks.apertus_decoder_block import ApertusDecoderBlock
from zepto.semantic.metadata import DType

_HIDDEN = 64
_NUM_HEADS = 4
_NUM_KV = 2
_HEAD_DIM = 16
_INTERMEDIATE = 128


def _flash_ctx(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
        "default_dtype": DType.FP16,
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def _block_module(_compose):
    return ApertusDecoderBlock(
        _HIDDEN,
        _INTERMEDIATE,
        _NUM_HEADS,
        _NUM_KV,
        head_dim=_HEAD_DIM,
        qk_norm=False,
    )


def _block_inputs(step, _ctx, _state):
    seq = step.seq_len
    return (
        Tensor(shape=(seq, _HIDDEN)),
        Tensor(shape=(1, seq, seq)),
        Tensor(shape=(seq, _HEAD_DIM)),
        Tensor(shape=(seq, _HEAD_DIM)),
    )


def test_horizon_structural_cache_matches_uncached_estimates() -> None:
    ctx = _flash_ctx()
    spec = HorizonSpec.inference(
        prefill=32,
        decode_steps=3,
        kv=KVConfig(
            num_layers=1,
            num_kv_heads=_NUM_KV,
            head_dim=_HEAD_DIM,
            dtype=DType.FP16,
        ),
    )

    cached_report = estimate_horizon(spec, _block_module, _block_inputs, ctx)
    uncached_sim = simulate_horizon(
        spec,
        _block_module,
        _block_inputs,
        ctx,
        use_structural_cache=False,
    )
    account_flops(uncached_sim)
    uncached_flops = tuple(
        account_step_flops(record.lowered).forward_flops
        for record in uncached_sim.timeline
    )
    cached_flops = tuple(
        step.flops.forward_flops for step in cached_report.per_step
    )

    assert cached_flops == uncached_flops
    assert cached_report.total_flops == sum(uncached_flops)


def test_decode_flops_grow_with_kv_while_cache_reuses_compose() -> None:
    ctx = _flash_ctx()
    spec = HorizonSpec.inference(
        prefill=16,
        decode_steps=4,
        kv=KVConfig(
            num_layers=1,
            num_kv_heads=_NUM_KV,
            head_dim=_HEAD_DIM,
            dtype=DType.FP16,
        ),
    )
    stats = HorizonStructuralCacheStats()
    sim = simulate_horizon(
        spec,
        _block_module,
        _block_inputs,
        ctx,
        cache_stats=stats,
    )
    account_flops(sim)

    decode_records = [
        record for record in sim.timeline if record.step.kind == StepKind.DECODE
    ]
    assert len(decode_records) >= 2
    flops = [
        account_step_flops(record.lowered).forward_flops for record in decode_records
    ]
    assert flops[1] > flops[0]

    # prefill + one decode structural key
    assert stats.cache_misses == 2
    assert stats.cache_hits == len(spec.steps) - 2
    assert stats.compose_skipped == stats.cache_hits
