"""Phase 3 integration: Apertus block prefill+decode horizon with KV state."""

from __future__ import annotations

from zepto.analysis import (
    HorizonSpec,
    KVConfig,
    estimate_horizon,
    reference_invocation,
)
from zepto.compose import Tensor, compose_graph
from zepto.modules.blocks.apertus_decoder_block import ApertusDecoderBlock
from zepto.semantic.metadata import DType

_HIDDEN = 128
_NUM_HEADS = 4
_NUM_KV = 2
_HEAD_DIM = 32
_INTERMEDIATE = 256


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
        qk_norm=True,
    )


def _block_inputs(step, _ctx, _state):
    seq = step.seq_len
    return (
        Tensor(shape=(seq, _HIDDEN)),
        Tensor(shape=(1, seq, seq)),
        Tensor(shape=(seq, _HEAD_DIM)),
        Tensor(shape=(seq, _HEAD_DIM)),
    )


def test_apertus_block_prefill_decode_horizon() -> None:
    ctx = _flash_ctx()
    spec = HorizonSpec.inference(
        prefill=64,
        decode_steps=4,
        kv=KVConfig(
            num_layers=1,
            num_kv_heads=_NUM_KV,
            head_dim=_HEAD_DIM,
            dtype=DType.FP16,
        ),
    )

    report = estimate_horizon(spec, _block_module, _block_inputs, ctx)

    assert len(report.per_step) == 5
    assert report.state_final.kv_caches
    assert report.state_final.kv_caches[0].seq_len == 68
    assert report.total_flops > 0
    assert report.peak_vram > 0

    prefill_flops = report.per_step[0].flops.forward_flops
    decode_flops = report.per_step[1].flops.forward_flops
    assert decode_flops < prefill_flops


def test_gqa_decode_uses_paged_flops_from_kv_state() -> None:
    from zepto.analysis import lower
    from zepto.analysis.horizon.state import KVCacheState
    from zepto.modules.attention.gqa import GroupedQueryAttention

    graph = compose_graph(
        lambda _ctx: GroupedQueryAttention(
            _HIDDEN, _NUM_HEADS, _NUM_KV, head_dim=_HEAD_DIM
        ),
        (
            Tensor(shape=(1, _HIDDEN), requires_grad=False),
            Tensor(shape=(1, 1, 1), requires_grad=False),
        ),
    )
    cache_len = 128
    kv = KVCacheState(
        num_layers=1,
        num_kv_heads=_NUM_KV,
        head_dim=_HEAD_DIM,
        seq_len=cache_len,
        dtype=DType.FP16,
    )
    ctx = reference_invocation(
        requested_capabilities=frozenset({"fused", "flash"}),
        default_dtype=DType.FP16,
        state=(("kv_cache:0", kv),),
    )
    lowered = lower(graph, ctx)
    gqa = next(n for n in lowered.nodes if n.implementation.startswith("region/gqa"))

    expected_paged = 4 * _NUM_HEADS * cache_len * _HEAD_DIM + 5 * _NUM_HEADS * cache_len
    assert gqa.forward_flops == expected_paged
    assert gqa.backward_flops == 0
