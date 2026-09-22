"""Horizon conv + recurrent state port accounting."""

from __future__ import annotations

from zepto.analysis.horizon.spec import ConvStateConfig, HorizonSpec, RecurrentStateConfig
from zepto.analysis.horizon.state import Conv1DState, RecurrentScanState, StatePortRegistry
from zepto.semantic.metadata import DType


def test_conv_and_recurrent_state_after_prefill() -> None:
    spec = HorizonSpec.inference(
        prefill=64,
        decode_steps=4,
        conv=ConvStateConfig(
            layer_indices=(0, 1),
            channels=128,
            kernel_size=4,
            dtype=DType.BF16,
        ),
        recurrent=RecurrentStateConfig(
            layers=(
                (0, "gated_delta", 2, 16, 16),
                (1, "mamba2", 4, 8, 32),
            ),
            dtype=DType.BF16,
        ),
    )
    registry = spec.build_state_registry()
    from zepto.analysis.horizon.spec import HorizonStep, StepKind

    prefill = HorizonStep(kind=StepKind.PREFILL, seq_len=64)
    registry = registry.advance(prefill, lowered=None)  # type: ignore[arg-type]

    snapshot = registry.snapshot()
    conv_entries = [
        v for k, v in snapshot.custom if k.startswith("conv_state:") and isinstance(v, Conv1DState)
    ]
    scan_entries = [
        v
        for k, v in snapshot.custom
        if k.startswith("scan_state:") and isinstance(v, RecurrentScanState)
    ]
    assert len(conv_entries) == 2
    assert len(scan_entries) == 2
    itemsize = DType.BF16.itemsize or 0
    assert conv_entries[0].bytes == 128 * 4 * itemsize
    assert scan_entries[0].bytes == 2 * 16 * 16 * itemsize
    assert scan_entries[1].bytes == 4 * 8 * 32 * itemsize


def test_conv_and_recurrent_state_bytes_scale_with_batch() -> None:
    from zepto.analysis.horizon.spec import HorizonStep, StepKind
    from zepto.analysis.horizon.state import KVCacheState

    spec = HorizonSpec.inference(
        prefill=64,
        decode_steps=1,
        batch=4,
        conv=ConvStateConfig(
            layer_indices=(0,),
            channels=128,
            kernel_size=4,
            dtype=DType.BF16,
        ),
        recurrent=RecurrentStateConfig(
            layers=((0, "mamba2", 4, 8, 32),),
            dtype=DType.BF16,
        ),
        kv=None,
    )
    registry = spec.build_state_registry()
    prefill = HorizonStep(kind=StepKind.PREFILL, seq_len=64, batch=4)
    registry = registry.advance(prefill, lowered=None)  # type: ignore[arg-type]
    snapshot = registry.snapshot()
    conv = next(v for k, v in snapshot.custom if k.startswith("conv_state:"))
    scan = next(v for k, v in snapshot.custom if k.startswith("scan_state:"))
    itemsize = DType.BF16.itemsize or 0
    assert conv.batch == 4
    assert conv.bytes == 4 * 128 * 4 * itemsize
    assert scan.batch == 4
    assert scan.bytes == 4 * 4 * 8 * 32 * itemsize
    kv = KVCacheState(
        num_layers=1, num_kv_heads=2, head_dim=32, seq_len=64, dtype=DType.FP16, batch=4
    )
    assert kv.bytes == 4 * KVCacheState(
        num_layers=1, num_kv_heads=2, head_dim=32, seq_len=64, dtype=DType.FP16
    ).bytes
