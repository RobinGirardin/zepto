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
