"""cuBLAS infer VRAM correction after process-wide handle warmup."""

from __future__ import annotations

from zepto.analysis import cublas_workspace_bytes_per_handle
from zepto.empirical.runner import infer_cublas_vram_correction_bytes

PRE_SM90 = 8_519_680
SM90_PLUS = 33_554_432


def test_cublas_workspace_bytes_per_handle_policy() -> None:
    assert cublas_workspace_bytes_per_handle((7, 5)) == PRE_SM90
    assert cublas_workspace_bytes_per_handle((9, 0)) == SM90_PLUS


def test_correction_zero_before_process_warmup() -> None:
    assert infer_cublas_vram_correction_bytes((7, 5), process_warmed=False) == 0


def test_correction_one_handle_after_warmup() -> None:
    assert infer_cublas_vram_correction_bytes((7, 5), process_warmed=True) == PRE_SM90
    assert infer_cublas_vram_correction_bytes((7, 5)) == PRE_SM90
    assert infer_cublas_vram_correction_bytes((9, 0)) == SM90_PLUS
