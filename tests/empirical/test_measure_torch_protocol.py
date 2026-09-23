"""Protocol locks for the HuggingFace / PyTorch measurement windows."""

from __future__ import annotations

import inspect

from zepto.empirical.measure_torch import measure_train_step_torch, warmup_cublas_process


def test_train_vram_window_precedes_flop_counter() -> None:
    source = inspect.getsource(measure_train_step_torch)
    assert source.index("reset_peak_memory_stats") < source.index("FlopCounterMode")


def test_train_scored_windows_wrap_optimizer_step() -> None:
    source = inspect.getsource(measure_train_step_torch)
    assert source.count("include_optimizer_step=True") >= 2


def test_warmup_cublas_process_is_defined() -> None:
    assert callable(warmup_cublas_process)
