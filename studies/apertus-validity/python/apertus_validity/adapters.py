"""Thin imports of the Zepto and twin measurement windows.

These functions are not the study. They are the locked probes used by
collect.py. Study rules (subject unit, K=1, scored_window, ledger) live
outside this module.

Borrowed from `src/zepto/empirical/`:
- Zepto estimate / estimate_horizon wiring
- twin FLOP then VRAM windows
- process-wide cuBLAS warmup
- one-handle inference subtract
- Apertus scored_window twin (max(S, 32))
"""

from __future__ import annotations

from apertus_validity.protocol import TRAINING_STEPS, TWIN_MODE
from apertus_validity.schema import EvaluationRow, LedgerRow, PrecisionOutcome, Subject


def warmup_process(device) -> None:
    from zepto.empirical.measure_torch import warmup_cublas_process

    warmup_cublas_process(device)


def measure_subject_precision(
    subject: Subject,
    precision: str,
    *,
    device,
    cuda_capability: tuple[int, int],
    input_seed: int,
    process_warmed: bool = True,
) -> PrecisionOutcome:
    """One precision on one subject: Zepto, infer, then K=1 training.

    Framework §7.5. Inference VRAM is raw minus one handle. Training is
    unadjusted. Failures after a successful window keep the completed row.
    """
    from zepto.empirical.measure_torch import (
        measure_infer_torch,
        measure_train_step_torch,
        training_warmup_step,
    )
    from zepto.empirical.measure_zepto import measure_infer_zepto, measure_train_zepto
    from zepto.empirical.models import get_model_family
    from zepto.empirical.parity_ctx import build_invocation_context
    from zepto.empirical.runner import infer_cublas_vram_correction_bytes
    from zepto.empirical.twin_hf import build_hf_inputs

    import torch

    family = get_model_family("apertus")
    opts = subject.architecture.as_options()
    rows: list[EvaluationRow] = []
    ledger: list[LedgerRow] = []
    model = None
    input_ids = None
    labels = None
    opt = None

    def fail(phase: str, exc: BaseException) -> None:
        ledger.append(
            LedgerRow(
                subject_id=subject.subject_id,
                configuration_id=subject.configuration_id,
                phase=phase,
                precision=precision,
                seq_len=subject.seq_len,
                batch_size=subject.batch_size,
                failure_kind=_failure_kind(exc),
                message=_short_message(exc),
            )
        )

    try:
        ctx = build_invocation_context(precision, cuda_capability=cuda_capability)
        zepto_infer = measure_infer_zepto(
            family,
            opts,
            seq_len=subject.seq_len,
            batch_size=subject.batch_size,
            ctx=ctx,
        )
        zepto_train = measure_train_zepto(
            family,
            opts,
            seq_len=subject.seq_len,
            batch_size=subject.batch_size,
            ctx=ctx,
        )
        model = family.build_hf_model(
            opts,
            seq_len=subject.seq_len,
            precision=precision,
            device=device,
            twin_mode=TWIN_MODE,
        )
        input_ids, labels = build_hf_inputs(
            family,
            opts,
            batch_size=subject.batch_size,
            seq_len=subject.seq_len,
            draw_seed=input_seed,
            device=device,
        )
    except Exception as exc:
        fail("setup", exc)
        _release_cuda(model, input_ids, labels, opt)
        return PrecisionOutcome(rows=(), ledger=tuple(ledger))

    try:
        torch_infer = measure_infer_torch(family, model, input_ids)
        correction = infer_cublas_vram_correction_bytes(
            cuda_capability, process_warmed=process_warmed
        )
        rows.append(
            EvaluationRow(
                subject_id=subject.subject_id,
                configuration_id=subject.configuration_id,
                phase="inference",
                precision=precision,
                seq_len=subject.seq_len,
                batch_size=subject.batch_size,
                step=0,
                y_vram=zepto_infer.y_vram,
                target_vram=torch_infer.target_vram_raw - correction,
                target_vram_raw=torch_infer.target_vram_raw,
                cublas_infer_correction_bytes=correction,
                y_flop_fcm=zepto_infer.y_flop_fcm,
                target_flop=torch_infer.target_flop,
                y_flop=zepto_infer.y_flop,
                y_runtime_workspace=zepto_infer.y_runtime_workspace,
                y_activations=zepto_infer.y_activations,
                peak_minus_before=torch_infer.peak_minus_before,
                alloc_before=torch_infer.alloc_before,
            )
        )
    except Exception as exc:
        fail("inference", exc)
        _release_cuda(model, input_ids, labels, opt)
        return PrecisionOutcome(rows=tuple(rows), ledger=tuple(ledger))

    try:
        if TRAINING_STEPS != 1:
            raise RuntimeError("study protocol locks K = 1")
        model.train()
        opt = torch.optim.AdamW(model.parameters())
        training_warmup_step(family, model, input_ids, labels, opt)
        torch_train = measure_train_step_torch(family, model, input_ids, labels, opt)
        rows.append(
            EvaluationRow(
                subject_id=subject.subject_id,
                configuration_id=subject.configuration_id,
                phase="training",
                precision=precision,
                seq_len=subject.seq_len,
                batch_size=subject.batch_size,
                step=1,
                y_vram=zepto_train.y_vram,
                target_vram=torch_train.target_vram_raw,
                target_vram_raw=torch_train.target_vram_raw,
                cublas_infer_correction_bytes=0,
                y_flop_fcm=zepto_train.y_flop_fcm,
                target_flop=torch_train.target_flop,
                y_flop=zepto_train.y_flop,
                y_runtime_workspace=zepto_train.y_runtime_workspace,
                y_activations=zepto_train.y_activations,
                peak_minus_before=torch_train.peak_minus_before,
                alloc_before=torch_train.alloc_before,
            )
        )
    except Exception as exc:
        fail("training", exc)

    _release_cuda(model, input_ids, labels, opt)
    return PrecisionOutcome(rows=tuple(rows), ledger=tuple(ledger))


def _failure_kind(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "outofmemory" in name or "out of memory" in text or isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "build"
    return "measure"


def _short_message(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text.replace("\n", " ")[:500]


def _release_cuda(model, input_ids, labels, opt) -> None:
    del model, input_ids, labels, opt
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
