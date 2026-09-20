"""CUDA PyTorch ground-truth measurements only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode

from zepto.empirical.models.base import ModelFamily


@dataclass(frozen=True, slots=True)
class TorchMeasureResult:
    target_flop: int
    target_vram_raw: int
    peak_minus_before: int
    alloc_before: int
    target_flop_no_opt: int = 0


def _cuda_sync_peak_vram() -> int:
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def measure_infer_torch(
    family: ModelFamily,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> TorchMeasureResult:
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            family.hf_forward_infer(model, input_ids)
    target_flop = int(flop_counter.get_total_flops())

    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    alloc_before = int(torch.cuda.memory_allocated())
    with torch.no_grad():
        family.hf_forward_infer(model, input_ids)
    torch.cuda.synchronize()
    target_vram_raw = _cuda_sync_peak_vram()
    return TorchMeasureResult(
        target_flop=target_flop,
        target_vram_raw=target_vram_raw,
        peak_minus_before=target_vram_raw - alloc_before,
        alloc_before=alloc_before,
        target_flop_no_opt=target_flop,
    )


def train_step_forward_backward(
    family: ModelFamily,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    opt: torch.optim.Optimizer,
    *,
    include_optimizer_step: bool,
) -> None:
    out = family.hf_forward_train(model, input_ids, labels)
    out.loss.backward()
    if include_optimizer_step:
        opt.step()


def measure_train_step_torch(
    family: ModelFamily,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    opt: torch.optim.Optimizer,
) -> TorchMeasureResult:
    opt.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with FlopCounterMode(display=False) as flop_counter:
        train_step_forward_backward(
            family,
            model,
            input_ids,
            labels,
            opt,
            include_optimizer_step=False,
        )
    torch.cuda.synchronize()
    target_flop_no_opt = int(flop_counter.get_total_flops())

    opt.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    alloc_before = int(torch.cuda.memory_allocated())
    with FlopCounterMode(display=False) as flop_counter:
        train_step_forward_backward(
            family,
            model,
            input_ids,
            labels,
            opt,
            include_optimizer_step=True,
        )
    torch.cuda.synchronize()
    target_vram_raw = _cuda_sync_peak_vram()
    return TorchMeasureResult(
        target_flop=int(flop_counter.get_total_flops()),
        target_vram_raw=target_vram_raw,
        peak_minus_before=target_vram_raw - alloc_before,
        alloc_before=alloc_before,
        target_flop_no_opt=target_flop_no_opt,
    )


def training_warmup_step(
    family: ModelFamily,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    opt: torch.optim.Optimizer,
) -> None:
    """One unscored train step before scored steps 1..K (smoke notebook protocol)."""
    opt.zero_grad(set_to_none=True)
    train_step_forward_backward(
        family,
        model,
        input_ids,
        labels,
        opt,
        include_optimizer_step=True,
    )
    torch.cuda.synchronize()
