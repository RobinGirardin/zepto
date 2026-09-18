"""CUDA PyTorch ground-truth measurements only."""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode

from zepto.empirical.models.base import ModelFamily


def _cuda_sync_peak_vram() -> int:
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def measure_infer_torch(
    family: ModelFamily,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> tuple[int, int]:
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            family.hf_forward_infer(model, input_ids)
    target_flop = int(flop_counter.get_total_flops())

    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        family.hf_forward_infer(model, input_ids)
    torch.cuda.synchronize()
    target_vram = _cuda_sync_peak_vram()
    return target_flop, target_vram


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
) -> tuple[int, int]:
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
            include_optimizer_step=True,
        )
    torch.cuda.synchronize()
    return int(flop_counter.get_total_flops()), _cuda_sync_peak_vram()


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
