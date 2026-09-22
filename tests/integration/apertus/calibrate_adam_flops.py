"""Measure AdamW ``opt.step()`` FLOPs/param on GOLDEN HF Apertus (CUDA)."""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode

from zepto.empirical.models.apertus import ApertusFamily

from tests.integration.apertus.shared import GOLDEN


def measure_adam_flops_per_param(
    *,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """Return ``(trainable_params, measured_flops / trainable_params)``."""
    if device is None:
        device = torch.device("cuda")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for Adam FLOP calibration")

    family = ApertusFamily()
    seq = GOLDEN["seq_len"]
    model = family.build_hf_model(
        GOLDEN,
        seq_len=seq,
        precision="fp32",
        device=device,
        twin_mode="apertus_parity",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.arange(seq, device=device, dtype=torch.long)
    labels = input_ids.clone()

    opt.zero_grad(set_to_none=False)
    out = family.hf_forward_train(model, input_ids, labels)
    out.loss.backward()
    with FlopCounterMode(display=False) as counter:
        opt.step()
    measured = int(counter.get_total_flops())
    return trainable, measured / trainable


if __name__ == "__main__":
    n, fpp = measure_adam_flops_per_param()
    print(f"trainable={n} flops_per_param={fpp:.4f}")
