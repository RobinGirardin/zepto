"""HF twin helpers (inputs and model build delegation)."""

from __future__ import annotations

import torch

from zepto.empirical.models.base import ModelFamily


def build_hf_inputs(
    family: ModelFamily,
    opts: dict[str, int],
    *,
    batch_size: int,
    seq_len: int,
    draw_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(draw_seed)
    vocab = opts["vocab_size"]
    input_ids = torch.randint(
        0,
        vocab,
        (batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    return input_ids, labels
