"""CUDA calibration of AdamWPolicy FLOPs vs FlopCounterMode on opt.step()."""

from __future__ import annotations

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from zepto.analysis import AdamW
from zepto.empirical.models.apertus import ApertusFamily

from tests.integration.apertus.shared import GOLDEN

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _rel_err(measured: float, reference: float) -> float:
    if reference == 0:
        return float("inf") if measured != 0 else 0.0
    return abs(measured - reference) / reference


def test_adamw_policy_matches_measured_opt_step_flops() -> None:
    family = ApertusFamily()
    device = torch.device("cuda")
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
    flops_per_param = measured / trainable

    from zepto.analysis import reference_invocation

    ctx = reference_invocation()
    policy_flops = AdamW.update_flops(trainable_elements=trainable, context=ctx)

    assert _rel_err(policy_flops, measured) < 0.15, (
        f"Adam policy rel err {_rel_err(policy_flops, measured):.3f} "
        f"(policy={policy_flops}, measured={measured}, "
        f"flops/param={flops_per_param:.4f})"
    )
