"""CUDA FLOP parity for RoPE on GOLDEN Apertus infer (smoke-aligned ctx)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from zepto.analysis import estimate
from zepto.compose import Tensor, compose_graph
from zepto.empirical.models.apertus import ApertusFamily
from zepto.modules import Apertus

from tests.integration.apertus.shared import (
    GOLDEN,
    golden_apertus_hf_flop_ctx,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _rel_err(measured: float, reference: float) -> float:
    if reference == 0:
        return float("inf") if measured != 0 else 0.0
    return abs(measured - reference) / reference


def test_apertus_infer_flops_rope_and_gqa_within_smoke_band() -> None:
    """Full infer forward FLOPs vs Zepto after RoPE regions + sdpa-math GQA."""
    seq = GOLDEN["seq_len"]
    ctx = golden_apertus_hf_flop_ctx()
    graph = compose_graph(
        lambda _ctx: Apertus(**GOLDEN),
        (Tensor(shape=(seq,)),),
    )
    zepto_flops = estimate(graph, ctx).flops.forward_flops

    family = ApertusFamily()
    device = torch.device("cuda")
    model = family.build_hf_model(
        GOLDEN,
        seq_len=seq,
        precision="fp32",
        device=device,
        twin_mode="apertus_parity",
    )
    model.eval()
    input_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            family.hf_forward_infer(model, input_ids.squeeze(0))
    measured = int(counter.get_total_flops())

    assert _rel_err(measured, zepto_flops) < 0.10, (
        f"infer FLOP rel err {_rel_err(measured, zepto_flops):.3f} "
        f"(measured={measured}, zepto={zepto_flops})"
    )
