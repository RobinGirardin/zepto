"""CUDA audit: Zepto sdpa-math GQA vs HF eager forward FLOPs (GOLDEN)."""

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


@pytest.mark.parametrize("seq_len", [8, 64])
def test_gqa_sdpa_math_infer_flops_match_hf_eager(seq_len: int) -> None:
    opts = {**GOLDEN, "seq_len": seq_len}
    ctx = golden_apertus_hf_flop_ctx()
    graph = compose_graph(
        lambda _ctx: Apertus(**opts),
        (Tensor(shape=(seq_len,)),),
    )
    zepto_flops = estimate(graph, ctx).flops.forward_flops

    family = ApertusFamily()
    device = torch.device("cuda")
    model = family.build_hf_model(
        opts,
        seq_len=seq_len,
        precision="fp32",
        device=device,
        twin_mode="apertus_parity",
    )
    model.eval()
    input_ids = torch.arange(seq_len, device=device, dtype=torch.long)

    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            family.hf_forward_infer(model, input_ids)
    measured = int(counter.get_total_flops())

    # Before WS4 recipe fixes, rel err on S=64 was ~0.12; target <10% post-audit.
    assert _rel_err(measured, zepto_flops) < 0.10, (
        f"S={seq_len}: rel err {_rel_err(measured, zepto_flops):.3f} "
        f"(measured={measured}, zepto={zepto_flops})"
    )
