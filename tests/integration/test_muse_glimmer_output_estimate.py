"""Muse Glimmer output tail prefill FLOP estimate."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.output_presets import granite_lm_output, muse_glimmer_lm_output
from zepto.modules.rms_norm import RMSNorm
from zepto.semantic.metadata import DType


class _OutputTail(Module):
    module_kind = "OutputTail"

    def __init__(self, head_factory) -> None:
        super().__init__()
        self.norm = RMSNorm(128)
        self.head = head_factory()

    def forward(self, hidden):
        return self.head(self.norm(hidden))


def test_muse_output_flops_exceed_matmul_only_baseline() -> None:
    ctx = reference_invocation(default_dtype=DType.FP16)
    baseline = compose_graph(
        lambda _ctx: _OutputTail(
            lambda: granite_lm_output(hidden_size=128, vocab_size=1024)
        ),
        (Tensor(shape=(64, 128), requires_grad=True),),
    )
    muse = compose_graph(
        lambda _ctx: _OutputTail(
            lambda: muse_glimmer_lm_output(hidden_size=128, vocab_size=1024)
        ),
        (Tensor(shape=(64, 128), requires_grad=True),),
    )
    base_flops = estimate(baseline, ctx).flops.total_flops
    muse_flops = estimate(muse, ctx).flops.total_flops
    assert muse_flops > base_flops
