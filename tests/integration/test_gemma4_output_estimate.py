"""Gemma 4 tied + capped output tail FLOP estimate."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Module, Parameter, Tensor, compose_graph
from zepto.modules.output_presets import gemma4_lm_output, granite_lm_output
from zepto.modules.rms_norm import RMSNorm
from zepto.semantic.metadata import DType


class _GemmaTail(Module):
    module_kind = "GemmaTail"

    def __init__(self, ctx, *, d: int, v: int) -> None:
        super().__init__()
        weight = ctx.parameter(Parameter(shape=(d, v), semantic_type="weight"))
        self.norm = RMSNorm(d)
        self.head = gemma4_lm_output(
            tied_weight=weight, hidden_size=d, vocab_size=v
        )

    def forward(self, hidden):
        return self.head(self.norm(hidden))


class _GraniteTail(Module):
    module_kind = "GraniteTail"

    def __init__(self, *, d: int, v: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d)
        self.head = granite_lm_output(hidden_size=d, vocab_size=v)

    def forward(self, hidden):
        return self.head(self.norm(hidden))


def test_gemma4_tied_cap_monotonic_in_vocab() -> None:
    ctx = reference_invocation(default_dtype=DType.FP16)
    s, d = 32, 64

    small = compose_graph(
        lambda ctx: _GemmaTail(ctx, d=d, v=256),
        (Tensor(shape=(s, d), requires_grad=True),),
    )
    large = compose_graph(
        lambda ctx: _GemmaTail(ctx, d=d, v=512),
        (Tensor(shape=(s, d), requires_grad=True),),
    )
    assert estimate(large, ctx).flops.total_flops > estimate(small, ctx).flops.total_flops


def test_gemma4_cap_exceeds_uncapped_granite_at_same_dims() -> None:
    ctx = reference_invocation(default_dtype=DType.FP16)
    s, d, v = 32, 64, 256
    capped = compose_graph(
        lambda ctx: _GemmaTail(ctx, d=d, v=v),
        (Tensor(shape=(s, d), requires_grad=True),),
    )
    uncapped = compose_graph(
        lambda _ctx: _GraniteTail(d=d, v=v),
        (Tensor(shape=(s, d), requires_grad=True),),
    )
    assert (
        estimate(capped, ctx).flops.total_flops
        > estimate(uncapped, ctx).flops.total_flops
    )
