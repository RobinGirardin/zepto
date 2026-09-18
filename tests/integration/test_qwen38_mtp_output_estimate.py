"""Qwen3.8 primary + MTP composed FLOP estimate."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Module, Tensor, compose_graph
from modules.mtp.mtp_presets import qwen38_mtp_head
from modules.output.output_presets import qwen38_lm_output
from modules.layers.rms_norm import RMSNorm
from zepto.semantic.metadata import DType


class _PrimaryOnly(Module):
    module_kind = "PrimaryOnly"

    def __init__(self, *, d: int, v: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d)
        self.head = qwen38_lm_output(hidden_size=d, vocab_size=v)

    def forward(self, trunk):
        return self.head(self.norm(trunk))


class _PrimaryAndMtp(Module):
    module_kind = "PrimaryAndMtp"

    def __init__(self, ctx, *, d: int, v: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d)
        self.head = qwen38_lm_output(hidden_size=d, vocab_size=v)
        self.mtp = qwen38_mtp_head(
            hidden_size=d, vocab_size=v, mlp_intermediate=256
        )
        self.trunk = ctx.input(Tensor(shape=(16, d), requires_grad=True))
        self.token_ids = ctx.input(
            Tensor(shape=(16,), semantic_type="token_ids", requires_grad=False)
        )

    def forward(self) -> Tensor:
        final = self.norm(self.trunk)
        _ = self.head(final)
        return self.mtp(self.trunk, self.token_ids)  # type: ignore[return-value]


def test_primary_plus_mtp_exceeds_primary_alone() -> None:
    ctx = reference_invocation(default_dtype=DType.FP16)
    d, v = 64, 128
    primary = compose_graph(
        lambda _ctx: _PrimaryOnly(d=d, v=v),
        (Tensor(shape=(16, d), requires_grad=True),),
    )
    combined = compose_graph(lambda ctx: _PrimaryAndMtp(ctx, d=d, v=v), ())
    primary_flops = estimate(primary, ctx).flops.total_flops
    combined_flops = estimate(combined, ctx).flops.total_flops
    assert combined_flops > primary_flops
