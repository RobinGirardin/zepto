"""MTP preset factory smoke tests."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.analysis.lowering import LoweringRegistry
from zepto.analysis.lowering.implementations import register_identity_defaults
from zepto.compose import Module, Tensor, compose_graph
from zepto.modules.attention.materialized_causal_mask import MaterializedCausalMask
from zepto.modules.mtp.mtp_presets import nemotron_h_mtp_stage, qwen38_mtp_head
from zepto.modules.moe.moe_presets import nemotron_moe_block


class _IdentityHold(Module):
    module_kind = "IdentityHold"

    def forward(self, value: Tensor) -> Tensor:
        return value


def test_qwen38_mtp_preset_builds_at_reference_dims() -> None:
    captured: list = []

    def factory(_ctx):
        stage = qwen38_mtp_head()
        captured.append(stage)
        return stage

    compose_graph(
        factory,
        (
            Tensor(shape=(4, 5120), requires_grad=True),
            Tensor(shape=(4,), semantic_type="token_ids", requires_grad=False),
        ),
    )
    assert captured[0].config.hidden_size == 5120
    assert captured[0].config.mlp_intermediate == 17408


def test_nemotron_mtp_preset_layer_specs() -> None:
    captured: list = []

    def factory(_ctx):
        stage = nemotron_h_mtp_stage(seq_len=16)
        captured.append(stage)
        return _IdentityHold()

    compose_graph(factory, (Tensor(shape=(4, 2688), requires_grad=True),))
    assert len(captured[0].config.layer_specs) == 2
    assert captured[0].config.layer_specs[0].mixer == "attention"
    assert captured[0].config.layer_specs[1].mixer == "moe"


def test_mtp_flops_scale_with_intermediate() -> None:
    s, d = 4, 64
    inputs = (
        Tensor(shape=(s, d), requires_grad=True),
        Tensor(shape=(s,), semantic_type="token_ids", requires_grad=False),
    )
    small = compose_graph(
        lambda _ctx: qwen38_mtp_head(
            hidden_size=d, vocab_size=128, mlp_intermediate=128
        ),
        inputs,
    )
    large = compose_graph(
        lambda _ctx: qwen38_mtp_head(
            hidden_size=d, vocab_size=128, mlp_intermediate=512
        ),
        inputs,
    )
    registry = LoweringRegistry()
    register_identity_defaults(registry)
    ctx = reference_invocation(phase="forward")
    small_flops = sum(
        n.forward_flops for n in lower(small, ctx, registry=registry).nodes
    )
    large_flops = sum(
        n.forward_flops for n in lower(large, ctx, registry=registry).nodes
    )
    assert large_flops > small_flops


def test_nemotron_mtp_compose_at_small_dims() -> None:
    s, d = 4, 64
    mask = MaterializedCausalMask(s)

    class NemotronProbe(Module):
        module_kind = "NemotronProbe"

        def __init__(self, _ctx) -> None:
            super().__init__()
            self.stage = nemotron_h_mtp_stage(
                hidden_size=d,
                vocab_size=128,
                seq_len=s,
                moe_factory=lambda: nemotron_moe_block(hidden_size=d, seq_len=s),
            )

        def forward(
            self,
            trunk: Tensor,
            token_ids: Tensor,
        ) -> Tensor:
            return self.stage(trunk, token_ids, mask())  # type: ignore[return-value]

    graph = compose_graph(
        NemotronProbe,
        (
            Tensor(shape=(s, d), requires_grad=True),
            Tensor(shape=(s,), semantic_type="token_ids", requires_grad=False),
        ),
    )
    assert len(graph.nodes) > 0
