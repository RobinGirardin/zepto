"""Preset factory tests for language-model output heads."""

from __future__ import annotations

import pytest

from zepto.compose import Parameter, Tensor, compose_graph
from modules.output.output_presets import (
    gemma4_lm_output,
    granite_lm_output,
    muse_glimmer_lm_output,
)


def _weight_shape(factory, *, d: int, v: int) -> tuple[int, ...]:
    graph = compose_graph(
        lambda _ctx: factory(hidden_size=d, vocab_size=v),
        (Tensor(shape=(4, d), requires_grad=True),),
    )
    param_id = next(iter(graph.parameters))
    return graph.parameter(param_id).shape


@pytest.mark.parametrize(
    ("factory", "d", "v"),
    [
        (granite_lm_output, 4096, 100352),
        (granite_lm_output, 2880, 201088),  # gpt-oss dims via override
    ],
)
def test_untied_preset_dims(factory, d, v) -> None:
    assert _weight_shape(factory, d=d, v=v) == (d, v)


def test_reference_checkpoint_dims() -> None:
    assert _weight_shape(granite_lm_output, d=4096, v=100352) == (4096, 100352)
    from modules.output.output_presets import (
        gpt_oss_lm_output,
        laguna_xs_lm_output,
        nemotron_h_lm_output,
        qwen38_lm_output,
    )

    assert _weight_shape(gpt_oss_lm_output, d=2880, v=201088) == (2880, 201088)
    assert _weight_shape(laguna_xs_lm_output, d=2048, v=100352) == (2048, 100352)
    assert _weight_shape(nemotron_h_lm_output, d=2688, v=131072) == (2688, 131072)
    assert _weight_shape(qwen38_lm_output, d=5120, v=248320) == (5120, 248320)


def test_muse_preset_has_tanh() -> None:
    graph = compose_graph(
        lambda _ctx: muse_glimmer_lm_output(hidden_size=64, vocab_size=128),
        (Tensor(shape=(4, 64), requires_grad=True),),
    )
    families = {graph.node(n).operation_family for n in graph.nodes}
    assert "tanh" in families


def test_gemma_preset_requires_tied_weight() -> None:
    def build(ctx):
        weight = ctx.parameter(Parameter(shape=(32, 64), semantic_type="weight"))
        return gemma4_lm_output(
            tied_weight=weight, hidden_size=32, vocab_size=64
        )

    graph = compose_graph(
        build,
        (Tensor(shape=(4, 32), requires_grad=True),),
    )
    assert "tanh" in {graph.node(n).operation_family for n in graph.nodes}
    assert len(graph.parameters) == 1
