"""Gated GQA: attention core fuses; gate ops remain separate."""

from __future__ import annotations

from zepto.analysis import lower, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.attention import FlexibleAttention
from zepto.modules.attention_config import gated_gqa

_SEQ = 8
_HIDDEN = 256
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 64


def _flash_context(**kwargs):
    defaults: dict = {
        "requested_capabilities": frozenset({"fused", "flash"}),
    }
    defaults.update(kwargs)
    return reference_invocation(**defaults)


def test_gated_gqa_fuses_core_leaves_gate() -> None:
    graph = compose_graph(
        lambda _ctx: FlexibleAttention(
            gated_gqa(_HIDDEN, _HEADS, _KV_HEADS, head_dim=_HEAD_DIM, gate="sigmoid")
        ),
        (
            Tensor(shape=(_SEQ, _HIDDEN), requires_grad=True),
            Tensor(shape=(1, _SEQ, _SEQ), requires_grad=False),
        ),
    )
    lowered = lower(graph, _flash_context())
    gqa_nodes = [
        n for n in lowered.nodes if n.implementation.startswith("region/gqa/")
    ]
    assert len(gqa_nodes) == 1
    sigmoid_nodes = [n for n in lowered.nodes if n.implementation == "sigmoid/identity"]
    multiply_nodes = [n for n in lowered.nodes if n.implementation == "multiply/identity"]
    assert len(sigmoid_nodes) >= 1
    assert len(multiply_nodes) >= 1
