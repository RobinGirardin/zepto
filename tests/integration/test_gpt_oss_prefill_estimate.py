"""GPT-OSS prefill integration estimate."""

from __future__ import annotations

from zepto.analysis import estimate, reference_invocation
from zepto.compose import Tensor, compose_graph
from zepto.modules.granite import Granite, GraniteConfig
from zepto.modules.gpt_oss import GptOss, GptOssConfig
from zepto.semantic.metadata import DType


def _flash_ctx():
    return reference_invocation(
        requested_capabilities=frozenset({"fused", "flash"}),
        default_dtype=DType.FP16,
    )


def test_gpt_oss_prefill_flops_exceed_granite_tiny_at_same_dims() -> None:
    seq_len = 128
    granite = compose_graph(
        lambda _ctx: Granite(
            config=GraniteConfig(
                hidden_size=2880,
                intermediate_size=5760,
                num_q_heads=64,
                num_kv_heads=8,
                head_dim=64,
                num_layers=2,
                vocab_size=512,
            ),
            seq_len=seq_len,
        ),
        (Tensor(shape=(seq_len,)),),
    )
    gpt = compose_graph(
        lambda _ctx: GptOss(
            config=GptOssConfig(hidden_size=2880, num_layers=2, vocab_size=512),
            seq_len=seq_len,
        ),
        (Tensor(shape=(seq_len,)),),
    )
    ctx = _flash_ctx()
    assert estimate(gpt, ctx).flops.total_flops > estimate(granite, ctx).flops.total_flops
