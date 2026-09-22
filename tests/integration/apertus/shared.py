"""Shared helpers for Apertus integration tests."""

from __future__ import annotations

from dataclasses import replace

from zepto.analysis import PrecisionPolicy, lower, reference_invocation

GOLDEN = dict(
    hidden_size=32,
    intermediate_size=64,
    num_heads=8,
    num_kv_heads=2,
    num_layers=2,
    vocab_size=100,
    seq_len=8,
    head_dim=4,
)


def golden_apertus_ctx(
    *,
    phase: str = "forward",
    fused: bool = True,
    cuda_capability: tuple[int, int] | None = (8, 0),
):
    """Mixed-precision context: fp16 params, fp32 grads, fp32 Adam state."""
    kwargs: dict = {
        "phase": phase,
        "precision": PrecisionPolicy.from_byte_sizes(param_bytes=2, grad_bytes=4),
        "optim_prec": 4,
        "attention_backend": "eager",
        "hardware": "cuda",
    }
    if cuda_capability is not None:
        kwargs["compute_capability"] = cuda_capability
    if fused:
        kwargs["requested_capabilities"] = frozenset({"fused", "flash"})
    return reference_invocation(**kwargs)


def golden_apertus_hf_flop_ctx(
    *,
    phase: str = "forward",
    cuda_capability: tuple[int, int] | None = (8, 0),
):
    """Zepto ctx aligned with smoke ``build_parity_ctx`` (HF eager ↔ sdpa-math GQA)."""
    from zepto.semantic.metadata import DType

    kwargs: dict = {
        "phase": phase,
        "default_dtype": DType.FP32,
        "optim_prec": 4,
        "attention_backend": "eager",
        "hardware": "cuda",
        "requested_capabilities": frozenset({"fused", "sdpa", "gqa"}),
        "state": (("sdpa_mode", "math"),),
    }
    if cuda_capability is not None:
        kwargs["compute_capability"] = cuda_capability
    return reference_invocation(**kwargs)


def merge_train_lowered(graph, ctx):
    """Merge forward and backward lowering events for training simulation."""
    base = ctx
    forward = lower(graph, replace(base, phase="forward"))
    backward = lower(graph, replace(base, phase="backward"))
    merged_nodes = []
    for fwd_node, bwd_node in zip(forward.nodes, backward.nodes, strict=True):
        backward_events = tuple(
            event
            for event in bwd_node.resource_events
            if event.phase == "backward"
        )
        merged_nodes.append(
            replace(
                fwd_node,
                resource_events=fwd_node.resource_events + backward_events,
                backward_flops=bwd_node.backward_flops,
            )
        )
    return replace(forward, nodes=tuple(merged_nodes), context=replace(base, phase="full"))
