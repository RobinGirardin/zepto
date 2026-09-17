"""Configurable dot-product attention module."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, MatMul, Multiply, RepeatKV, Reshape, Sigmoid, Transpose

from .affine_linear import AffineLinear
from .attention_config import AttentionConfig
from .attention_softmax_with_sink import AttentionSoftmaxWithSink
from .qk_norm import QKNormRMSNorm
from .rms_norm import RMSNorm
from .rope_apply import RoPEApply
from .softmax import Softmax, attention_softmax_scale
from .softplus import Softplus


class FlexibleAttention(Module):
    """Masked grouped-query self-attention driven by a frozen ``AttentionConfig``.

    Pipeline: Q/K/V projections → head reshape/transpose → optional QK-Norm →
    optional RoPE → RepeatKV → ``QK^T`` → mask add → softmax (or sink) →
    ``PV`` → optional output gate → output projection.

    ``hidden_states`` is the normalized input to attention (caller's
    responsibility), consistent with ``ApertusDecoderBlock``.

    """

    module_kind = "FlexibleAttention"

    def __init__(
        self,
        config: AttentionConfig,
        *,
        qk_norm: QKNormRMSNorm | None = None,
        rope: RoPEApply | None = None,
        v_norm: RMSNorm | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        cfg = config
        scale = attention_softmax_scale(cfg.head_dim)

        self.q_proj = AffineLinear(
            cfg.hidden_size, cfg.q_proj_size, bias=cfg.qkv_bias
        )
        self.k_proj = AffineLinear(
            cfg.hidden_size, cfg.kv_proj_size, bias=cfg.qkv_bias
        )
        if cfg.kv_sharing == "separate":
            self.v_proj = AffineLinear(
                cfg.hidden_size, cfg.kv_proj_size, bias=cfg.qkv_bias
            )
        self.o_proj = AffineLinear(
            cfg.q_proj_size, cfg.hidden_size, bias=cfg.o_bias
        )

        if cfg.softmax == "standard":
            self._softmax = Softmax(scale=scale)
        elif cfg.softmax == "sink":
            self._softmax = AttentionSoftmaxWithSink(
                cfg.num_q_heads, scale=scale
            )

        if cfg.output_gate != "none":
            self.gate_proj = AffineLinear(cfg.hidden_size, cfg.q_proj_size)
            if cfg.output_gate == "sigmoid":
                self._gate_activation = _SigmoidGate()
            else:
                self._gate_activation = Softplus()

        if qk_norm is not None:
            self.qk_norm = qk_norm
        if rope is not None and cfg.position != "none":
            self.rope = rope
        if v_norm is not None:
            self.v_norm = v_norm

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rope_cos: Tensor | None = None,
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        rank = len(hidden_states.shape)
        if rank not in (2, 3):
            raise ValueError(
                "FlexibleAttention expects rank-2 (S, d) or rank-3 (B, S, d) "
                f"hidden states, got {hidden_states.shape}"
            )

        cfg = self.config
        hidden = hidden_states.shape[-1]
        if hidden != cfg.hidden_size:
            raise ValueError(
                f"hidden size mismatch: expected {cfg.hidden_size}, got {hidden}"
            )

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        if cfg.kv_sharing == "separate":
            value = self.v_proj(hidden_states)  # type: ignore[attr-defined]
        else:
            value = key

        if rank == 2:
            seq_len = hidden_states.shape[0]
            query_heads = Transpose(permutation=(1, 0, 2))(
                Reshape(shape=(seq_len, cfg.num_q_heads, cfg.head_dim))(query)
            )
            key_heads = Transpose(permutation=(1, 0, 2))(
                Reshape(shape=(seq_len, cfg.num_kv_heads, cfg.head_dim))(key)
            )
            value_heads = Transpose(permutation=(1, 0, 2))(
                Reshape(shape=(seq_len, cfg.num_kv_heads, cfg.head_dim))(value)
            )
            kv_axis = 0
            kt_perm = (0, 2, 1)
            merge_perm = (1, 0, 2)
            merge_shape = (seq_len, cfg.q_proj_size)
        else:
            batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            query_heads = Transpose(permutation=(0, 2, 1, 3))(
                Reshape(shape=(batch, seq_len, cfg.num_q_heads, cfg.head_dim))(
                    query
                )
            )
            key_heads = Transpose(permutation=(0, 2, 1, 3))(
                Reshape(shape=(batch, seq_len, cfg.num_kv_heads, cfg.head_dim))(
                    key
                )
            )
            value_heads = Transpose(permutation=(0, 2, 1, 3))(
                Reshape(shape=(batch, seq_len, cfg.num_kv_heads, cfg.head_dim))(
                    value
                )
            )
            kv_axis = 1
            kt_perm = (0, 1, 3, 2)
            merge_perm = (0, 2, 1, 3)
            merge_shape = (batch, seq_len, cfg.q_proj_size)

        qk_norm = getattr(self, "qk_norm", None)
        if qk_norm is not None:
            query_heads, key_heads = qk_norm(query_heads, key_heads)  # type: ignore[misc]

        v_norm = getattr(self, "v_norm", None)
        if v_norm is not None:
            value_heads = v_norm(value_heads)  # type: ignore[assignment]

        if cfg.position != "none":
            rope = getattr(self, "rope", None)
            if rope is not None:
                if rope_cos is None or rope_sin is None:
                    raise ValueError(
                        "FlexibleAttention RoPE requires rope_cos and rope_sin"
                    )
                query_heads = rope(query_heads, rope_cos, rope_sin)
                key_heads = rope(key_heads, rope_cos, rope_sin)

        key_heads = RepeatKV(n_rep=cfg.num_kv_groups, axis=kv_axis)(key_heads)
        value_heads = RepeatKV(n_rep=cfg.num_kv_groups, axis=kv_axis)(value_heads)

        key_transposed = Transpose(permutation=kt_perm)(key_heads)
        scores = MatMul()(query_heads, key_transposed)
        masked_scores = Add()(scores, attention_mask)
        attention_weights = self._softmax(masked_scores)
        context = MatMul()(attention_weights, value_heads)

        merged = Reshape(shape=merge_shape)(
            Transpose(permutation=merge_perm)(context)
        )

        if cfg.output_gate != "none":
            gate = self.gate_proj(hidden_states)  # type: ignore[attr-defined]
            gate_activation = self._gate_activation(gate)  # type: ignore[attr-defined]
            merged = Multiply()(merged, gate_activation)  # type: ignore[call-arg]

        return self.o_proj(merged)  # type: ignore[return-value]


class _SigmoidGate(Module):
    """Thin wrapper so gate activation shares FlexibleAttention provenance."""

    module_kind = "SigmoidGate"

    def forward(self, x: Tensor) -> Tensor:
        return Sigmoid()(x)  # type: ignore[return-value]


__all__ = ["FlexibleAttention"]
