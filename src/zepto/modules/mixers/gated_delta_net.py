"""Qwen3-Next gated DeltaNet mixer module."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import (
    Add,
    Cast,
    Concat,
    Exp,
    LinearMatMul,
    Log,
    Multiply,
    RepeatKV,
    Reshape,
    Sigmoid,
    Split,
    Transpose,
)
from zepto.semantic.metadata import DType

from zepto.modules._internal._batch import batch_seq_dims, expect_sequence_hidden_states
from zepto.modules._internal._concat import split_last_channels
from zepto.modules._internal._helpers import (
    RMSNORM_COMPUTE_DTYPE,
    add_bias_parameter,
    scale_by_parameter,
)
from .depthwise_causal_conv1d import DepthwiseCausalConv1d
from .gated_delta_scan import GatedDeltaScan
from zepto.modules.layers.gated_rms_norm import GatedRMSNorm
from zepto.modules.layers.l2_normalize import L2Normalize
from zepto.modules.layers.linear import Linear
from .mixer_config import GatedDeltaNetConfig


class GatedDeltaNet(Module):
    """Gated DeltaNet token mixer with conv + recurrent scan state."""

    module_kind = "GatedDeltaNet"

    def __init__(self, config: GatedDeltaNetConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config
        qkvz_out = 2 * cfg.qk_proj_size + 2 * cfg.v_proj_size
        self.qkvz_proj = Linear(cfg.hidden_size, qkvz_out)
        self.ba_proj = Linear(cfg.hidden_size, 2 * cfg.num_v_heads)
        conv_c = 2 * cfg.qk_proj_size + cfg.v_proj_size
        self.conv = DepthwiseCausalConv1d(
            conv_c,
            cfg.conv_kernel_size,
            activation="silu",
            bias=cfg.conv_bias,
        )
        self.l2_norm = L2Normalize()
        self.scan = GatedDeltaScan(
            cfg.num_v_heads, cfg.head_dim, cfg.head_dim
        )
        self.out_norm = GatedRMSNorm(cfg.head_dim)
        self.o_proj = Linear(cfg.v_proj_size, cfg.hidden_size)
        self.decay_rate = Parameter(shape=(cfg.num_v_heads,), semantic_type="weight")
        self.dt_bias = Parameter(shape=(cfg.num_v_heads,), semantic_type="bias")
        self._one = Tensor(shape=(1,), semantic_type="one", requires_grad=False)
        self._inv_sqrt_head = Tensor(
            shape=(1,), semantic_type="inv_sqrt_head", requires_grad=False
        )
        self._minus_one = Tensor(
            shape=(1,), semantic_type="minus_one", requires_grad=False
        )

    def _softplus(self, x: Tensor) -> Tensor:
        exp_x = Exp()(x)
        return Log()(Add()(self._one, exp_x))  # type: ignore[return-value]

    def forward(
        self,
        hidden_states: Tensor,
        conv_state_in: Tensor | None = None,
        scan_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        rank = expect_sequence_hidden_states(hidden_states, cfg.hidden_size)
        batch, seq_len = batch_seq_dims(hidden_states)
        channel_axis = 2 if rank == 3 else 1

        qkvz = LinearMatMul()(hidden_states, parameters=(self.qkvz_proj.weight,))  # type: ignore[call-arg]
        ba = LinearMatMul()(hidden_states, parameters=(self.ba_proj.weight,))  # type: ignore[call-arg]

        q, k, v, z = split_last_channels(
            qkvz,
            (
                cfg.qk_proj_size,
                cfg.qk_proj_size,
                cfg.v_proj_size,
                cfg.v_proj_size,
            ),
        )
        b, a = split_last_channels(ba, (cfg.num_v_heads, cfg.num_v_heads))

        qkv = Concat(axis=channel_axis, input_count=3)(q, k, v)  # type: ignore[call-arg]
        conv_out, conv_state_out = self.conv(qkv, conv_state_in)
        q_c, k_c, v_c = split_last_channels(
            conv_out,
            (cfg.qk_proj_size, cfg.qk_proj_size, cfg.v_proj_size),
        )

        if rank == 2:
            q_heads = Reshape(shape=(seq_len, cfg.num_qk_heads, cfg.head_dim))(q_c)
            k_heads = Reshape(shape=(seq_len, cfg.num_qk_heads, cfg.head_dim))(k_c)
            q_heads = self.l2_norm(q_heads)
            k_heads = self.l2_norm(k_heads)
            repeat = cfg.num_v_heads // cfg.num_qk_heads
            q_heads = RepeatKV(n_rep=repeat, axis=1)(q_heads)
            k_heads = RepeatKV(n_rep=repeat, axis=1)(k_heads)
            q_heads = Multiply()(q_heads, self._inv_sqrt_head)  # type: ignore[call-arg]
            v_heads = Reshape(shape=(seq_len, cfg.num_v_heads, cfg.head_dim))(v_c)
            beta = Sigmoid()(b)
            a_fp32 = Cast(to_dtype=RMSNORM_COMPUTE_DTYPE)(a)
            a_biased = add_bias_parameter(a_fp32, self.dt_bias)  # type: ignore[arg-type]
            sp = self._softplus(a_biased)
            neg_decay = Multiply()(
                scale_by_parameter(sp, self.decay_rate), self._minus_one
            )  # type: ignore[call-arg]
            scan_out, scan_state_out = self.scan(
                q_heads,
                k_heads,
                v_heads,
                beta,
                neg_decay,
                scan_state_in,
            )
            scan_heads = Reshape(shape=(seq_len, cfg.num_v_heads, cfg.head_dim))(
                scan_out
            )
            z_heads = Reshape(shape=(seq_len, cfg.num_v_heads, cfg.head_dim))(z)
            scan_by_head = Transpose(permutation=(1, 0, 2))(scan_heads)
            z_by_head = Transpose(permutation=(1, 0, 2))(z_heads)
            scan_splits = Split(sizes=(1,) * cfg.num_v_heads)(scan_by_head)
            z_splits = Split(sizes=(1,) * cfg.num_v_heads)(z_by_head)
            merge_axis = 1
            head_seq = seq_len
        else:
            q_heads = Reshape(
                shape=(batch, seq_len, cfg.num_qk_heads, cfg.head_dim)
            )(q_c)
            k_heads = Reshape(
                shape=(batch, seq_len, cfg.num_qk_heads, cfg.head_dim)
            )(k_c)
            q_heads = self.l2_norm(q_heads)
            k_heads = self.l2_norm(k_heads)
            repeat = cfg.num_v_heads // cfg.num_qk_heads
            q_heads = RepeatKV(n_rep=repeat, axis=2)(q_heads)
            k_heads = RepeatKV(n_rep=repeat, axis=2)(k_heads)
            q_heads = Multiply()(q_heads, self._inv_sqrt_head)  # type: ignore[call-arg]
            v_heads = Reshape(
                shape=(batch, seq_len, cfg.num_v_heads, cfg.head_dim)
            )(v_c)
            beta = Sigmoid()(b)
            a_fp32 = Cast(to_dtype=RMSNORM_COMPUTE_DTYPE)(a)
            a_biased = add_bias_parameter(a_fp32, self.dt_bias)  # type: ignore[arg-type]
            sp = self._softplus(a_biased)
            neg_decay = Multiply()(
                scale_by_parameter(sp, self.decay_rate), self._minus_one
            )  # type: ignore[call-arg]
            scan_out, scan_state_out = self.scan(
                q_heads,
                k_heads,
                v_heads,
                beta,
                neg_decay,
                scan_state_in,
            )
            scan_heads = Reshape(
                shape=(batch, seq_len, cfg.num_v_heads, cfg.head_dim)
            )(scan_out)
            z_heads = Reshape(shape=(batch, seq_len, cfg.num_v_heads, cfg.head_dim))(
                z
            )
            scan_by_head = Transpose(permutation=(0, 2, 1, 3))(scan_heads)
            z_by_head = Transpose(permutation=(0, 2, 1, 3))(z_heads)
            moved_scan = Transpose(permutation=(1, 0, 2, 3))(scan_by_head)
            moved_z = Transpose(permutation=(1, 0, 2, 3))(z_by_head)
            scan_splits = Split(sizes=(1,) * cfg.num_v_heads)(moved_scan)
            z_splits = Split(sizes=(1,) * cfg.num_v_heads)(moved_z)
            merge_axis = 2
            head_seq = seq_len

        normed_heads: list[Tensor] = []
        for scan_h, z_h in zip(scan_splits, z_splits, strict=True):
            if rank == 2:
                val = Reshape(shape=(head_seq, cfg.head_dim))(scan_h)
                gate_h = Reshape(shape=(head_seq, cfg.head_dim))(z_h)
            else:
                val = Reshape(shape=(batch, head_seq, cfg.head_dim))(scan_h)
                gate_h = Reshape(shape=(batch, head_seq, cfg.head_dim))(z_h)
            normed_heads.append(self.out_norm(val, gate_h))
        merged = Concat(axis=merge_axis, input_count=len(normed_heads))(*normed_heads)  # type: ignore[call-arg]
        output = LinearMatMul()(merged, parameters=(self.o_proj.weight,))  # type: ignore[return-value]
        return output, conv_state_out, scan_state_out


__all__ = ["GatedDeltaNet"]
