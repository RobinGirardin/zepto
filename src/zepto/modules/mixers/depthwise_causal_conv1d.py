"""Depthwise causal 1-D convolution with optional SiLU."""

from __future__ import annotations

from typing import Literal

from zepto.compose import Module, Parameter, Tensor
from zepto.compose.context import Compose
from zepto.semantic import Add, Concat, Multiply, Reshape, Sigmoid, Split, Transpose

from zepto.modules._internal._batch import batch_seq_dims
from zepto.modules._internal._concat import concat_on_sequence, split_leading
from zepto.modules._internal._helpers import add_bias_parameter, scale_by_parameter


class DepthwiseCausalConv1d(Module):
    """Depthwise causal conv along the sequence axis (reference K-sample buffer)."""

    module_kind = "DepthwiseCausalConv1d"

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        *,
        activation: Literal["none", "silu"] = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 0:
            raise ValueError("channels and kernel_size must be positive")
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = bias
        for lag in range(kernel_size):
            setattr(
                self,
                f"weight_{lag}",
                Parameter(shape=(channels,), semantic_type="weight"),
            )
        if bias:
            self.bias = Parameter(shape=(channels,), semantic_type="bias")

    def _weight(self, lag: int) -> Parameter:
        return getattr(self, f"weight_{lag}")  # type: ignore[no-any-return]

    @staticmethod
    def _graph_const(shape: tuple[int, ...], *, semantic_type: str) -> Tensor:
        ctx = Compose.current()
        if ctx is None:
            raise RuntimeError("DepthwiseCausalConv1d requires an active Compose context")
        return ctx.input(
            Tensor(shape=shape, semantic_type=semantic_type, requires_grad=False)
        )

    def _conv_sample(self, window: tuple[Tensor, ...]) -> Tensor:
        if len(window) != self.kernel_size:
            raise ValueError("window length must match kernel_size")
        terms = [
            scale_by_parameter(window[lag], self._weight(lag))
            for lag in range(self.kernel_size)
        ]
        acc = terms[0]
        for term in terms[1:]:
            acc = Add()(acc, term)  # type: ignore[assignment]
        if self.use_bias:
            acc = add_bias_parameter(acc, self.bias)  # type: ignore[arg-type]
        if self.activation == "silu":
            sig = Sigmoid()(acc)  # type: ignore[arg-type]
            acc = Multiply()(acc, sig)  # type: ignore[assignment]
        return acc  # type: ignore[return-value]

    def _validate_conv_state(self, conv_state_in: Tensor, batch: int) -> None:
        k = self.kernel_size
        c = self.channels
        if batch == 1:
            expected = (c, k)
        else:
            expected = (batch, c, k)
        if conv_state_in.shape != expected:
            raise ValueError(
                f"conv_state_in expected {expected} for batch={batch}, "
                f"got {conv_state_in.shape}"
            )

    def _history_prefill(self, value: Tensor, *, batch: int, seq_len: int) -> tuple[Tensor, ...]:
        k = self.kernel_size
        if batch == 1:
            pad = self._graph_const(
                (k - 1, self.channels),
                semantic_type="zero",
            )
            extended = Concat(axis=0, input_count=2)(pad, value)  # type: ignore[call-arg]
            return split_leading(extended, seq_len + k - 1)

        pad = self._graph_const(
            (batch, k - 1, self.channels),
            semantic_type="zero",
        )
        extended = Concat(axis=1, input_count=2)(pad, value)  # type: ignore[call-arg]
        moved = Transpose(permutation=(1, 0, 2))(extended)
        return split_leading(moved, seq_len + k - 1)

    def _window_from_history(
        self, history: tuple[Tensor, ...], t: int, *, batch: int
    ) -> tuple[Tensor, ...]:
        k = self.kernel_size
        if batch == 1:
            return history[t : t + k]
        return tuple(
            Reshape(shape=(batch, self.channels))(history[t + lag])
            for lag in range(k)
        )

    def _history_decode(
        self, value: Tensor, conv_state_in: Tensor, *, batch: int
    ) -> tuple[Tensor, ...]:
        k = self.kernel_size
        if batch == 1:
            state_seq = Transpose(permutation=(1, 0))(conv_state_in)
            state_parts = Split(sizes=(1,) * k)(state_seq)
            return state_parts[1:] + (value,)

        state_kb = Transpose(permutation=(0, 2, 1))(conv_state_in)
        moved = Transpose(permutation=(1, 0, 2))(state_kb)
        state_parts = split_leading(moved, k)
        tail = Reshape(shape=(batch, self.channels))(value)
        return state_parts[1:] + (tail,)

    def _state_out_prefill(self, value: Tensor, *, batch: int, seq_len: int) -> Tensor:
        k = self.kernel_size
        if batch == 1:
            if seq_len >= k:
                tail = Split(sizes=(seq_len - k, k))(value)
                return Transpose(permutation=(1, 0))(tail[1])  # type: ignore[return-value]
            pad_len = k - seq_len
            pad = self._graph_const((pad_len, self.channels), semantic_type="zero")
            padded = Concat(axis=0, input_count=2)(pad, value)  # type: ignore[call-arg]
            return Transpose(permutation=(1, 0))(padded)  # type: ignore[return-value]

        moved = Transpose(permutation=(1, 0, 2))(value)
        if seq_len >= k:
            tail = Split(sizes=(seq_len - k, k))(moved)
            return Transpose(permutation=(1, 2, 0))(tail[1])  # type: ignore[return-value]
        pad_len = k - seq_len
        pad = self._graph_const(
            (pad_len, batch, self.channels), semantic_type="zero"
        )
        padded = Concat(axis=0, input_count=2)(pad, moved)  # type: ignore[call-arg]
        return Transpose(permutation=(1, 2, 0))(padded)  # type: ignore[return-value]

    def _state_out_decode(
        self, value: Tensor, conv_state_in: Tensor, *, batch: int
    ) -> Tensor:
        window = self._history_decode(value, conv_state_in, batch=batch)
        if batch == 1:
            stacked = Concat(axis=0, input_count=len(window))(*window)  # type: ignore[call-arg]
            return Transpose(permutation=(1, 0))(stacked)  # type: ignore[return-value]

        stacked = Concat(axis=0, input_count=len(window))(*window)  # type: ignore[call-arg]
        return Transpose(permutation=(1, 2, 0))(stacked)  # type: ignore[return-value]

    def forward(
        self,
        value: Tensor,
        conv_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if value.shape[-1] != self.channels:
            raise ValueError(
                f"expected channel dim {self.channels}, got {value.shape}"
            )
        rank = len(value.shape)
        if rank not in (2, 3):
            raise ValueError(
                "DepthwiseCausalConv1d expects rank-2 (S, C) or rank-3 (B, S, C)"
            )
        batch, seq_len = batch_seq_dims(value)

        if conv_state_in is None:
            history = self._history_prefill(value, batch=batch, seq_len=seq_len)
            outputs: list[Tensor] = []
            for t in range(seq_len):
                window = self._window_from_history(history, t, batch=batch)
                out = self._conv_sample(window)
                if batch == 1:
                    outputs.append(Reshape(shape=(1, self.channels))(out))
                else:
                    outputs.append(Reshape(shape=(batch, 1, self.channels))(out))
            output = concat_on_sequence(tuple(outputs))
            state_out = self._state_out_prefill(value, batch=batch, seq_len=seq_len)
            return output, state_out

        self._validate_conv_state(conv_state_in, batch)
        window = self._history_decode(value, conv_state_in, batch=batch)
        out = self._conv_sample(window)
        if batch == 1:
            output = Reshape(shape=(1, self.channels))(out)
        else:
            output = Reshape(shape=(batch, 1, self.channels))(out)
        state_out = self._state_out_decode(value, conv_state_in, batch=batch)
        return output, state_out


__all__ = ["DepthwiseCausalConv1d"]
