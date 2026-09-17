"""Depthwise causal 1-D convolution with optional SiLU."""

from __future__ import annotations

from typing import Literal

from zepto.compose import Module, Parameter, Tensor
from zepto.compose.context import Compose
from zepto.semantic import Add, Concat, Multiply, Reshape, Sigmoid, Split, Transpose

from ._concat import concat_leading
from ._helpers import add_bias_parameter, scale_by_parameter


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

    def _history_prefill(self, value: Tensor) -> tuple[Tensor, ...]:
        seq_len = value.shape[0]
        k = self.kernel_size
        pad = self._graph_const(
            (k - 1, self.channels),
            semantic_type="zero",
        )
        extended = Concat(axis=0, input_count=2)(pad, value)  # type: ignore[call-arg]
        parts = Split(sizes=(1,) * (seq_len + k - 1))(extended)
        return parts

    def _history_decode(
        self, value: Tensor, conv_state_in: Tensor
    ) -> tuple[Tensor, ...]:
        if conv_state_in.shape != (self.channels, self.kernel_size):
            raise ValueError(
                f"conv_state_in expected ({self.channels}, {self.kernel_size}), "
                f"got {conv_state_in.shape}"
            )
        state_seq = Transpose(permutation=(1, 0))(conv_state_in)
        state_parts = Split(sizes=(1,) * self.kernel_size)(state_seq)
        window = state_parts[1:] + (value,)
        return window

    def _state_out_prefill(self, value: Tensor) -> Tensor:
        seq_len = value.shape[0]
        k = self.kernel_size
        if seq_len >= k:
            tail = Split(sizes=(seq_len - k, k))(value)
            return Transpose(permutation=(1, 0))(tail[1])  # type: ignore[return-value]
        pad_len = k - seq_len
        pad = self._graph_const((pad_len, self.channels), semantic_type="zero")
        padded = Concat(axis=0, input_count=2)(pad, value)  # type: ignore[call-arg]
        return Transpose(permutation=(1, 0))(padded)  # type: ignore[return-value]

    def _state_out_decode(
        self, value: Tensor, conv_state_in: Tensor
    ) -> Tensor:
        state_seq = Transpose(permutation=(1, 0))(conv_state_in)
        state_parts = Split(sizes=(1,) * self.kernel_size)(state_seq)
        window = state_parts[1:] + (value,)
        stacked = Concat(axis=0, input_count=len(window))(*window)  # type: ignore[call-arg]
        return Transpose(permutation=(1, 0))(stacked)  # type: ignore[return-value]

    def forward(
        self,
        value: Tensor,
        conv_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if value.shape[-1] != self.channels:
            raise ValueError(
                f"expected channel dim {self.channels}, got {value.shape}"
            )
        if len(value.shape) != 2:
            raise ValueError("DepthwiseCausalConv1d expects rank-2 (S, C) input")

        if conv_state_in is None:
            history = self._history_prefill(value)
            seq_len = value.shape[0]
            outputs: list[Tensor] = []
            k = self.kernel_size
            for t in range(seq_len):
                window = history[t : t + k]
                out = self._conv_sample(window)
                outputs.append(Reshape(shape=(1, self.channels))(out))
            output = concat_leading(tuple(outputs))
            state_out = self._state_out_prefill(value)
            return output, state_out

        window = self._history_decode(value, conv_state_in)
        out = self._conv_sample(window)
        output = Reshape(shape=(1, self.channels))(out)
        state_out = self._state_out_decode(value, conv_state_in)
        return output, state_out


__all__ = ["DepthwiseCausalConv1d"]
