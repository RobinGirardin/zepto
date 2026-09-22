"""Logit soft-capping: cap * tanh(logits / cap) with optional output scale."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Tensor
from zepto.modules._internal._batch import expect_logits_rank
from zepto.semantic import Divide, Multiply, Tanh


@dataclass(frozen=True, slots=True)
class LogitSoftCapConfig:
    """Soft-cap parameters for bounded logits (Muse 20, Gemma 30)."""

    cap: float
    output_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.cap <= 0:
            raise ValueError("cap must be positive")
        if self.output_scale <= 0:
            raise ValueError("output_scale must be positive")


class LogitSoftCap(Module):
    """Apply ``output_scale * cap * tanh(logits / cap)`` on ``(S, V)`` or ``(B, S, V)``."""

    module_kind = "LogitSoftCap"

    def __init__(self, config: LogitSoftCapConfig) -> None:
        super().__init__()
        self.config = config
        self._cap = Tensor(
            shape=(1,), semantic_type="soft_cap", requires_grad=False
        )
        self._output_scale = Tensor(
            shape=(1,), semantic_type="output_scale", requires_grad=False
        )

    def forward(self, logits: Tensor) -> Tensor:
        expect_logits_rank(logits)
        scaled = Divide()(logits, self._cap)
        bounded = Tanh()(scaled)
        capped = Multiply()(bounded, self._cap)
        return Multiply()(capped, self._output_scale)  # type: ignore[return-value]


__all__ = ["LogitSoftCap", "LogitSoftCapConfig"]
