"""Vocabulary projection with optional logit soft-cap and tied weights."""

from __future__ import annotations

from dataclasses import dataclass

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import LinearMatMul

from .logit_soft_cap import LogitSoftCap, LogitSoftCapConfig


@dataclass(frozen=True, slots=True)
class LanguageModelOutputConfig:
    """Output head geometry and optional soft-cap / weight tying."""

    hidden_size: int
    vocab_size: int
    tie_weights: bool = False
    soft_cap: LogitSoftCapConfig | None = None


class LanguageModelOutput(Module):
    """``Linear(d→V)`` plus optional ``LogitSoftCap`` on logits."""

    module_kind = "LanguageModelOutput"

    def __init__(
        self,
        config: LanguageModelOutputConfig,
        *,
        weight: Parameter | None = None,
    ) -> None:
        super().__init__()
        d, v = config.hidden_size, config.vocab_size
        if d <= 0 or v <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        self.config = config
        self.hidden_size = d
        self.vocab_size = v

        if weight is None:
            if config.tie_weights:
                raise ValueError(
                    "tie_weights=True requires an injected shared weight Parameter"
                )
            resolved = Parameter(shape=(d, v), semantic_type="weight")
        else:
            if weight.shape != (d, v):
                raise ValueError(
                    f"weight shape must be {(d, v)}, got {weight.shape}"
                )
            resolved = weight
        self.weight = resolved

        self._soft_cap: LogitSoftCap | None = None
        if config.soft_cap is not None:
            self._soft_cap = LogitSoftCap(config.soft_cap)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if len(hidden_states.shape) != 2:
            raise ValueError(
                f"LanguageModelOutput expects rank-2 hidden states (S, d), "
                f"got {hidden_states.shape}"
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden dim mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.shape[1]}"
            )
        logits = LinearMatMul()(
            hidden_states,
            parameters=(self.weight,),
        )
        if self._soft_cap is not None:
            logits = self._soft_cap(logits)  # type: ignore[assignment]
        return logits  # type: ignore[return-value]


__all__ = ["LanguageModelOutput", "LanguageModelOutputConfig"]
