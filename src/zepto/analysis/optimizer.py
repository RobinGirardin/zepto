"""Optimizer policy contracts for horizon training boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .lowering.context import InvocationContext


class OptimizerPolicy(ABC):
    """Optimizer persistent state and boundary update costs."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def state_bytes(
        self,
        *,
        trainable_elements: int,
        context: InvocationContext,
    ) -> int:
        """Persistent VRAM for optimizer moments (live after first training boundary)."""

    @abstractmethod
    def update_flops(
        self,
        *,
        trainable_elements: int,
        context: InvocationContext,
    ) -> int:
        """FLOPs for one optimizer step (fwd+bwd graph excluded)."""

    def update_workspace_bytes(
        self,
        *,
        context: InvocationContext,
    ) -> int:
        """Ephemeral scratch during optimizer update; default 0."""
        del context
        return 0


@dataclass(frozen=True, slots=True)
class AdamWPolicy(OptimizerPolicy):
    """AdamW: 2 state tensors × optim_prec bytes; ~7 FLOPs/param update."""

    @property
    def name(self) -> str:
        return "adamw"

    def state_bytes(
        self, *, trainable_elements: int, context: InvocationContext
    ) -> int:
        optim_prec = context.optim_prec or 4
        return 2 * trainable_elements * optim_prec

    def update_flops(
        self, *, trainable_elements: int, context: InvocationContext
    ) -> int:
        del context
        # 3 mul-add per moment + 1 param update ≈ 7 N
        return 7 * trainable_elements


@dataclass(frozen=True, slots=True)
class SGDPolicy(OptimizerPolicy):
    @property
    def name(self) -> str:
        return "sgd"

    def state_bytes(
        self, *, trainable_elements: int, context: InvocationContext
    ) -> int:
        del trainable_elements, context
        return 0

    def update_flops(
        self, *, trainable_elements: int, context: InvocationContext
    ) -> int:
        del context
        return 2 * trainable_elements


AdamW: AdamWPolicy = AdamWPolicy()
SGD: SGDPolicy = SGDPolicy()
