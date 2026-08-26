"""xIELU activation module."""

from __future__ import annotations

from ..core.composition import GraphTensor, Module
from ..core.functional import add, exp, minimum, multiply, scalar_input, subtract, where
from ..core.metadata import ValueMetadata
from ._helpers import require_context


class XIELU(Module):
    """xIELU activation matching HuggingFace ``XIELUActivation`` eager Python path.

    Piecewise over rank-2 ``(S, d_ff)``. Trainable ``alpha_p`` / ``alpha_n`` are
    registered as parameters; structural graph inputs ``effective_alpha_p`` and
    ``effective_alpha_n`` stand in for ``softplus(parameter)`` during composition
    (mirrors Atto folding softplus FLOPs into the leaf).
    """

    module_kind = "XIELU"

    def __init__(
        self,
        *,
        beta: float = 0.5,
        eps: float = -1e-6,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps
        self._one: GraphTensor | None = None
        self._beta: GraphTensor | None = None
        self._eps: GraphTensor | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        ctx = require_context()
        self._one = scalar_input(semantic_type="one")
        self._beta = scalar_input(semantic_type="xielu_beta")
        self._eps = scalar_input(semantic_type="xielu_eps")
        if "alpha_p" not in self._parameters:
            self.alpha_p = ctx.parameter(ValueMetadata((1,), semantic_type="weight"))
        if "alpha_n" not in self._parameters:
            self.alpha_n = ctx.parameter(ValueMetadata((1,), semantic_type="weight"))
        self._initialized = True

    def forward(self, value: GraphTensor) -> GraphTensor:
        self._ensure_initialized()
        assert self._one is not None and self._beta is not None and self._eps is not None

        if len(value.metadata.shape) != 2:
            raise ValueError(
                f"XIELU expects rank-2 input (S, d_ff), got {value.metadata.shape}"
            )

        ctx = require_context()
        is_positive = ctx.input(
            ValueMetadata(
                value.metadata.shape,
                semantic_type="gt_zero_mask",
                requires_grad=False,
            )
        )
        effective_alpha_p = ctx.input(
            ValueMetadata((1,), semantic_type="effective_alpha_p", requires_grad=False)
        )
        effective_alpha_n = ctx.input(
            ValueMetadata((1,), semantic_type="effective_alpha_n", requires_grad=False)
        )

        squared = multiply(value, value)
        positive = add(
            multiply(effective_alpha_p, squared),
            multiply(self._beta, value),
        )

        clamped = minimum(value, self._eps)
        expm1 = subtract(exp(clamped), self._one)
        negative = add(
            multiply(subtract(expm1, value), effective_alpha_n),
            multiply(self._beta, value),
        )

        return where(is_positive, positive, negative)
