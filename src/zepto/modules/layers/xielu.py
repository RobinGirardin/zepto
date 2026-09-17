"""xIELU activation module."""

from __future__ import annotations

from zepto.compose import Module, Parameter, Tensor
from zepto.semantic import Add, Exp, GreaterThan, Minimum, Multiply, Subtract, Where


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

        self._one = Tensor(shape=(1,), semantic_type="one", requires_grad=False)
        self._beta = Tensor(shape=(1,), semantic_type="xielu_beta", requires_grad=False)
        self._eps = Tensor(shape=(1,), semantic_type="xielu_eps", requires_grad=False)
        self._zero = Tensor(
            shape=(1,), semantic_type="constant_zero", requires_grad=False
        )
        self._effective_alpha_p = Tensor(
            shape=(1,), semantic_type="effective_alpha_p", requires_grad=False
        )
        self._effective_alpha_n = Tensor(
            shape=(1,), semantic_type="effective_alpha_n", requires_grad=False
        )
        self.alpha_p = Parameter(shape=(1,), semantic_type="weight")
        self.alpha_n = Parameter(shape=(1,), semantic_type="weight")

    def forward(self, value: Tensor) -> Tensor:
        if len(value.shape) != 2:
            raise ValueError(
                f"XIELU expects rank-2 input (S, d_ff), got {value.shape}"
            )

        is_positive = GreaterThan()(value, self._zero)

        squared = Multiply()(value, value)
        positive = Add()(
            Multiply()(self._effective_alpha_p, squared),
            Multiply()(self._beta, value),
        )

        clamped = Minimum()(value, self._eps)
        expm1 = Subtract()(Exp()(clamped), self._one)
        negative = Add()(
            Multiply()(Subtract()(expm1, value), self._effective_alpha_n),
            Multiply()(self._beta, value),
        )

        return Where()(is_positive, positive, negative)  # type: ignore[return-value]
