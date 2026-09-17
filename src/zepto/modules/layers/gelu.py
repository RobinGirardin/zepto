"""GELU activation modules (tanh approximation and exact erf)."""

from __future__ import annotations

from zepto.compose import Module, Tensor
from zepto.semantic import Add, Erf, Multiply, Tanh


class GELUTanh(Module):
    """Elementwise GELU with tanh approximation (HF ``NewGELUActivation`` chain).

    Lowers to ``region/gelu/tanh`` when the nine-op decomposed chain matches the
    fused region pattern.
    """

    module_kind = "GELUTanh"

    def __init__(self) -> None:
        super().__init__()
        self._kappa = Tensor(
            shape=(1,), semantic_type="gelu_kappa", requires_grad=False
        )
        self._sqrt_2_over_pi = Tensor(
            shape=(1,), semantic_type="gelu_sqrt_2_over_pi", requires_grad=False
        )
        self._one = Tensor(shape=(1,), semantic_type="one", requires_grad=False)
        self._half = Tensor(shape=(1,), semantic_type="gelu_half", requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_squared = Multiply()(x, x)
        x_cubed = Multiply()(x_squared, x)
        kappa_x_cubed = Multiply()(self._kappa, x_cubed)
        inner_sum = Add()(x, kappa_x_cubed)
        tanh_input = Multiply()(self._sqrt_2_over_pi, inner_sum)
        tanh_out = Tanh()(tanh_input)
        one_plus_tanh = Add()(self._one, tanh_out)
        half_x = Multiply()(self._half, x)
        return Multiply()(half_x, one_plus_tanh)  # type: ignore[return-value]


class GELUErf(Module):
    """Elementwise exact GELU via erf (``approximate="none"`` path).

    Lowers to ``region/gelu/erf`` when the five-op decomposed chain matches the
    fused region pattern.
    """

    module_kind = "GELUErf"

    def __init__(self) -> None:
        super().__init__()
        self._inv_sqrt2 = Tensor(
            shape=(1,), semantic_type="gelu_inv_sqrt2", requires_grad=False
        )
        self._one = Tensor(shape=(1,), semantic_type="one", requires_grad=False)
        self._half = Tensor(shape=(1,), semantic_type="gelu_half", requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        scaled = Multiply()(x, self._inv_sqrt2)
        erf_out = Erf()(scaled)
        one_plus_erf = Add()(self._one, erf_out)
        x_times_cdf = Multiply()(x, one_plus_erf)
        return Multiply()(self._half, x_times_cdf)  # type: ignore[return-value]
