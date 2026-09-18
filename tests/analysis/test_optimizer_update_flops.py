"""Unit tests for optimizer update FLOP helpers."""

from __future__ import annotations

from zepto.analysis.optimizer import AdamW, OptimizerPolicy, SGD, optimizer_update_flops


def test_adamw_update_flops() -> None:
    assert optimizer_update_flops(AdamW, 400, bytes_per_element=4) == 700


def test_sgd_update_flops() -> None:
    policy = OptimizerPolicy(
        name="sgd",
        state_bytes_per_parameter=0,
        flops_per_parameter=2,
    )
    assert optimizer_update_flops(policy, 400, bytes_per_element=4) == 200


def test_zero_params_returns_zero() -> None:
    assert optimizer_update_flops(AdamW, 0, bytes_per_element=4) == 0
    assert optimizer_update_flops(SGD, 0, bytes_per_element=4) == 0
