"""FLOP counting dialects for Zepto vs FlopCounterMode comparison."""

from typing import Literal

FlopCountingPolicy = Literal["zepto", "flop_counter_mode"]

DEFAULT_FLOP_POLICY: FlopCountingPolicy = "zepto"

FCM_OPERATION_FAMILIES: frozenset[str] = frozenset(
    {
        "matmul",
        "linear_matmul",
        "conv2d",
        "conv3d",
    }
)


def is_fcm_operation_family(family: str) -> bool:
    return family in FCM_OPERATION_FAMILIES
