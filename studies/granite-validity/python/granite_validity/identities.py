"""Re-export GQA-dense identities from the shared validity kernel."""

from validity_common.identities import (
    Architecture,
    FreeKnobs,
    assert_identities,
    derive_architecture,
    round_to_multiple,
)

__all__ = [
    "Architecture",
    "FreeKnobs",
    "assert_identities",
    "derive_architecture",
    "round_to_multiple",
]
