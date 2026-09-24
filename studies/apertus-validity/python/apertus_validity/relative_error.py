"""Relative error on the locked scale. Framework §2.

RE = (y − t) / t. The same absolute gap is a smaller relative error when
t is larger. The TOST bounds stay on this scale.
"""

from __future__ import annotations


def relative_error(y: float, t: float) -> float:
    if t == 0:
        raise ValueError("twin probe t must be non-zero")
    return (y - t) / t
