"""Enumerate the valid subject frame, then draw unique subjects.

Framework §6–7: n counts distinct (configuration_id, seq_len, batch_size).
Two workloads on the same graph are two subjects. Precision is not a knob.
"""

from __future__ import annotations

import random

from apertus_validity.catalog import Catalog
from apertus_validity.schema import Subject, configuration_id, subject_id


def enumerate_frame(catalog: Catalog | None = None) -> list[Subject]:
    """Every valid architecture × every feasible (B, S)."""
    catalog = catalog or Catalog()
    subjects: list[Subject] = []
    seen: set[str] = set()
    for knobs, architecture in catalog.architectures():
        cfg_id = configuration_id(architecture.as_options())
        for batch_size, seq_len in catalog.workload.feasible_pairs():
            sid = subject_id(cfg_id, seq_len=seq_len, batch_size=batch_size)
            if sid in seen:
                continue
            seen.add(sid)
            subjects.append(
                Subject(
                    subject_id=sid,
                    configuration_id=cfg_id,
                    knobs=knobs,
                    architecture=architecture,
                    seq_len=seq_len,
                    batch_size=batch_size,
                )
            )
    return subjects


def sample_subjects(
    n: int,
    *,
    seed: int,
    catalog: Catalog | None = None,
) -> list[Subject]:
    """Uniform draw without replacement from the finite valid frame."""
    if n <= 0:
        raise ValueError("n must be positive")
    frame = enumerate_frame(catalog)
    if n > len(frame):
        raise ValueError(
            f"requested n={n} subjects but the frame has only {len(frame)}"
        )
    rng = random.Random(seed)
    return rng.sample(frame, n)
