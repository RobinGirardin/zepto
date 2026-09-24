"""Granite-bound collect over the shared study kernel."""

from __future__ import annotations

from pathlib import Path

from granite_validity.protocol import MODEL_ID, TWIN_MODE
from validity_common.catalog import Catalog
from validity_common.collect import (
    MeasurePrecision,
    WarmupFn,
    classify_failure,
)
from validity_common.collect import collect_subjects as _collect
from validity_common.collect import require_cuda
from validity_common.schema import Subject

__all__ = ["classify_failure", "collect_subjects", "run_collection"]


def collect_subjects(
    subjects: list[Subject],
    output_dir: Path,
    *,
    master_seed: int,
    catalog: Catalog | None = None,
    measure_precision: MeasurePrecision | None = None,
    warmup: WarmupFn | None = None,
    device=None,
    cuda_capability: tuple[int, int] | None = None,
    **kwargs,
) -> Path:
    if measure_precision is None:
        from granite_validity.adapters import (
            measure_subject_precision,
            warmup_process,
        )

        measure_precision = measure_subject_precision
        warmup = warmup or warmup_process
        device, cuda_capability = require_cuda()
    return _collect(
        subjects,
        output_dir,
        master_seed=master_seed,
        catalog=catalog,
        measure_precision=measure_precision,
        warmup=warmup,
        device=device,
        cuda_capability=cuda_capability,
        study="granite-validity",
        model_id=MODEL_ID,
        twin_mode=TWIN_MODE,
        **kwargs,
    )


def run_collection(
    n: int,
    *,
    seed: int,
    output_dir: Path,
    catalog: Catalog | None = None,
    measure_precision: MeasurePrecision | None = None,
    warmup: WarmupFn | None = None,
) -> Path:
    from validity_common.sample import sample_subjects

    subjects = sample_subjects(n, seed=seed, catalog=catalog)
    return collect_subjects(
        subjects,
        output_dir,
        master_seed=seed,
        catalog=catalog,
        measure_precision=measure_precision,
        warmup=warmup,
    )
