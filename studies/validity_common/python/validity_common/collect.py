"""Collect the 2×2 within-subject factorial. Framework §4 and §7.5.

Process-wide cuBLAS warmup happens once. Each subject is measured in both
precisions. Completed rows go to evaluation.csv. Failures go to ledger.csv.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from validity_common.catalog import Catalog
from validity_common.io import (
    write_evaluation,
    write_ledger,
    write_run_meta,
    write_subjects,
)
from validity_common.protocol_defaults import (
    ALPHA,
    EQUIVALENCE_MARGIN,
    PER_COMBINATION_POWER,
    PHASES,
    PRECISIONS,
    TRAINING_STEPS,
)
from validity_common.schema import (
    LedgerRow,
    PrecisionOutcome,
    Subject,
    input_seed,
)

logger = logging.getLogger(__name__)

MeasurePrecision = Callable[..., PrecisionOutcome]
WarmupFn = Callable[..., None]


def classify_failure(exc: BaseException) -> str:
    """Map an exception to a ledger kind. Used by tests and adapters."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "outofmemory" in name or "out of memory" in text or isinstance(exc, MemoryError):
        return "oom"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "build"
    return "measure"


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
    study: str,
    model_id: str,
    twin_mode: str,
    write_subjects_fn=None,
) -> Path:
    """Measure every subject. Creates evaluation.csv and ledger.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = catalog or Catalog()

    write = write_subjects_fn or write_subjects
    write(output_dir / "subjects.csv", subjects)
    write_evaluation(output_dir / "evaluation.csv", [])
    write_ledger(output_dir / "ledger.csv", [])
    write_run_meta(
        output_dir / "run_meta.json",
        _run_meta(
            subjects,
            master_seed=master_seed,
            catalog=catalog,
            study=study,
            model_id=model_id,
            twin_mode=twin_mode,
        ),
    )

    if measure_precision is None:
        raise ValueError(
            "measure_precision is required; family collect shims bind the default adapter"
        )

    if warmup is not None:
        warmup(device)

    for subject in subjects:
        seed = input_seed(master_seed, subject.subject_id)
        for precision in PRECISIONS:
            try:
                outcome = measure_precision(
                    subject,
                    precision,
                    device=device,
                    cuda_capability=cuda_capability,
                    input_seed=seed,
                    process_warmed=True,
                )
            except Exception as exc:
                outcome = PrecisionOutcome(
                    rows=(),
                    ledger=(
                        LedgerRow(
                            subject_id=subject.subject_id,
                            configuration_id=subject.configuration_id,
                            phase="setup",
                            precision=precision,
                            seq_len=subject.seq_len,
                            batch_size=subject.batch_size,
                            failure_kind=classify_failure(exc),
                            message=f"{type(exc).__name__}: {exc}"[:500],
                        ),
                    ),
                )
            if outcome.rows:
                write_evaluation(
                    output_dir / "evaluation.csv",
                    outcome.rows,
                    append=True,
                )
            if outcome.ledger:
                write_ledger(
                    output_dir / "ledger.csv",
                    outcome.ledger,
                    append=True,
                )
                for row in outcome.ledger:
                    logger.warning(
                        "ledger %s %s %s %s: %s",
                        row.subject_id,
                        row.phase,
                        row.precision,
                        row.failure_kind,
                        row.message,
                    )

    return output_dir


def run_collection(
    n: int,
    *,
    seed: int,
    output_dir: Path,
    catalog: Catalog | None = None,
    measure_precision: MeasurePrecision | None = None,
    warmup: WarmupFn | None = None,
    study: str,
    model_id: str,
    twin_mode: str,
) -> Path:
    """Draw n unique subjects and collect them."""
    from validity_common.sample import sample_subjects

    subjects = sample_subjects(n, seed=seed, catalog=catalog)
    return collect_subjects(
        subjects,
        output_dir,
        master_seed=seed,
        catalog=catalog,
        measure_precision=measure_precision,
        warmup=warmup,
        study=study,
        model_id=model_id,
        twin_mode=twin_mode,
    )


def require_cuda() -> tuple[object, tuple[int, int]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "collection requires a CUDA device (framework §1). "
            "Pass a fake measure_precision to collect without a GPU."
        )
    return torch.device("cuda"), torch.cuda.get_device_capability(0)


def _run_meta(
    subjects: list[Subject],
    *,
    master_seed: int,
    catalog: Catalog,
    study: str,
    model_id: str,
    twin_mode: str,
) -> dict[str, object]:
    cuda_name = None
    cuda_cap: list[int] | None = None
    torch_version = None
    transformers_version = None
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_name = torch.cuda.get_device_name(0)
            cuda_cap = list(torch.cuda.get_device_capability(0))
    except ImportError:
        pass
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        pass

    return {
        "study": study,
        "model_id": model_id,
        "master_seed": master_seed,
        "n_subjects": len(subjects),
        "k_training_steps": TRAINING_STEPS,
        "twin_mode": twin_mode,
        "phases": list(PHASES),
        "precisions": list(PRECISIONS),
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "alpha": ALPHA,
        "per_combination_power": PER_COMBINATION_POWER,
        "catalog": catalog.as_meta(),
        "git_commit": _git_commit(),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "cuda_device_name": cuda_name,
        "cuda_capability": cuda_cap,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "subject_ids": [subject.subject_id for subject in subjects],
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
