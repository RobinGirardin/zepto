"""Failed measurements are written to the ledger, not dropped. Framework §5.3."""

from __future__ import annotations

from pathlib import Path

from apertus_validity import Catalog
from apertus_validity.catalog import KnobCatalog, WorkloadCatalog
from apertus_validity.collect import classify_failure, collect_subjects
from apertus_validity.io import read_evaluation, read_ledger
from apertus_validity.sample import sample_subjects
from apertus_validity.schema import EvaluationRow, LedgerRow, PrecisionOutcome, Subject


def _catalog() -> Catalog:
    return Catalog(
        knobs=KnobCatalog(
            gqa_group=(2,),
            num_kv_heads=(4,),
            head_dim=(128,),
            num_layers=(4,),
            ffn_mult=(5.25,),
            vocab_size=(1024,),
        ),
        workload=WorkloadCatalog(seq_len=(32,), batch_size=(1,)),
    )


def _row(subject: Subject, phase: str, precision: str) -> EvaluationRow:
    return EvaluationRow(
        subject_id=subject.subject_id,
        configuration_id=subject.configuration_id,
        phase=phase,
        precision=precision,
        seq_len=subject.seq_len,
        batch_size=subject.batch_size,
        step=0 if phase == "inference" else 1,
        y_vram=100,
        target_vram=100,
        target_vram_raw=100,
        cublas_infer_correction_bytes=0,
        y_flop_fcm=100,
        target_flop=100,
        y_flop=110,
    )


def test_classify_oom_from_message_and_type() -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    assert classify_failure(OutOfMemoryError("CUDA out of memory")) == "oom"
    assert classify_failure(MemoryError("oom")) == "oom"
    assert classify_failure(ImportError("no transformers")) == "build"
    assert classify_failure(RuntimeError("boom")) == "measure"


def test_collect_writes_ledger_and_keeps_completed_rows(tmp_path: Path) -> None:
    catalog = _catalog()
    subjects = sample_subjects(1, seed=1, catalog=catalog)
    subject = subjects[0]

    def fake_measure(subject: Subject, precision: str, **_kwargs) -> PrecisionOutcome:
        if precision == "fp16":
            return PrecisionOutcome(
                rows=(_row(subject, "inference", "fp16"),),
                ledger=(
                    LedgerRow(
                        subject_id=subject.subject_id,
                        configuration_id=subject.configuration_id,
                        phase="training",
                        precision="fp16",
                        seq_len=subject.seq_len,
                        batch_size=subject.batch_size,
                        failure_kind="oom",
                        message="CUDA out of memory",
                    ),
                ),
            )
        return PrecisionOutcome(
            rows=(
                _row(subject, "inference", precision),
                _row(subject, "training", precision),
            ),
            ledger=(),
        )

    collect_subjects(
        subjects,
        tmp_path,
        master_seed=1,
        catalog=catalog,
        measure_precision=fake_measure,
        warmup=lambda _device: None,
    )

    evaluation = read_evaluation(tmp_path / "evaluation.csv")
    ledger = read_ledger(tmp_path / "ledger.csv")
    assert {(row.phase, row.precision) for row in evaluation} == {
        ("inference", "fp32"),
        ("training", "fp32"),
        ("inference", "fp16"),
    }
    assert len(ledger) == 1
    assert ledger[0].failure_kind == "oom"
    assert ledger[0].phase == "training"
    assert ledger[0].precision == "fp16"


def test_unhandled_exception_becomes_setup_ledger_row(tmp_path: Path) -> None:
    catalog = _catalog()
    subjects = sample_subjects(1, seed=1, catalog=catalog)

    def exploding(_subject: Subject, _precision: str, **_kwargs) -> PrecisionOutcome:
        raise MemoryError("CUDA out of memory")

    collect_subjects(
        subjects,
        tmp_path,
        master_seed=1,
        catalog=catalog,
        measure_precision=exploding,
        warmup=lambda _device: None,
    )
    evaluation = read_evaluation(tmp_path / "evaluation.csv")
    ledger = read_ledger(tmp_path / "ledger.csv")
    assert evaluation == []
    assert len(ledger) == 2  # both precisions
    assert {row.failure_kind for row in ledger} == {"oom"}
    assert {row.phase for row in ledger} == {"setup"}
