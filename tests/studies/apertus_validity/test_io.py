"""CSV round-trip for subjects, evaluation, and ledger."""

from __future__ import annotations

from pathlib import Path

from apertus_validity import Catalog, sample_subjects
from apertus_validity.catalog import KnobCatalog, WorkloadCatalog
from apertus_validity.io import (
    read_evaluation,
    read_ledger,
    read_subjects,
    write_evaluation,
    write_ledger,
    write_subjects,
)
from apertus_validity.schema import EvaluationRow, LedgerRow


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


def test_subject_csv_round_trip(tmp_path: Path) -> None:
    subjects = sample_subjects(1, seed=1, catalog=_catalog())
    path = tmp_path / "subjects.csv"
    write_subjects(path, subjects)
    loaded = read_subjects(path)
    assert loaded == subjects


def test_evaluation_and_ledger_round_trip(tmp_path: Path) -> None:
    subject = sample_subjects(1, seed=1, catalog=_catalog())[0]
    row = EvaluationRow(
        subject_id=subject.subject_id,
        configuration_id=subject.configuration_id,
        phase="inference",
        precision="fp32",
        seq_len=subject.seq_len,
        batch_size=subject.batch_size,
        step=0,
        y_vram=110,
        target_vram=100,
        target_vram_raw=108,
        cublas_infer_correction_bytes=8,
        y_flop_fcm=90,
        target_flop=100,
        y_flop=120,
    )
    ledger = LedgerRow(
        subject_id=subject.subject_id,
        configuration_id=subject.configuration_id,
        phase="training",
        precision="fp16",
        seq_len=subject.seq_len,
        batch_size=subject.batch_size,
        failure_kind="oom",
        message="CUDA out of memory",
    )
    evaluation_path = tmp_path / "evaluation.csv"
    ledger_path = tmp_path / "ledger.csv"
    write_evaluation(evaluation_path, [row])
    write_ledger(ledger_path, [ledger])
    assert read_evaluation(evaluation_path) == [row]
    assert read_ledger(ledger_path) == [ledger]
