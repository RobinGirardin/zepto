"""Subjects are unique (configuration_id, seq_len, batch_size). Framework §4."""

from __future__ import annotations

import pytest

from granite_validity import Catalog, enumerate_frame, sample_subjects
from granite_validity.catalog import KnobCatalog, WorkloadCatalog


def _tiny_catalog() -> Catalog:
    return Catalog(
        knobs=KnobCatalog(
            gqa_group=(2,),
            num_kv_heads=(4,),
            head_dim=(128,),
            num_layers=(4,),
            ffn_mult=(5.25,),
            vocab_size=(1024,),
        ),
        workload=WorkloadCatalog(seq_len=(32, 64), batch_size=(1, 2)),
    )


def test_default_frame_size_is_architectures_times_workloads() -> None:
    frame = enumerate_frame(Catalog())
    assert len(frame) == 216 * 9
    assert len({subject.subject_id for subject in frame}) == 1944


def test_frame_is_architecture_times_workload() -> None:
    frame = enumerate_frame(_tiny_catalog())
    assert len(frame) == 4
    assert len({subject.subject_id for subject in frame}) == 4


def test_same_architecture_different_workload_are_distinct_subjects() -> None:
    frame = enumerate_frame(_tiny_catalog())
    by_s = {(subject.seq_len, subject.batch_size): subject for subject in frame}
    assert by_s[(32, 1)].configuration_id == by_s[(64, 1)].configuration_id
    assert by_s[(32, 1)].subject_id != by_s[(64, 1)].subject_id
    assert by_s[(32, 1)].subject_id != by_s[(32, 2)].subject_id


def test_subject_has_no_precision() -> None:
    subject = enumerate_frame(_tiny_catalog())[0]
    assert not hasattr(subject, "precision")


def test_sample_without_replacement_is_reproducible() -> None:
    catalog = _tiny_catalog()
    first = sample_subjects(3, seed=7, catalog=catalog)
    second = sample_subjects(3, seed=7, catalog=catalog)
    assert [subject.subject_id for subject in first] == [
        subject.subject_id for subject in second
    ]
    assert len({subject.subject_id for subject in first}) == 3


def test_sample_rejects_n_larger_than_frame() -> None:
    with pytest.raises(ValueError, match="frame has only"):
        sample_subjects(99, seed=1, catalog=_tiny_catalog())
