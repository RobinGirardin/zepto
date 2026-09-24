"""Qwen3.8 text-only validity study: sample subjects, collect, write a ledger."""

from qwen38_validity.catalog import Catalog, KnobCatalog, WorkloadCatalog
from qwen38_validity.collect import collect_subjects, run_collection
from qwen38_validity.identities import Architecture, FreeKnobs, derive_architecture
from qwen38_validity.relative_error import relative_error
from qwen38_validity.sample import enumerate_frame, sample_subjects
from qwen38_validity.schema import Subject

__all__ = [
    "Architecture",
    "Catalog",
    "FreeKnobs",
    "KnobCatalog",
    "Subject",
    "WorkloadCatalog",
    "collect_subjects",
    "derive_architecture",
    "enumerate_frame",
    "relative_error",
    "run_collection",
    "sample_subjects",
]
