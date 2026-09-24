"""Catalog identities match the locked worked example. Framework §7.2."""

from __future__ import annotations

from apertus_validity import Catalog, FreeKnobs, derive_architecture
from apertus_validity.identities import assert_identities, round_to_multiple


def test_derivation_matches_apertus_family_core() -> None:
    from zepto.empirical.models.apertus_core import ApertusFamilyCore

    knobs = FreeKnobs(
        num_kv_heads=4,
        gqa_group=2,
        head_dim=128,
        ffn_mult=5.25,
        num_layers=4,
        vocab_size=1024,
    )
    architecture = derive_architecture(knobs)
    family = ApertusFamilyCore().derive_options(
        {
            "num_kv_heads": 4,
            "gqa_group": 2,
            "head_dim": 128,
            "ffn_mult": 5.25,
            "num_layers": 4,
            "vocab_size": 1024,
        }
    )
    assert architecture.as_options() == family


def test_worked_example_derivation() -> None:
    knobs = FreeKnobs(
        num_kv_heads=4,
        gqa_group=2,
        head_dim=128,
        ffn_mult=5.25,
        num_layers=4,
        vocab_size=1024,
    )
    architecture = derive_architecture(knobs)
    assert architecture.num_heads == 8
    assert architecture.hidden_size == 1024
    assert architecture.intermediate_size == 5376
    assert architecture.intermediate_size == round_to_multiple(5.25 * 1024)
    assert_identities(architecture)


def test_default_catalog_architectures_are_valid() -> None:
    catalog = Catalog()
    pairs = catalog.architectures()
    assert len(pairs) == 3 * 3 * 2 * 3 * 2 * 2
    for knobs, architecture in pairs:
        assert_identities(architecture)
        assert architecture.num_heads == knobs.num_kv_heads * knobs.gqa_group
        assert architecture.hidden_size == architecture.num_heads * knobs.head_dim
        assert knobs.gqa_group not in architecture.as_options()


def test_large_vocab_is_off_by_default() -> None:
    catalog = Catalog()
    assert 131072 not in catalog.knobs.vocab_values()
    on = Catalog(knobs=type(catalog.knobs)(include_large_vocab=True, large_vocab=131072))
    assert 131072 in on.knobs.vocab_values()


def test_workload_respects_rope_window_and_token_cap() -> None:
    from apertus_validity.catalog import WorkloadCatalog

    pairs = WorkloadCatalog(max_tokens=64).feasible_pairs()
    assert (1, 32) in pairs
    assert (2, 32) in pairs
    assert (4, 32) not in pairs
    assert all(seq >= 32 for _, seq in pairs)
