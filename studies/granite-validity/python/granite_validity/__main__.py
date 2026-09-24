"""Command-line entry: python -m granite_validity ..."""

from __future__ import annotations

import argparse
from pathlib import Path

from granite_validity.catalog import Catalog, KnobCatalog, WorkloadCatalog
from granite_validity.collect import run_collection
from granite_validity.io import write_subjects
from granite_validity.protocol import OPTIONAL_LARGE_VOCAB
from granite_validity.sample import enumerate_frame, sample_subjects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Granite validity study: sample subjects and collect measurements."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="draw unique subjects and write subjects.csv")
    _add_shared(sample)

    run = sub.add_parser("run", help="draw subjects and collect evaluation + ledger")
    _add_shared(run)

    frame = sub.add_parser("frame", help="print the size of the valid subject frame")
    frame.add_argument("--max-tokens", type=int, default=None)
    frame.add_argument("--include-large-vocab", action="store_true")

    args = parser.parse_args(argv)
    catalog = _catalog_from_args(args)

    if args.command == "frame":
        subjects = enumerate_frame(catalog)
        print(f"frame size: {len(subjects)} subjects")
        return 0

    output_dir = Path(args.out)
    if args.command == "sample":
        subjects = sample_subjects(args.n, seed=args.seed, catalog=catalog)
        write_subjects(output_dir / "subjects.csv", subjects)
        print(f"wrote {len(subjects)} subjects to {output_dir / 'subjects.csv'}")
        return 0

    run_collection(args.n, seed=args.seed, output_dir=output_dir, catalog=catalog)
    print(f"wrote collection artifacts under {output_dir}")
    return 0


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, required=True, help="number of unique subjects")
    parser.add_argument("--seed", type=int, required=True, help="master seed")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--max-tokens", type=int, default=None, help="T_max (B·S cap)")
    parser.add_argument(
        "--include-large-vocab",
        action="store_true",
        help="add vocab_size=100352 to the catalog",
    )


def _catalog_from_args(args: argparse.Namespace) -> Catalog:
    return Catalog(
        knobs=KnobCatalog(
            include_large_vocab=getattr(args, "include_large_vocab", False),
            large_vocab=OPTIONAL_LARGE_VOCAB,
        ),
        workload=WorkloadCatalog(max_tokens=getattr(args, "max_tokens", None)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
