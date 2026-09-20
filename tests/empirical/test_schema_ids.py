"""Deterministic empirical dataset IDs and CSV round-trip."""

from __future__ import annotations

from pathlib import Path

from zepto.empirical.schema import (
    EvaluationRow,
    OptionRow,
    configuration_id,
    draw_id,
    option_id,
    parse_evaluation_csv,
    write_csv_rows,
    EVALUATION_FIELDNAMES,
)


def test_option_id_stable() -> None:
    assert option_id("num_layers", 2) == "num_layers=2"


def test_configuration_id_order_invariant() -> None:
    a = {"num_layers": 2, "hidden_size": 32, "vocab_size": 100}
    b = {"vocab_size": 100, "hidden_size": 32, "num_layers": 2}
    assert configuration_id("apertus", a) == configuration_id("apertus", b)


def test_draw_id_changes_with_workload() -> None:
    cfg = configuration_id("apertus", {"hidden_size": 32, "num_layers": 2})
    d1 = draw_id(cfg, seq_len=8, batch_size=1, precision="fp32", draw_seed=1)
    d2 = draw_id(cfg, seq_len=16, batch_size=1, precision="fp32", draw_seed=1)
    assert d1 != d2
    assert len(d1) == 16


def test_evaluation_csv_round_trip(tmp_path: Path) -> None:
    row = EvaluationRow(
        model_id="apertus",
        configuration_id="abc",
        draw_id="def",
        phase="inference",
        precision="fp32",
        seq_len=8,
        batch_size=1,
        step=0,
        zepto_batch_representation="micro_sum",
        y_flop=100,
        y_vram=200,
        target_flop=101,
        target_vram=190,
        target_vram_raw=201,
        cublas_infer_correction_bytes=11,
        y_runtime_workspace=50,
        y_activations=120,
        peak_minus_before=21,
        alloc_before=180,
        target_flop_no_opt=101,
        y_flop_no_opt=100,
    )
    path = tmp_path / "evaluation.csv"
    write_csv_rows(path, EVALUATION_FIELDNAMES, [row.to_csv_row()])
    loaded = parse_evaluation_csv(path)
    assert loaded == [row]


def test_evaluation_csv_legacy_columns_default_extended_fields(
    tmp_path: Path,
) -> None:
    legacy_header = (
        "model_id,configuration_id,draw_id,phase,precision,seq_len,batch_size,"
        "step,zepto_batch_representation,y_flop,y_vram,target_flop,target_vram\n"
    )
    legacy_row = (
        "apertus,abc,def,inference,fp32,8,1,0,micro_sum,100,200,101,201\n"
    )
    path = tmp_path / "evaluation.csv"
    path.write_text(legacy_header + legacy_row, encoding="utf-8")
    loaded = parse_evaluation_csv(path)
    assert len(loaded) == 1
    row = loaded[0]
    assert row.target_vram == 201
    assert row.target_vram_raw == 201
    assert row.cublas_infer_correction_bytes == 0
    assert row.y_runtime_workspace == 0
    assert row.y_flop_no_opt == 0
