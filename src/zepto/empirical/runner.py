"""End-to-end empirical cost study runner."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

from zepto.analysis.runtime.policy import cublas_workspace_bytes_per_handle

from zepto.empirical import HARNESS_VERSION
from zepto.empirical.dataset_io import (
    append_evaluation_rows,
    write_dimension_tables,
    write_evaluation_rows,
    write_methodology,
    write_run_meta,
)
from zepto.empirical.measure_torch import (
    measure_infer_torch,
    measure_train_step_torch,
    training_warmup_step,
)
from zepto.empirical.measure_zepto import measure_infer_zepto, measure_train_zepto
from zepto.empirical.models import get_model_family
from zepto.empirical.parity_ctx import build_invocation_context
from zepto.empirical.sampler import SamplerConfig, sample_configurations, sample_draws
from zepto.empirical.schema import EvaluationRow
from zepto.empirical.twin_hf import build_hf_inputs

logger = logging.getLogger(__name__)

METHODOLOGY_TEMPLATE = Path(__file__).resolve().parent / "methodology_template.md"


def infer_cublas_vram_correction_bytes(
    global_draw_index: int,
    compute_capability: tuple[int, int],
) -> int:
    """Extra cuBLAS handle workspace on infer after prior draws ran training."""
    if global_draw_index <= 0:
        return 0
    return cublas_workspace_bytes_per_handle(compute_capability)


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path
    master_seed: int
    model_id: str
    n_configs: int
    m_draws_per_config: int
    training_steps_per_draw: int
    sampler: SamplerConfig
    append_results: bool = False
    log_first_draw_spec: bool = True
    twin_mode: Literal["scored_window", "apertus_parity", "transformers_defaults"] = (
        "scored_window"
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _transformers_version() -> str | None:
    try:
        import transformers

        return transformers.__version__
    except ImportError:
        return None


def _run_meta(config: RunConfig) -> dict[str, Any]:
    cuda_name = None
    cuda_cap: tuple[int, int] | None = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
        cuda_cap = torch.cuda.get_device_capability(0)
    return {
        "harness_version": HARNESS_VERSION,
        "master_seed": config.master_seed,
        "model_id": config.model_id,
        "N_configs": config.n_configs,
        "M_draws_per_config": config.m_draws_per_config,
        "training_steps_per_draw": config.training_steps_per_draw,
        "sampler": _sampler_meta(config.sampler),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "transformers_version": _transformers_version(),
        "cuda_device_name": cuda_name,
        "cuda_capability": list(cuda_cap) if cuda_cap else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "twin_mode": config.twin_mode,
    }


def _sampler_meta(sampler: SamplerConfig) -> dict[str, Any]:
    arch = sampler.architecture

    def r(rng) -> list[int]:
        return [rng.low, rng.high]

    return {
        "seq_len": r(sampler.seq_len),
        "batch_size": r(sampler.batch_size),
        "precisions": list(sampler.precisions),
        "architecture": {
            "head_dim": r(arch.head_dim),
            "intermediate_size": r(arch.intermediate_size),
            "num_heads": r(arch.num_heads),
            "num_kv_heads": r(arch.num_kv_heads),
            "num_layers": r(arch.num_layers),
            "vocab_size": r(arch.vocab_size),
        },
        "architecture_mins": dict(sampler.architecture_mins),
    }


def _run_parameters_md(config: RunConfig, meta: dict[str, Any]) -> str:
    lines = [
        "| Field | Value |",
        "|-------|-------|",
        f"| master_seed | {config.master_seed} |",
        f"| N_configs | {config.n_configs} |",
        f"| M_draws_per_config | {config.m_draws_per_config} |",
        f"| training_steps_per_draw (K) | {config.training_steps_per_draw} |",
        f"| model_id | {config.model_id} |",
        f"| git_commit | {meta.get('git_commit')} |",
        f"| torch | {meta.get('torch_version')} |",
        f"| transformers | {meta.get('transformers_version')} |",
        f"| cuda_device | {meta.get('cuda_device_name')} |",
        f"| timestamp_utc | {meta.get('timestamp_utc')} |",
    ]
    return "\n".join(lines)


def run_draw(
    config: RunConfig,
    *,
    family,
    configuration,
    draw,
    device: torch.device,
    cuda_capability: tuple[int, int],
    global_draw_index: int,
    log_spec: bool = False,
) -> list[EvaluationRow]:
    opts = configuration.options
    ctx = build_invocation_context(draw.precision, cuda_capability=cuda_capability)
    zepto_infer = measure_infer_zepto(
        family,
        opts,
        seq_len=draw.seq_len,
        batch_size=draw.batch_size,
        ctx=ctx,
    )
    zepto_train = measure_train_zepto(
        family,
        opts,
        seq_len=draw.seq_len,
        batch_size=draw.batch_size,
        ctx=ctx,
    )
    batch_rep = zepto_infer.zepto_batch_representation
    if log_spec:
        logger.info(
            "first draw Zepto train spec: seq_len=%s batch=%s micro_batches=1",
            draw.seq_len,
            draw.batch_size,
        )
        logger.info(
            "first draw Zepto infer: seq_len=%s batch=%s",
            draw.seq_len,
            draw.batch_size,
        )

    model = family.build_hf_model(
        opts,
        seq_len=draw.seq_len,
        precision=draw.precision,
        device=device,
        twin_mode=config.twin_mode,
    )
    input_ids, labels = build_hf_inputs(
        family,
        opts,
        batch_size=draw.batch_size,
        seq_len=draw.seq_len,
        draw_seed=draw.draw_seed,
        device=device,
    )

    torch_infer = measure_infer_torch(family, model, input_ids)
    infer_correction = infer_cublas_vram_correction_bytes(
        global_draw_index, cuda_capability
    )
    rows: list[EvaluationRow] = [
        EvaluationRow(
            model_id=config.model_id,
            configuration_id=configuration.configuration_id,
            draw_id=draw.draw_id,
            phase="inference",
            precision=draw.precision,
            seq_len=draw.seq_len,
            batch_size=draw.batch_size,
            step=0,
            zepto_batch_representation=batch_rep,
            y_flop=zepto_infer.y_flop,
            y_vram=zepto_infer.y_vram,
            target_flop=torch_infer.target_flop,
            target_vram=torch_infer.target_vram_raw - infer_correction,
            target_vram_raw=torch_infer.target_vram_raw,
            cublas_infer_correction_bytes=infer_correction,
            y_runtime_workspace=zepto_infer.y_runtime_workspace,
            y_activations=zepto_infer.y_activations,
            peak_minus_before=torch_infer.peak_minus_before,
            alloc_before=torch_infer.alloc_before,
            target_flop_no_opt=torch_infer.target_flop_no_opt,
            y_flop_no_opt=zepto_infer.y_flop_no_opt,
        )
    ]

    model.train()
    opt = torch.optim.Adam(model.parameters())
    training_warmup_step(family, model, input_ids, labels, opt)

    for step in range(1, config.training_steps_per_draw + 1):
        torch_train = measure_train_step_torch(
            family, model, input_ids, labels, opt
        )
        rows.append(
            EvaluationRow(
                model_id=config.model_id,
                configuration_id=configuration.configuration_id,
                draw_id=draw.draw_id,
                phase="training",
                precision=draw.precision,
                seq_len=draw.seq_len,
                batch_size=draw.batch_size,
                step=step,
                zepto_batch_representation=batch_rep,
                y_flop=zepto_train.y_flop,
                y_vram=zepto_train.y_vram,
                target_flop=torch_train.target_flop,
                target_vram=torch_train.target_vram_raw,
                target_vram_raw=torch_train.target_vram_raw,
                cublas_infer_correction_bytes=0,
                y_runtime_workspace=zepto_train.y_runtime_workspace,
                y_activations=zepto_train.y_activations,
                peak_minus_before=torch_train.peak_minus_before,
                alloc_before=torch_train.alloc_before,
                target_flop_no_opt=torch_train.target_flop_no_opt,
                y_flop_no_opt=zepto_train.y_flop_no_opt,
            )
        )

    del model, input_ids, labels, opt
    torch.cuda.empty_cache()
    return rows


def run_study(config: RunConfig) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Empirical cost study requires CUDA PyTorch (Phase 1 smoke milestone)."
        )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    family = get_model_family(config.model_id)
    device = torch.device("cuda")
    cuda_capability = torch.cuda.get_device_capability(0)

    configurations = sample_configurations(
        family,
        config.sampler,
        n_configs=config.n_configs,
        master_seed=config.master_seed,
    )

    all_options: list = []
    all_links: list = []
    for cfg in configurations:
        all_options.extend(cfg.option_rows)
        all_links.extend(cfg.link_rows)
    write_dimension_tables(output_dir, all_options, all_links)

    meta = _run_meta(config)
    write_run_meta(output_dir, meta)
    write_methodology(output_dir, METHODOLOGY_TEMPLATE, _run_parameters_md(config, meta))

    if config.append_results and (output_dir / "evaluation.csv").exists():
        pass
    else:
        write_evaluation_rows(output_dir, [])

    first_draw_logged = False
    global_draw_index = 0
    for config_index, configuration in enumerate(configurations):
        draws = sample_draws(
            configuration,
            config.sampler,
            m_draws=config.m_draws_per_config,
            master_seed=config.master_seed,
            config_index=config_index,
        )
        for draw in draws:
            log_spec = config.log_first_draw_spec and not first_draw_logged
            rows = run_draw(
                config,
                family=family,
                configuration=configuration,
                draw=draw,
                device=device,
                cuda_capability=cuda_capability,
                global_draw_index=global_draw_index,
                log_spec=log_spec,
            )
            append_evaluation_rows(output_dir, rows)
            global_draw_index += 1
            if log_spec:
                first_draw_logged = True

    return output_dir
