# Empirical cost study methodology

## Objective

Collect a small, reproducible dataset comparing Zepto analytical **FLOPs** and **peak VRAM** to CUDA PyTorch indicative measurements (`FlopCounterMode`, `max_memory_allocated`) for cost planning. This is not a benchmark leaderboard.

## Model identity

Each row uses a **family slug** (`model_id`, e.g. `apertus`). Models use **random-init** HuggingFace modules matching Zepto architecture options only — no checkpoint download. HF classes are an executable reference graph.

## Twin contract

Default **`twin_mode=scored_window`** (see `run_meta.json`). **VRAM parity studies use `scored_window`.**

| Mode | HF `max_position_embeddings` | When to use |
|------|------------------------------|-------------|
| **`scored_window`** (default) | `max(seq_len, 32)` | VRAM parity vs the scored window (same as smoke). HF static buffers (RoPE/mask) are sized to **S**, not unused context. Transformers may warn because checkpoint `original_max_position_embeddings` (8192) is not **<** MPE; that is acceptable. llama3 `rope_parameters` are **not** shrunk. |
| **`apertus_parity`** | `max(seq_len, 8193)` | llama3 config validates without warning (`original_max_position_embeddings` must stay **<** MPE). HF still allocates ~8k positions; not a scored-window VRAM twin. |
| **`transformers_defaults`** | omitted (HF default, typically 65536) | Trial 1 full-context experiment. |

- **HF (Apertus):** `attn_implementation="eager"`, `use_cache=False`, no gradient checkpointing, fused cross-entropy on training forward. Checkpoint **llama3** `rope_parameters` (θ=12M, YaRN factor 8, `original_max_position_embeddings=8192`) are kept — do not set `rope_scaling=None`.
- **Zepto (Apertus):** same llama3 RoPE via `apertus_rope` on `RoPEMaterialize`; `attention_backend="eager"`, `requested_capabilities` includes `fused`, `sdpa`, `gqa`. Zepto materializes RoPE/mask at the scored **S** in all modes — do not inflate Zepto caches to chase HF MPE.

## Phases

**Inference:** `model.eval()`, `torch.no_grad()`, single forward `model(input_ids=..., use_cache=False)`.

**Training:** one scored window = forward (with loss) + backward + `optimizer.step()`; Adam on HF; Zepto `AdamW` at the training horizon boundary.

## Ground truth windows

- **FLOPs:** `FlopCounterMode` wraps exactly the scored ops (training: forward + backward + `opt.step()` inside the counter).
- **VRAM:** reset peak stats → synchronize → record `alloc_before` (`memory_allocated()`) → run scored ops → synchronize → read `max_memory_allocated()` as `target_vram_raw`.

### Infer `target_vram` (cuBLAS handle correction)

CUDA peak VRAM often includes cuBLAS workspace tied to GEMM handles. Zepto models **one** handle on infer; after a prior draw has run training, measured infer peak can reflect **two** handles’ workspace (~8.5 MiB on pre-SM90 GPUs).

- `target_vram_raw`: unadjusted `max_memory_allocated()` peak for the scored infer window.
- `cublas_infer_correction_bytes`: subtracted on **inference** rows only when `global_draw_index > 0` (0 on the first draw in a run). Value = `cublas_workspace_bytes_per_handle(compute_capability)` from runtime policy.
- `target_vram`: comparable ground truth — `target_vram_raw - cublas_infer_correction_bytes` on infer; training rows use raw peak (`correction = 0`).

Draw order: configurations and draws are processed sequentially; index increments once per draw (not per CSV row).

### FLOP columns (training vs optimizer)

- `target_flop` / `y_flop`: full training cycle including `optimizer.step()` / Zepto `AdamW` horizon boundary.
- `target_flop_no_opt` / `y_flop_no_opt`: forward + backward only (separate CUDA counter window; Zepto horizon with `optimizer=None`). Inference rows duplicate infer FLOPs in both pairs.

### Zepto VRAM diagnostics (`evaluation.csv`)

- `y_runtime_workspace`: infer uses `runtime_workspace + workspace` on the single `estimate`. Train uses that sum on the **peak-relevant** step (`max` over `report.per_step`); merged horizon `breakdown.runtime_workspace` is also that max cuBLAS pool, not 1+2+0 handles.
- `y_activations`: Zepto `memory.breakdown.activations`.
- `peak_minus_before`: `target_vram_raw - alloc_before` at infer/train window start (HF probe).
- `alloc_before`: HF `memory_allocated()` immediately before scored ops after peak reset.

## Training warmup (HF only, before step 1)

One **unscored** train step: forward + backward + `optimizer.step()` outside `FlopCounterMode`, after inference and before scored steps `1..K`.

## Zepto measurement

- **Inference, batch B:** compose `(B, S)` token ids; one `estimate` (single invocation). Does **not** use `HorizonSpec.repeat`.
- **Training, batch B:** `HorizonSpec.training(seq_len=S, batch=B, micro_batches=1, optimizer=AdamW)` on `ApertusForCausalLM`.
- `y_flop` ← forward FLOPs (infer) or `report.total_flops` (train); `y_vram` ← `report.memory.peak_live_bytes` (infer) or `report.peak_vram` (train).
- Column `zepto_batch_representation=parallel_batch`.

## Batch semantics (`parallel_batch`)

Zepto and HF both run one parallel `(B, S)` pass. Parallel batch is `HorizonStep.batch`; gradient accumulation is `micro_batches`. Trial 1's `micro_sum` mode (`batch=1`, `micro_batches=B`, or infer `HorizonSpec.repeat(B, batch=1)`) is a **rejected** sequential-forwards mapping — it is not the measurement contract.

## Precision mapping

| precision | HF module dtype | Zepto context |
|-----------|-----------------|---------------|
| fp32 | float32 | `reference_invocation(..., default_dtype=FP32)` |
| fp16 | float16 | uniform fp16 (`default_dtype=FP16`, 2-byte params and grads) |
| mixed | float16 params | `PrecisionPolicy.from_byte_sizes(param_bytes=2, grad_bytes=4)` + `optim_prec=4` |

Base flags: `hardware=cuda`, device compute capability, eager attention, fused/sdpa/gqa capabilities.

## Step column semantics

Option A: sequential training on one model+optimizer per draw; `target_*` vary by step; Zepto `y_*` are duplicated per step (steady-state per-cycle analytical estimate).

## Known limitations

`FlopCounterMode` may undercount Adam elementwise ops vs Zepto policy FLOPs; cuBLAS workspace; no prefill/decode split; no KV cache.
