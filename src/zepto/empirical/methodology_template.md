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
- **Zepto (Apertus):** same llama3 RoPE via `apertus_rope` on `RoPEMaterialize`; `attention_backend="eager"`, `requested_capabilities={"fused"}`, region pins to `region/{rmsnorm,xielu,softmax,linear_ce}/reference` and `region/swiglu/decomposed` (eager ATen leaves). Do not request `sdpa`/`gqa`/`flash` — those profiles block the ATen leaves and/or fuse attention. Zepto materializes RoPE/mask at the scored **S** in all modes — do not inflate Zepto caches to chase HF MPE.
- **HF (Granite):** random-init `GraniteForCausalLM`, `hidden_act=silu`, default RoPE θ=50M, `tie_word_embeddings=False`, identity residual/embedding/attention/logit scales. No `apertus_parity` mode and no llama3 `rope_parameters`.
- **Zepto (Granite):** default RoPE via `granite_rope`; same eager ATen invocation context as Apertus, including the `region/swiglu/decomposed` pin (HF `F.silu` + mul + three GEMMs). Do not enable Liger SwiGLU (`fused_post_gemm` / `fused_gate_up`).
- **HF (Qwen3.8):** random-init text-only `Qwen3_5ForCausalLM` / `Qwen3_5TextConfig`, hybrid `layer_types` (3× `linear_attention` + 1× `full_attention` per cycle), GDN `linear_*` heads, gated GQA `head_dim`, mRoPE `rotary_dim=64` (`mrope_section=(8, 12, 12)` when the config accepts it). Never instantiate `*ForConditionalGeneration`. No `apertus_parity` mode. Do not rely on `fla` / `causal_conv1d` on the parity path.
- **Zepto (Qwen3.8):** `Qwen38` / `Qwen38ForCausalLM` with `include_vision=False` and `include_mtp=False`. Residual width is free (not `q * d`). Same invocation context as Granite (`requested_capabilities={"fused","sdpa","gqa"}`) so `region/linear_ce` can fuse; do not enable FlashAttention or Liger SwiGLU.

## Phases

**Inference:** `model.eval()`, `torch.no_grad()`, single forward `model(input_ids=..., use_cache=False)`.

**Training:** one scored window = forward (with loss) + backward + `optimizer.step()`; `torch.optim.AdamW` on HF (CUDA default `foreach=True`); Zepto `AdamWPolicy(foreach=True)` bills one parameter-sized update workspace at the training horizon boundary, not as persistent moments. The Apertus HF twin's Adam moments follow `param_prec`, so the harness passes `optim_prec=param_prec`.

## Ground truth windows

- **FLOPs:** `FlopCounterMode` wraps exactly the scored ops (training: forward + backward + `opt.step()` inside the counter).
- **VRAM:** reset peak stats → synchronize → record `alloc_before` (`memory_allocated()`) → run scored ops → synchronize → read `max_memory_allocated()` as `target_vram_raw`.

### Infer `target_vram` (cuBLAS handle correction)

CUDA peak VRAM includes cuBLAS workspace tied to GEMM handles. A process-wide dummy forward and backward is run **once** before any scored row so two handles are already live. Zepto bills one handle on inference and two on training.

- `target_vram_raw`: unadjusted `max_memory_allocated()` peak for the scored infer window.
- `cublas_infer_correction_bytes`: one handle (`cublas_workspace_bytes_per_handle(compute_capability)`) subtracted on **every** inference row after that warmup, including the first subject and its second precision. Training rows use correction 0.
- `target_vram`: `target_vram_raw - cublas_infer_correction_bytes` on infer; training rows use the raw peak.

`empty_cache()` and deleting the module do not free the handle pool. Remaining leftover after the one-handle subtract is treated as constant.

### Training windows

Training VRAM and FLOPs are scored in **separate** windows. Both wrap `optimizer.step()`. `FlopCounterMode` is not active during the VRAM window.

### FLOP columns (training vs optimizer)

- `target_flop` / `y_flop`: full training cycle including `optimizer.step()` / Zepto `AdamW` horizon boundary.
- `target_flop_no_opt` / `y_flop_no_opt`: forward + backward only (separate CUDA counter window; Zepto horizon with `optimizer=None`). Inference rows duplicate infer FLOPs in both pairs.
- `y_flop` = Zepto theoretical FLOP (fused regions + extras + Adam on train).
- `y_flop_fcm` = same graphs, allowlisted operations only (`matmul` / `linear_matmul` / `conv2d` / `conv3d`), Adam excluded. Hypothesis column vs `target_flop`.
- `y_flop_fcm_no_opt` = FCM-comparable total of the `optimizer=None` horizon (infer: same as `y_flop_fcm`).
- `FlopCounterMode` is a GEMM/conv registry, not a complete cost model.

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
- `y_flop` ← named Zepto totals (`zepto_forward_flops` infer / `zepto_total_flops` train); `y_flop_fcm` ← named FCM fields (`fcm_forward_flops` / `fcm_total_flops`). Do not flip `flop_policy`. `y_vram` ← `report.memory.peak_live_bytes` (infer) or `report.peak_vram` (train).
- Column `zepto_batch_representation=parallel_batch`.

## Batch semantics (`parallel_batch`)

Zepto and HF both run one parallel `(B, S)` pass. Parallel batch is `HorizonStep.batch`; library `micro_batches=G` is sequential `TRAIN` micros on `(b, S)` (this study still uses `G=1`). Trial 1's `micro_sum` mode (`batch=1`, `micro_batches=B`, or infer `HorizonSpec.repeat(B, batch=1)`) is a **rejected** sequential-forwards mapping — it is not the measurement contract.

## Precision mapping

| precision | HF module dtype | Zepto context | Comparable? |
|-----------|-----------------|---------------|-------------|
| fp32 | float32 | `default_dtype=FP32` | yes |
| fp16 | float16 | uniform fp16 (`from_byte_sizes(2, 2)`) | yes, if no extra master weights |

Empirical studies do **not** sample or measure `"mixed"`. Zepto’s role-split policy (`param_bytes=2, grad_bytes=4`) and HF Trainer AMP (`--fp16` / GradScaler) are **out of scope** until a dedicated follow-up.

Base flags: `hardware=cuda`, device compute capability, eager attention, `requested_capabilities={"fused"}`, ATen reference region pins.

## Step column semantics

Option A: sequential training on one model+optimizer per draw; `target_*` vary by step; Zepto `y_*` are duplicated per step (steady-state per-cycle analytical estimate).

## Known limitations

`FlopCounterMode` may undercount AdamW elementwise ops vs Zepto policy FLOPs; cuBLAS workspace; no prefill/decode split; no KV cache.
