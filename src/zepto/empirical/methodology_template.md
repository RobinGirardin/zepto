# Empirical cost study methodology

## Objective

Collect a small, reproducible dataset comparing Zepto analytical **FLOPs** and **peak VRAM** to CUDA PyTorch indicative measurements (`FlopCounterMode`, `max_memory_allocated`) for cost planning. This is not a benchmark leaderboard.

## Model identity

Each row uses a **family slug** (`model_id`, e.g. `apertus`). Models use **random-init** HuggingFace modules matching Zepto architecture options only — no checkpoint download. HF classes are an executable reference graph.

## Twin contract

- HF: `attn_implementation="eager"`, `use_cache=False`, no gradient checkpointing, fused cross-entropy on training forward. **Apertus:** stock `ApertusConfig` context and RoPE (transformers defaults for `max_position_embeddings` and `rope_parameters` / YaRN) — not a shortened context window.
- Zepto: `attention_backend="eager"`, `requested_capabilities` includes `fused`, `sdpa`, `gqa`.

## Phases

**Inference:** `model.eval()`, `torch.no_grad()`, single forward `model(input_ids=..., use_cache=False)`.

**Training:** one scored window = forward (with loss) + backward + `optimizer.step()`; Adam on HF; Zepto `AdamW` at the training horizon boundary.

## Ground truth windows

- **FLOPs:** `FlopCounterMode` wraps exactly the scored ops (training: forward + backward + `opt.step()` inside the counter).
- **VRAM:** reset peak stats → synchronize → run scored ops → synchronize → read `max_memory_allocated()`.

## Training warmup (HF only, before step 1)

One **unscored** train step: forward + backward + `optimizer.step()` outside `FlopCounterMode`, after inference and before scored steps `1..K`.

## Zepto measurement

- **Inference, batch B:** `HorizonSpec.repeat(invocations=B, seq_len=S, batch=1)` (B=1 uses single `estimate` for smoke parity). FLOPs summed across micro-forwards; peak VRAM from horizon peak.
- **Training, batch B:** `HorizonSpec.training(seq_len=S, micro_batches=B, batch=1, optimizer=AdamW)` on the family training module (e.g. `ApertusForCausalLM`).
- `y_flop` ← forward FLOPs (infer) or `report.total_flops` (train); `y_vram` ← `report.memory.peak_live_bytes` (infer) or `report.peak_vram` (train).

## Batch semantics (`micro_sum`)

Zepto sums `B` micro-forwards at batch size 1; HF runs one parallel `(B, S)` forward. FLOP comparability is an **assumption**; VRAM may diverge. Column `zepto_batch_representation=micro_sum` records this.

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
