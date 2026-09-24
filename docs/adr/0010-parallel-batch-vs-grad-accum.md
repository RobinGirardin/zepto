---
status: accepted
---
# Parallel batch vs gradient accumulation vs `micro_sum`

HuggingFace trains with a **single** forward/backward on batched activations
`(B, S, …)`. Zepto records that width on `HorizonStep.batch` and reads it from
**tensor shapes** at compose/lowering time. Trial 1 empirical work showed a
dominant VRAM error when a harness mapped HF `per_device_train_batch_size=B`
to **`micro_batches=B` with `batch=1`** instead of **`batch=B` with
`micro_batches=1`**.

## Decision

| Symbol | Zepto field | HuggingFace analogue | Meaning |
|--------|-------------|----------------------|---------|
| **B** | `HorizonStep.batch` | `per_device_train_batch_size` (single pass) | Parallel batch width of one compose/lowering: tensor leading dim before `S`. |
| **S** | `HorizonStep.seq_len` | sequence length in the batch | Tokens per sequence in that step. |
| **G** | `HorizonSpec.training(..., micro_batches=G)` | `gradient_accumulation_steps` | Count of sequential `TRAIN` (`phase="full"`) micros on `(b, S)` before optimizer. |
| **b** | `batch` on each train micro | micro-batch size per accum step | Tensor width **per** `TRAIN` step. Activations stay width `b`, not `G × b`. |

**HF single batched training step (no grad accum):**

```text
HorizonSpec.training(seq_len=S, batch=B, micro_batches=1)
→ one forward (B,S) + one backward (B,S) + optimizer
```

**HF with gradient accumulation:**

```text
HorizonSpec.training(seq_len=S, batch=b, micro_batches=G)
→ TRAIN × G on (b,S), then optimizer
Peak VRAM is max over those TRAIN micros, not the sum.
training() does not create GradAccumState.
Effective batch ≈ G × b (per device) is FLOP / optimizer frequency, not VRAM.
```

**Rejected for HF VRAM parity** (unless explicitly labeled `micro_sum`):

```text
batch=1, micro_batches=B   # B sequential (1,S) forwards — not HF (B,S) VRAM
```

## Consequences

- Parallel batch is **`HorizonStep.batch`**, never `micro_batches` substituting
  for B. `GradAccumState.bytes` equal `parameter_bytes` and do **not** scale
  with B.
- `InvocationContext` has **no** `batch` field. `estimate` reads batch from
  graph tensor shapes; `estimate_horizon` requires `inputs_fn` to emit a root
  whose leading dim matches `step.batch` (mismatch is a silent caller bug).
- Decoder models use rank-2 token ids `(B, S)` via `inputs_from_token_ids()`.
  Hidden-state modules use rank-3 `(B, S, H)` via `inputs_from_shape(H)`.
  Rank-1 `(S,)` / rank-2 `(S, H)` aliases remain valid for `B=1`.
- Model constructors take `seq_len` for mask/RoPE materialization, **not**
  batch. Peak VRAM is **max over horizon steps**, not the sum of micro peaks.
- Params and AdamW optimizer state (`state_bytes_per_parameter=8`) are
  constant in B. Ratio tests must compare the **activation/transient**
  remainder, not raw `peak_vram`.

See `docs/plans/parallel-batch-training.md` §2 and `CONTEXT.md`
("Parallel batch vs gradient accumulation").
