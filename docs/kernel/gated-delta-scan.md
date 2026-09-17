# Zepto kernel research: Gated Delta Scan

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Standalone gated delta-rule recurrent scan leaf (`GatedDeltaScan`); prefill (\(S>1\)) and decode (\(S=1\)) with optional recurrent state port; training + inference; CUDA/ROCm primary (FLA Triton); identity reference is S-unrolled at compose time
**Context:** Core recurrence in Qwen3-Next / Qwen3.5 **Gated DeltaNet** mixers after conv + L2-normalized Q/K. Zepto module [`GatedDeltaScan`](../src/zepto/modules/gated_delta_scan.py) is registered; fused region `region/gated_delta_scan` is **implemented** under [`gated_delta_scan/`](../src/zepto/analysis/lowering/implementations/regions/gated_delta_scan/). Parent `GatedDeltaNet` supplies \(\beta=\sigma(b)\), log-decay \(g_t\le 0\), and head-expanded Q/K/V — **not** billed in this leaf. Default cost leaf should be **kernel-accurate** fused scan (FLA `chunk_gated_delta_rule` prefill / `fused_recurrent_gated_delta_rule` decode), not the sum of per-token graph overhead.

---

## Section 0: Mathematical definition

**Gated delta-rule linear attention** maintains a matrix-valued state \(\mathbf H_t \in \mathbb{R}^{d_k \times d_v}\) per value head and applies a gated, rank-one delta update each timestep. Zepto uses the **staged** form (matches [`gated_delta_scan.py`](../src/zepto/modules/gated_delta_scan.py) and [step5 plan §Phase 4](../docs/plans/step5-stateful-mixers.md)):

\[
\begin{aligned}
\widetilde{\mathbf H}_t &= \alpha_t \mathbf H_{t-1},
&&\alpha_t = \exp(g_t) \in (0,1],\; g_t \le 0,\\
\widehat{\mathbf v}_t &= \mathbf k_t^\top \widetilde{\mathbf H}_t,\\
\boldsymbol\delta_t &= \beta_t \bigl(\mathbf v_t - \widehat{\mathbf v}_t\bigr),
&&\beta_t \in (0,1),\\
\mathbf H_t &= \widetilde{\mathbf H}_t + \mathbf k_t \boldsymbol\delta_t^\top,\\
\mathbf o_t &= \mathbf q_t^\top \mathbf H_t.
\end{aligned}
\]

Equivalently (Atto / paper form),
\(\mathbf H_t = \alpha_t(\mathbf I - \beta_t \mathbf k_t \mathbf k_t^\top)\mathbf H_{t-1} + \beta_t \mathbf k_t \mathbf v_t^\top\)
([Atto gated DeltaNet](https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md); [Gated DeltaNet, Yang et al., ICLR 2025](https://arxiv.org/abs/2412.06464)).

**I/O shapes (Zepto module, batch \(B=1\) in reference graph):**

| Tensor | Shape | Role |
|--------|-------|------|
| `query` | \((S, h_v, d_k)\) | L2-normalized, scaled Q (after parent `RepeatKV`) |
| `key` | \((S, h_v, d_k)\) | L2-normalized K |
| `value` | \((S, h_v, d_v)\) | V |
| `beta` | \((S, h_v)\) | Post-sigmoid gate from parent |
| `log_decay` | \((S, h_v)\) | \(g_t\le 0\) in fp32 before `Exp` |
| `scan_state_in` | \((h_v, d_k, d_v)\) or `None` | Initial state; fp32 accumulation |
| `output` | \((S, h_v, d_v)\) | Scan outputs (cast to activation dtype) |
| `scan_state_out` | \((h_v, d_k, d_v)\) | Final state for horizon / decode |

**Numerics policies:**

- Recurrent state \(\mathbf H\) accumulates in **fp32** (`RMSNORM_COMPUTE_DTYPE`); outputs cast back to bf16/fp16 ([`gated_delta_scan.py`](../src/zepto/modules/gated_delta_scan.py)).
- \(\alpha_t=\exp(g_t)\) with \(g_t\le 0\) guarantees non-expanding decay (Qwen uses \(g_t = -\exp(A_{\log})\,\mathrm{softplus}(a_t + \mathrm{dt\_bias})\) in parent — **outside** this region).
- Q/K L2 normalization is **upstream** (`region/l2_normalize` or FLA `use_qk_l2norm_in_kernel`) — do not double-count when fused into scan.
- Production chunk kernels may fuse gate cumsum, beta sigmoid, and optional in-kernel L2 ([FLA `chunk_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)).

**Structural vs eager:**

- **Identity lowering:** Python `for t in range(S)` emits per-step `Exp`, `Multiply`, `MatMul`, `Subtract`, `Add` at compose time (graph size \(\Theta(S)\), arithmetic \(\Theta(S h_v d_k d_v)\)).
- **Fused leaf:** Single Triton/CUDA kernel over \(S\) (chunk-parallel prefill) or one recurrent step (decode); same asymptotic FLOPs, elides per-step HBM temps and `Exp`/`MatMul` boundary traffic.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | FLA [`chunk_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py) (prefill/train), [`fused_recurrent_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/) (decode); optional FlashQLA backend on SM90/SM100 ([FLA PR #998](https://github.com/fla-org/flash-linear-attention/pull/998)); HF Qwen3-Next modular path delegates to FLA when installed ([`modular_qwen3_next.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py)) |
| **Identity lowering** | Unfused semantic chain | Zepto `GatedDeltaScan`: S-unrolled loop of rank-1 update + query read ([`gated_delta_scan.py`](../src/zepto/modules/gated_delta_scan.py)); horizon `RecurrentStateConfig` for `scan_state_out` ([`step5` §7.4](../docs/plans/step5-stateful-mixers.md)) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/gated_delta_scan` (stub) — closed-form **\(f(S,h_v,d_k,d_v)\)** forward/backward; `region/gated_delta_scan/decode` when `S=1` + state port; elides per-timestep temps |

**Default estimates use the fused region leaf when `requested_capabilities` includes `fused`, not the identity graph node count.**

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **FLA chunk GDR** | [`chunk_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py) | Yes (full scan) | **C** | CUDA / ROCm (Triton) | Prefill + training; optional `output_final_state` |
| **FLA fused recurrent** | `fused_recurrent_gated_delta_rule` | Yes (single step) | **C** | CUDA / ROCm | Decode / streaming |
| **FLA GatedDeltaNet layer** | [`fla/layers/gated_deltanet.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py) | Yes (scan ± conv/gate/norm flags) | **B/C** | CUDA / ROCm | `mode=chunk` vs `fused_recurrent` |
| **FlashQLA GDN** | Optional `flash_qla` via FLA backend | Yes (chunk prefill) | **C** | SM90/SM100 | Prefill fwd+bwd; strict shape/config verifier |
| **HF eager / modular Qwen3** | Transformers `Qwen3NextGatedDeltaNet` | Partial (Python loop or FLA) | C | CUDA + optional FLA | Both |
| **Zepto module (identity)** | `GatedDeltaScan` | No (S-unrolled) | C | Any | Both; emits recurrent state port |
| **Zepto region (stub)** | `region/gated_delta_scan` in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py) | Cost leaf TBD | C | `fused` gate | `status: to_implement` |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **C** — full recurrent delta-rule scan (analogous to `region/gqa/*` for softmax attention: one kernel owns state update + output read). Sub-leaf **A** only when modeling isolated primitives (not recommended for default billing).

**Mutually exclusive (same scan slot on matched provenance):**

- `region/gated_delta_scan/*` vs identity `GatedDeltaScan` S-unrolled graph — fused region wins when pattern + `fused` capability match.
- `region/gated_delta_scan` (scan only) vs **`region/gated_delta_net`** block envelope — block region subsumes child scan FLOPs/memory when architect enables Tier-B composition ([`step5` §8.1](../docs/plans/step5-stateful-mixers.md)).
- Standalone `region/l2_normalize` on Q/K vs FLA `use_qk_l2norm_in_kernel=True` inside scan — **must not double-count** (composition rule TBD in architect; default Qwen Zepto stack uses explicit L2 before scan).

**Composable:**

- **GatedDeltaNet:** `Linear → DepthwiseCausalConv1d → L2Normalize (Q,K) → GatedDeltaScan → GatedRMSNorm → Linear` — scan bills only recurrence + read; projections/conv/norm are separate regions.
- **Horizon decode:** prefill uses full \(S\); decode steps use `region/gated_delta_scan/decode` with `scan_state_in` **PERSIST** (related gap **G3** recurrent state bytes across steps).

**Execution constraints:**

- GVA: \(h_v\) value heads may exceed Q/K head count after `RepeatKV` in parent; scan requires equal head counts on Q/K/V at boundary ([`gated_delta_net.py`](../src/zepto/modules/gated_delta_net.py)).
- Chunk kernels require sufficient sequence length for parallel chunks; decode uses recurrent kernel (\(S=1\)).
- `requires_grad=False` → backward FLOPs = 0; inference decode persists **one** final state \((h_v, d_k, d_v)\), not all \(S\) states.
- Batch dimension: reference graph uses \(B=1\); scale FLOPs/bytes by \(B\) for \((B,S,\ldots)\) layouts.

---

## Section 4: Forward FLOPs — step-by-step derivation

Notation: \(S\) sequence length, \(h\) value heads (\(h_v\)), \(d_k\) key dim, \(d_v\) value dim, one multiply-add = **2 FLOPs**.

### 4.1 Identity lowering (Zepto `GatedDeltaScan` per timestep \(t\), per head)

| Step | Primitive | Tile / shape | FLOPs (per head) | Subtotal |
|------|-----------|--------------|------------------|----------|
| 1 | `Exp` | scalar \(g_t\) | 1 | 1 |
| 2 | `Multiply` | \(\alpha_t \odot \mathbf H_{t-1}\) | \(d_k d_v\) | \(d_k d_v\) |
| 3 | `MatMul` | \(\mathbf k_t^\top \widetilde{\mathbf H}_t\) | \(2 d_k d_v\) | \(2 d_k d_v\) |
| 4 | `Subtract` | \(\mathbf v_t - \widehat{\mathbf v}_t\) | \(d_v\) | \(d_v\) |
| 5 | `Multiply` | \(\beta_t \odot (\cdots)\) | \(d_v\) | \(d_v\) |
| 6 | `MatMul` | outer \(\mathbf k_t \boldsymbol\delta_t^\top\) | \(2 d_k d_v\) | \(2 d_k d_v\) |
| 7 | `Add` | \(\widetilde{\mathbf H}_t + \text{outer}\) | \(d_k d_v\) | \(d_k d_v\) |
| 8 | `MatMul` | \(\mathbf q_t^\top \mathbf H_t\) | \(2 d_k d_v\) | \(2 d_k d_v\) |
| | **Per head per step** | | | \(8 d_k d_v + 2 d_v + 1\) |

Over \(S\) steps and \(h\) heads:

\[
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{identity}} = S h \bigl(8 d_k d_v + 2 d_v + 1\bigr).
\]

When \(d_k d_v \gg 1\): \(\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{identity}} \approx 8 S h d_k d_v\).

### 4.2 Fused region leaf (kernel-accurate)

Chunk/recurrent FLA kernels implement the same recurrence with identical asymptotic MAC count; fused path bills the **same** closed form as §4.1 (dominant term \(8 S h d_k d_v\)), not extra compose-time `Reshape`/`Split` nodes.

**Decode (\(S=1\)):**

\[
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{decode}} = h \bigl(8 d_k d_v + 2 d_v + 1\bigr).
\]

### 4.3 Paper-comparable (Atto staged recurrence + read)

Atto aggregates an equivalent rank-one + gated update into **\(7 d_k d_v\)** recurrence MACs plus **\(2 d_k d_v\)** output read per head per step (different grouping of \(\alpha\) scaling vs Zepto’s explicit `Multiply` on full state), then reports **\(9 S h d_k d_v\)** when \(d_k=d_v\) and \(h d_v = d\) ([Atto §Computation](https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md)). Difference vs Zepto identity: **\(\approx 8/9\)** on the dominant term — label **paper-comparable** when citing Atto; **default Zepto leaf = §4.1**.

**Closed form (kernel-accurate default):**

\[
\boxed{\mathrm{FLOPs}_{\mathrm{fwd}} = S h \bigl(8 d_k d_v + 2 d_v + 1\bigr)}
\]

**Arithmetic intensity (numeric example — Qwen3.5-style DeltaNet layer, scan leaf only):**

Take \(S=8192\), \(h=48\), \(d_k=d_v=128\), bf16 \(e=2\).

- \(\mathrm{FLOPs}_{\mathrm{fwd}} \approx 8192 \times 48 \times 8 \times 128 \times 128 \approx 5.2 \times 10^{10}\).
- Minimum HBM moved if inputs/outputs each touched once (no fusion):  
  \(|Q|+|K|+|V| = 3 S h d_k e\), \(|\beta|+|g| = 2 S h e\), \(|O| = S h d_v e\)  
  \(\Rightarrow \approx (3 d_k + 2 + d_v) S h e \approx 3.1 \times 10^7\) bytes \(\approx 29\) MiB.
- Intensity \(\approx 5.2 \times 10^{10} / 3.1 \times 10^7 \approx 1.7 \times 10^3\) FLOP/byte (compute-bound at HBM bandwidth \(\sim 2\) TB/s would need \(\mathcal{O}(10^3)\)+; scan is moderately compute-heavy vs memory for this shape).

**Apertus-8B note:** Default Apertus stack is GQA attention, not DeltaNet. For cross-model checklist, use the Qwen-style trace above; Apertus symbols \((S,d,h,d_h)\) apply only when a DeltaNet layer is substituted with the same \(S=8192\), \(d_h=128\).

---

## Section 5: Backward FLOPs — step-by-step derivation

Training requires backward-through-time (BPTT) across \(S\) steps. Dominant forward ops (state decay, two head-state matmuls, outer product, output read) each admit reverse matmuls/outer products of similar order.

### 5.1 Saved vs recomputed

| Policy | Saved tensors | Backward effect |
|--------|---------------|-----------------|
| **Identity / generic BPTT** | All \(\mathbf H_t\) for \(t=1..S\) | Standard reverse recurrence; memory \(\Theta(S h d_k d_v)\) |
| **Fused FLA chunk bwd** | Kernel-dependent; may checkpoint chunks | Same asymptotic FLOPs; lower peak via chunk recomputation (implementation-specific) |
| **Inference** | Final state only | `requires_grad=False` → **0** backward FLOPs |

### 5.2 Backward step table (per head, per timestep, kernel-accurate order-of-magnitude)

| Step | Reverse of | FLOPs (per head) |
|------|------------|------------------|
| 1 | Output read \(\mathbf o_t = \mathbf q_t^\top \mathbf H_t\) | \(2 d_k d_v\) |
| 2 | State add + outer product | \(3 d_k d_v\) |
| 3 | Delta / beta / subtract branch | \(2 d_v + 2 d_k d_v\) |
| 4 | Key-state prediction matmul | \(2 d_k d_v\) |
| 5 | Decay multiply + \(\exp(g)\) chain | \(2 d_k d_v\) |
| | **Total (order)** | \(\approx 10 d_k d_v + 2 d_v\) |

Atto reports **\(10 S h d_k d_v\)** recurrence backward ([Atto §Backward](https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md)) — **paper-comparable** grouping.

**Closed form (Zepto default leaf, matches ~2× dominant forward MACs on recurrence + read):**

\[
\boxed{\mathrm{FLOPs}_{\mathrm{bwd}} \approx S h \bigl(16 d_k d_v + 4 d_v + c\bigr)}
\]

with small \(c\) from scalar gates (\(c \ll d_k d_v\)). Equivalently \(\mathrm{FLOPs}_{\mathrm{bwd}} \approx 2 \times \mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{dominant}}\) when \(d_k d_v \gg 1\).

**Decode (\(S=1\)):** replace \(S\) with 1 in the boxed formula.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

Assume fp32 state accumulation (\(e_s=4\)) for \(\mathbf H\); activations \(e=2\) (bf16) unless noted. Scan leaf only — Q/K/V/\(\beta\)/\(g\) treated as **inputs**.

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|--------------------|------------------|
| **Identity lowering** | Per-\(t\) temps: `decayed`, `prediction`, `delta`, `outer`, `out_t` each \(\mathcal{O}(h d_k d_v)\) or \(\mathcal{O}(h d_v)\) | \(\mathcal{O}(S \cdot h d_k d_v)\) if all materialized | N/A (worst case) |
| **Fused chunk scan** | Output \(O\), optional workspace; state updated in-place / registers | \(\mathcal{O}(h d_k d_v + S h d_v)\) | Per-step temps |
| **Decode recurrent** | Single-step workspace + state | \(\mathcal{O}(h d_k d_v)\) | Full-sequence temps |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto `SAVE` policy |
|------|---------------|-------|-------|---------------------|
| **BPTT default** | \(\mathbf H_t\) all timesteps | \(S \times h \times d_k \times d_v\) | \(4 \cdot S h d_k d_v\) (fp32 state) | `SAVE(state_t)` for \(t=1..S\) unless kernel checkpoints |
| **Inputs** | Q, K, V, \(\beta\), \(g\) | as §0 | \(\mathcal{O}(S h (d_k + d_v))\) | Parent regions own unless scan saves for fusion |
| **Inference decode** | Final \(\mathbf H_S\) only | \(h \times d_k \times d_v\) | \(4 h d_k d_v\) | `PERSIST` on horizon recurrent port (**G3**) |

**Qwen3.5-scale example:** \(S=8192\), \(h=48\), \(d_k=d_v=128\) → saved states \(\approx 8192 \times 48 \times 128^2 \times 4 \approx 2.0\) GiB per layer (dominant training memory for scan leaf).

### 6.3 Resource event chains

**Identity lowering (unfused):**

```
ALLOCATE(state_0) → for t in 1..S:
  ALLOCATE(decayed_t) → ALLOCATE(pred_t) → ALLOCATE(delta_t) → ALLOCATE(outer_t)
  → SAVE(state_t) [training] → ALLOCATE(out_t)
→ RELEASE(decayed_t...) [if not saved]
→ PERSIST(scan_state_out)
```

**Fused region leaf (prefill):**

```
ALLOCATE(output[S,h,d_v]) → WORKSPACE(chunk) → SAVE(state_checkpoint policy)
→ PERSIST(scan_state_out) → RELEASE(workspace)
```

**Fused decode:**

```
LOAD(scan_state_in) → WORKSPACE(step) → ALLOCATE(output[1,h,d_v]) → PERSIST(scan_state_out)
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Zepto identity | `GatedDeltaScan` | No | C | \(8 S h d_k d_v\) | \(\approx 16 S h d_k d_v\) | Materialize per-step temps (worst case) | (lowering graph) |
| FLA chunk | `chunk_gated_delta_rule` | Yes | C | \(8 S h d_k d_v\) | \(\approx 16 S h d_k d_v\) | Chunk workspace; optional final state | `region/gated_delta_scan` |
| FLA recurrent | `fused_recurrent_gated_delta_rule` | Yes | C | \(8 h d_k d_v\) per step | \(\approx 16 h d_k d_v\) | Single state | `region/gated_delta_scan/decode` |
| Atto formula | Atto doc | N/A | C | \(9 S h d_k d_v\) (paper group) | \(10 S h d_k d_v\) | All states saved | cross-check only |
| FlashQLA | `flash_qla` | Yes | C | Same asymptotic | Fused bwd | SM90+ optimized | variant gate |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: gated_delta_scan
recommended_region_ids:
  - id: region/gated_delta_scan
    variant: fla_chunk
    hardware_gate: cuda
    fusion_boundary: C
    status: to_implement
  - id: region/gated_delta_scan/decode
    variant: fla_recurrent
    hardware_gate: cuda
    fusion_boundary: C
    status: to_implement
  - id: region/gated_delta_scan
    variant: reference
    hardware_gate: any
    fusion_boundary: C
    status: to_implement
pattern_rule:
  op_families: [Exp, Multiply, MatMul, Subtract, Add, Cast, Reshape, Split, Concat]
recipe:
  forward_flops: "S * h * (8 * d_k * d_v + 2 * d_v + 1)"
  backward_flops: "S * h * (16 * d_k * d_v + 4 * d_v + 4)"
  forward_flops_per_element: null
  backward_flops_per_element: null
  materialize_per_step_temps: false
  save_state_all_timesteps: true
  save_scan_state_out: true
  elided_temps: [decayed_t, prediction_t, delta_t, outer_t]
  saved_backward:
    - "state_t: [S, h, d_k, d_v] fp32 when training and not checkpointed"
    - "scan_state_out: [h, d_k, d_v] for horizon persist"
  resource_events_forward:
    - "ALLOCATE(output)"
    - "WORKSPACE(chunk_scan)"
    - "PERSIST(scan_state_out)"
  resource_events_backward:
    - "LOAD(saved state_t or recompute from checkpoint)"
    - "SAVE(grad_state_t) optional"
  numerics_tags: [fp32_state_accum, log_decay_input, structural_s_unroll_identity]
capabilities: [fused, recurrent_state_port]
priority: 3
variants:
  prefill:
    trigger: "seq_len > 1 or phase == prefill"
    forward_flops: "S * h * (8 * d_k * d_v + 2 * d_v + 1)"
  decode:
    trigger: "seq_len == 1 and scan_state_in present"
    forward_flops: "h * (8 * d_k * d_v + 2 * d_v + 1)"
composition:
  parent_module: GatedDeltaNet
  do_not_double_bill:
    - region: region/l2_normalize
      when: "use_qk_l2norm_in_kernel on fused FLA path"
  subsumed_by:
    region: region/gated_delta_net
    when: "block-level region enabled"
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Yang et al., Gated Delta Networks (ICLR 2025): https://arxiv.org/abs/2412.06464
- Yang et al., Parallelizing Linear Transformers with Delta Rule: https://arxiv.org/abs/2406.06484
- Schlag et al., Linear Transformers Are Secretly Fast Weight Programmers: https://arxiv.org/abs/2006.16216
- Atto cost framework — Gated DeltaNet: https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md
- FLA `chunk_gated_delta_rule`: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
- FLA `GatedDeltaNet` layer: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
- FLA FlashQLA backend: https://github.com/fla-org/flash-linear-attention/pull/998
- Zepto module: https://github.com/RobinGirardin/zepto/blob/main/src/zepto/modules/gated_delta_scan.py (local [`gated_delta_scan.py`](../src/zepto/modules/gated_delta_scan.py))
- Zepto plan step5 stateful mixers: [`docs/plans/step5-stateful-mixers.md`](../docs/plans/step5-stateful-mixers.md)
- Qwen3 Next modular (HF): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| P1 | `region/gated_delta_scan` | §8 | C | Stub in `mixer_stub.py`; blocks accurate Qwen3.5 DeltaNet layer costing |
| P1 | `region/gated_delta_scan/decode` | §8 variants | C | Horizon decode FLOPs/state differ from prefill |
| P2 | Composition with `region/l2_normalize` | §3 | A/C | Avoid double billing when FLA fuses Q/K norm into scan |
| P3 | `region/gated_delta_net` | step5 §8.1 | B/C | Block region should subsume child scan when enabled |

**Known gaps flagged:** **G3** (recurrent state persist across decode steps); architect must resolve L2-in-scan mutual exclusion with default explicit `L2Normalize` stack.
