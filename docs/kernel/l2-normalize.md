# Zepto kernel research: L2-normalize (last-dim ℓ₂)

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Standalone last-axis L2 normalization leaf (boundary A); DeltaNet / GatedDeltaNet Q/K head normalization; prefill primary; training + inference; CUDA/ROCm via FLA Triton; identity Zepto module for reference
**Context:** Qwen3.5-style gated DeltaNet (\(S{=}8192\), per-head \(d_h{=}128\), \(h_k{=}8\) key heads, bf16 \(e{=}2\)); kernel-accurate Zepto leaf (\(3n\) forward / \(4n\) backward with saved `rstd`)

---

## Section 0: Mathematical definition

For each vector \(\mathbf x \in \mathbb{R}^{n}\) along the **last** dimension (feature / head dim), with stabilizer \(\varepsilon > 0\):

\[
\|\mathbf x\|_2^2 = \sum_{i=1}^{n} x_i^2, \qquad
\mathrm{rstd}(\mathbf x) = \bigl(\|\mathbf x\|_2^2 + \varepsilon\bigr)^{-1/2}, \qquad
\mathbf y = \mathbf x \odot \mathrm{rstd}(\mathbf x).
\]

Equivalently \(\mathbf y = \mathbf x / \sqrt{\|\mathbf x\|_2^2 + \varepsilon}\). This is **not** RMSNorm: RMSNorm uses \(\sqrt{\frac{1}{n}\sum_i x_i^2 + \varepsilon}\) and typically applies learned \(\gamma\). L2-normalize uses the **un-averaged** sum of squares and **no** affine scale inside the norm ([`docs/plans/step5-stateful-mixers.md` §7.6](docs/plans/step5-stateful-mixers.md)).

**I/O shapes:**

| Use case | Input \(X\) | Output \(Y\) | Reduction axis | \(n = \lvert X \rvert\) |
|----------|-------------|--------------|----------------|------------------------|
| DeltaNet Q (per head) | \((B, S, h_k, d_h)\) | same | last (\(d_h\)) | \(B S h_k d_h\) |
| DeltaNet K (per head) | \((B, S, h_k, d_h)\) | same | last (\(d_h\)) | \(B S h_k d_h\) |
| Generic last-dim norm | \((\ldots, n)\) | same | last | \(\prod \text{dims}\) |

In Qwen / torchtitan reference paths, L2-normalize runs on Q and K in **fp32**, then queries may be scaled by \(d_k^{-1/2}\) **outside** `L2Normalize` ([`torchtitan/models/qwen3_5/model.py`](https://github.com/pytorch/torchtitan/blob/cd8950ba/torchtitan/models/qwen3_5/model.py)).

**Numerics policies (bytes, not FLOPs):**
- Zepto `L2Normalize`: `Cast` to fp32 for reduction, restore activation dtype (default bf16) on output ([`src/zepto/modules/l2_normalize.py`](src/zepto/modules/l2_normalize.py)).
- FLA Triton `l2norm_fwd`: accumulate \(\sum x^2\) in fp32 on-chip; store `rstd` as fp32 scalar per row ([`fla/modules/l2norm.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py)).
- Default \(\varepsilon = 10^{-6}\) (Zepto module, Qwen reference, FLA kernels).
- **Not** identical to `torch.nn.functional.normalize`: PyTorch divides by \(\max(\lVert v \rVert_p, \epsilon)\) ([docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)); FLA/Qwen use \(\sqrt{\sum x^2 + \varepsilon}\) in the denominator. Zepto bills the **FLA/Qwen** semantics.

**Zepto identity reference:** 7-op chain — `Cast(fp32) → Multiply(square) → ReduceSum(keepdim) → Add(ε) → SquareRoot → Divide → Cast(act)`.

**Structural vs eager:** Production FLA kernel fuses square + sum + rsqrt + scale in one Triton launch; identity lowering materializes `squared`, `(…,1)` `norm_sq`, `(…,1)` `denom`, and full-rank `normalized`.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | FLA **`l2norm_fwd` / `l2norm_bwd`** (Triton; CUDA/ROCm); optional in-kernel Q/K norm inside **`chunk_gated_delta_rule`** (`use_qk_l2norm_in_kernel=True`) |
| **Identity lowering** | Unfused semantic primitives | Zepto `L2Normalize`: 7-op fp32-reduction chain; multiple full-rank / row-scalar `ALLOCATE`s; primitive FLOP sum \(3n + S'\) for \(S'\) rows |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/l2_normalize/fla` (planned) — **\(3n\)** forward, **\(4n\)** backward; elides `squared` / `normalized` temps; **`SAVE rstd`** |

Default Zepto estimates use the **fused region leaf**, not the sum of identity primitives.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **Zepto identity module** | [`L2Normalize`](src/zepto/modules/l2_normalize.py) | No | A | Any | Both (semantic chain) |
| **PyTorch decomposed** | `pow` / `sum` / `sqrt` / `div` or `rsqrt` | No | A | Any | Both |
| **`F.normalize`** | [`torch.nn.functional.normalize`](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html) | ATen may fuse | A | Any | Both; **different ε semantics** |
| **FLA standalone** | [`fla/modules/l2norm.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py) `l2norm_fwd`, `l2norm_bwd`, `L2Norm` nn.Module | Yes | A | CUDA / ROCm (Triton) | Both; returns \((y, \mathrm{rstd})\) |
| **FLA in DeltaNet scan** | [`fla/ops/gated_delta_rule/chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py), [`delta_rule/chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/chunk.py) | Yes (Q/K norm inside scan) | A (sub-leaf) | CUDA / ROCm | Both; caches `q_rstd`, `k_rstd` ([PR #506](https://github.com/fla-org/flash-linear-attention/pull/506)) |
| **Qwen reference (eager)** | [`torchtitan` `_l2norm`](https://github.com/pytorch/torchtitan/blob/cd8950ba/torchtitan/models/qwen3_5/model.py) | No | A | Any | Reference / `torch_native` backend |
| **vLLM / SGLang FLA ports** | vLLM `l2norm_fwd` in FLA chunk ops; SGLang FLA layers | Yes (in scan) | A | CUDA | Inference / serving |
| **HF Transformers hub** | No dedicated `@use_kernel_forward_from_hub("L2Normalize")` today | — | A | — | DeltaNet models use FLA or eager reference |
| **Zepto (registered stub)** | [`mixer_stub.py`](src/zepto/analysis/lowering/implementations/regions/mixer_stub.py) `region/l2_normalize` | Cost leaf (stub) | A | Any | **`to_implement`** — raises until kernel-full |

There is **no** Liger-style Hub kernel for L2-normalize separate from FLA; Zepto should treat **FLA Triton** as the primary fused reference for this leaf.

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **A** — standalone last-dimension normalization (same class as `region/rmsnorm`, but different math: un-averaged ℓ₂, no \(\gamma\)).

**Mutually exclusive (same `L2Normalize` module invocation):**
- Any future `region/l2_normalize/*` vs the 7-op identity chain — fusion replaces the entire decomposed subgraph.
- `region/rmsnorm/*` vs `region/l2_normalize/*` — different reductions (\(1/n\) mean-square vs raw sum-square); do not substitute.
- `region/layernorm/*` vs `region/l2_normalize/*` — LayerNorm adds mean centering (\(5n\) leaf).

**Composable:**
- Two sequential `L2Normalize` invocations on Q and K in `GatedDeltaNet` — bill **\(2 \times 3n\)** forward when modeled as standalone leaves (\(n_q = B S h_k d_h\) each).
- FLA **`use_qk_l2norm_in_kernel=True`**: Q/K L2 runs **inside** `region/gated_delta_scan` (boundary C for the scan, boundary A sub-leaf for norm). Zepto should **not** double-count standalone `region/l2_normalize` on the same tensors when the scan region already includes in-kernel norm (composition rule: scan fused variant elides separate L2 regions on matched provenance — **TBD in architect**; default stack in `GatedDeltaNet` uses explicit `L2Normalize` modules before scan).
- L2 on Q/K is orthogonal to RoPE, RMSNorm, and FlashAttention — no boundary C replacement.

**Execution constraints:**
- Last-axis normalization only; FLA kernels flatten leading dims to \(T\) rows × \(D\) features.
- Contiguous tensors preferred in FLA entry points.
- Query scaling \(d_k^{-1/2}\) is **downstream** of L2 in Qwen — not part of this region.
- Inference `requires_grad=False` → backward FLOPs = 0; no `SAVE rstd` required for peak VRAM on forward-only graphs.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert x \rvert\). For one headed tensor \(X \in \mathbb{R}^{S \times d_h}\), \(n = S \cdot d_h\). Let \(S'\) = number of independent row reductions (= \(n / d_h\)).

Per row: \(\mathrm{rstd} = \bigl(\sum_i x_i^2 + \varepsilon\bigr)^{-1/2}\), \(y_i = x_i \cdot \mathrm{rstd}\).

### 4.1 Identity lowering (7-op Zepto chain)

| Step | Primitive | Tile shape | FLOPs rule | Subtotal |
|------|-----------|------------|------------|----------|
| 0,6 | `Cast` ×2 | \((S', d_h)\) | 0 (retype) | 0 |
| 1 | `Multiply` (square) | \((S', d_h)\) | 1/elem | \(n\) |
| 2 | `ReduceSum` | \((S', d_h)→(S',1)\) | \(n - S'\) | \(n - S'\) |
| 3 | `Add` (+ε) | \((S',1)\) | 1/elem | \(S'\) |
| 4 | `SquareRoot` | \((S',1)\) | 1/elem | \(S'\) |
| 5 | `Divide` (normalize) | \((S', d_h)\) | 1/elem | \(n\) |
| **Identity total** | | | | **\(3n + S'\)** |

Casts bill **0 FLOPs**. Row-scalar `+ε` and `sqrt` are \(O(S')\) and omitted from the fused leaf (same policy as RMSNorm §5 in `docs/kernel-implementation.md`).

### 4.2 Fused region leaf (kernel-accurate)

Three billed stages, one FLOP per element of \(X\) per stage ([planned `L2NormalizeRecipe.forward_flops_per_element = 3`](docs/plans/step5-stateful-mixers.md)):

| Stage | Arithmetic | Per row (\(d_h\)) | For \(X\) |
|-------|------------|-------------------|-----------|
| Square | \(x_i^2\) | \(d_h\) muls | \(n\) |
| Sum | \(\sum_i x_i^2\) (no `/d_h`) | \(d_h\) (reduction billing) | \(n\) |
| Normalize | \(x_i \cdot \mathrm{rstd}\) | \(d_h\) muls | \(n\) |
| **Fused leaf total** | | \(3 d_h\) | **\(3n\)** |

FLA Triton implements square + sum + `rsqrt` + scale in one kernel; Zepto counts **\(3n\)** elementwise-equivalent work, not separate `sqrt`/`add` scalar ops.

**DeltaNet Q+K (standalone leaves, one layer, one stream):**

\[
\mathrm{FLOPs}_{\mathrm{fwd,QK}} = 3 \cdot \bigl(\lvert Q \rvert + \lvert K \rvert\bigr)
= 3 \cdot B S h_k d_h \cdot 2 \quad \text{(when } Q,K \text{ same shape)}.
\]

Example: \(B{=}1\), \(S{=}8192\), \(h_k{=}8\), \(d_h{=}128\) → \(n_{\mathrm{qk}} = 8{,}388{,}608\) per tensor → **\(3n\)** ≈ 25.2M FLOPs per Q or K → **≈ 50.3M FLOPs** for both.

### 4.3 Paper-comparable (Appendix E / Apertus)

Apertus-8B Appendix E does **not** enumerate DeltaNet Q/K L2-normalize (Apertus uses RMSNorm + optional QK-Norm, not ℓ₂ head norm). No separate paper-comparable constant applies; **kernel-accurate \(3n\)** is the Zepto default.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 3 \cdot \lvert x \rvert = 3n.
\]

**Arithmetic intensity (numeric example):**

One Q tensor, \(B{=}1\), \(S{=}8192\), \(h_k{=}8\), \(d_h{=}128\), bf16 (\(e{=}2\)):

- FLOPs: \(3 \times 8192 \times 8 \times 128 = 25{,}165{,}824 \approx 2.52 \times 10^7\)
- Minimum HBM traffic (fused FLA): read \(x\) (\(8192 \times 8 \times 128 \times 2 = 16\,\mathrm{MiB}\)) + write \(y\) (\(16\,\mathrm{MiB}\)) + write `rstd` (\(8192 \times 8 \times 4 \approx 256\,\mathrm{KiB}\) fp32) \(\approx 32.3\,\mathrm{MiB}\)
- AI \(\approx 2.52 \times 10^7 / (32.3 \times 2^{20}) \approx 0.74\,\mathrm{FLOP/byte}\) — **memory-bound**

Unfused identity adds \(\sim 1\text{–}2 \times n e\) peak from `squared` + `normalized` (\(\sim 32\,\mathrm{MiB}\) elidable per norm at this shape).

---

## Section 5: Backward FLOPs — step-by-step derivation

Training context with `requires_grad=True`. FLA saves **`rstd`** per row and uses normalized output \(y\) in the backward kernel ([`l2norm_bwd_kernel`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py)):

\[
\frac{\partial L}{\partial x_i} = \mathrm{rstd}\left(\frac{\partial L}{\partial y_i} - y_i \sum_j \frac{\partial L}{\partial y_j} y_j\right).
\]

### 5.1 Saved vs recomputed

| Path | Saved for backward | Recomputed |
|------|-------------------|------------|
| Zepto identity (autograd on chain) | Intermediate activations + \(x\) | Partial |
| FLA standalone / in-scan | **`rstd`**, normalized **\(y\)** (or \(x\) depending on API) | — |
| `requires_grad=False` | — | **backward FLOPs = 0** |

### 5.2 Fused region backward (kernel-accurate, \(4n\))

| Step | Primitive | Per element | Subtotal |
|------|-----------|-------------|----------|
| 1 | \(\partial L/\partial y_i \cdot \mathrm{rstd}\) | 1 mul | \(n\) |
| 2 | \(y_i \cdot \partial L/\partial y_i\) and row sum | 1 mul + reduction share | \(n\) |
| 3 | scale dot product by \(y_i \cdot \mathrm{rstd}\) | 2 mul | \(n\) |
| 4 | subtract from step 1 | 1 add | \(n\) |
| **Fused bwd total** | | **4/elem** | **\(4n\)** |

Matches RMSNorm-style backward without \(\partial\gamma\) ([`RMSNormRecipe.backward_flops_per_element = 4`](src/zepto/analysis/lowering/recipes/rmsnorm.py)).

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} =
\begin{cases}
4 \cdot \lvert x \rvert = 4n & \text{if } \texttt{requires\_grad=True} \\
0 & \text{if } \texttt{requires\_grad=False}
\end{cases}
\]

---

## Section 6: Memory — forward-lived tensors & peak VRAM

Let \(n = \lvert x \rvert\), activation element size \(e\) (bf16: \(e{=}2\)), \(S'\) = number of norm rows.

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|-------------------|------------------|
| Identity (7-op) | `x_fp32`, `squared`, `norm_sq`, `denom`, `normalized`, `y` | \(\sim 2n e\) full-rank temps + output | — |
| Fused FLA / planned Zepto leaf | `y`, `rstd` \((S',1)\) fp32 per logical row | \(n e + S' \cdot 4\) | `squared`, `normalized`, fp32 cast temps |
| In-kernel inside gated delta scan | `y` consumed by scan; `rstd` cached for bwd | bundled in scan workspace | separate Q/K norm output buffers may alias scan pipeline |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| Identity eager | chain + \(x\) | various | \(\gg n e\) | per-op SAVE on unfused chain |
| FLA / planned `region/l2_normalize/fla` | **`rstd`** | `(*batch, 1)` fp32 | \(S' \cdot 4\) | `SAVE rstd` |
| Inference-only | none | — | 0 | no SAVE |

No persistent weights for pure L2-normalize (contrast RMSNorm \(\gamma\)).

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(x_fp32) → ALLOCATE(squared) → ALLOCATE(norm_sq) → ALLOCATE(denom)
  → ALLOCATE(normalized) → ALLOCATE(y)
  → SAVE(x_fp32) → SAVE(squared) → SAVE(norm_sq) → … (per-op autograd)
```

**Fused region leaf (FLA / planned Zepto):**
```
ALLOCATE(y) → ALLOCATE(rstd) → SAVE(rstd)
```

**Apertus-8B trace:** N/A for native Apertus (no L2 Q/K). **DeltaNet-shaped example** (\(S{=}8192\), \(h_k{=}8\), \(d_h{=}128\), bf16, one Q norm, training):
- Input \(x\): \(16\,\mathrm{MiB}\)
- Output \(y\): \(16\,\mathrm{MiB}\)
- Saved `rstd`: \(8192 \times 8 \times 4 \approx 256\,\mathrm{KiB}\)
- Elided vs identity: \(\sim 32\,\mathrm{MiB}\) (`squared` + `normalized` not allocated)
- Full `GatedDeltaNet` block: **two** L2 leaves (Q, K) before scan → **≈ 64 MiB** elidable peak vs unfused identity for the pair

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Zepto identity | `modules/l2_normalize.py` | No | A | \(3n + S'\) | autograd chain | Full-rank temps | identity 7-op |
| FLA Triton | flash-linear-attention | Yes | A | \(3n\) | \(4n\) | On-chip; SAVE `rstd` | `region/l2_normalize/fla` (planned) |
| FLA in scan | gated_delta_rule chunk | Yes (sub-leaf) | A | \(3n\) per Q/K (in scan) | \(4n\) | `q_rstd`, `k_rstd` | compose with `region/gated_delta_scan` |
| `F.normalize` | PyTorch | varies | A | \(\approx 3n\) | \(\approx 4n\) | ATen temps | **not** Zepto-default (ε semantics) |
| Qwen `_l2norm` | torchtitan | No | A | \(3n + S'\) | eager autograd | materialized temps | identity-equivalent |
| RMSNorm | Liger / Zepto | Yes | A | \(4n\) | \(4n\) | SAVE `rstd` + \(\gamma\) | distinct `region/rmsnorm/*` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: l2_normalize
recommended_region_ids:
  - id: region/l2_normalize/fla
    variant: fla
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - id: region/l2_normalize/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
pattern_rule:
  op_families:
    - cast
    - multiply
    - reduce_sum
    - add
    - square_root
    - divide
    - cast
recipe:
  forward_flops: "3 * numel(input)"
  backward_flops: "4 * numel(input) if requires_grad else 0"
  forward_flops_per_element: 3
  backward_flops_per_element: 4
  materialize_rstd: true
  save_rstd: true
  elided_temps:
    - squared
    - normalized
    - norm_sq
    - denom
    - x_fp32_cast_temp
  saved_backward:
    - name: rstd
      shape: "(*batch_dims, 1)"
      dtype: fp32
  resource_events_forward:
    - "ALLOCATE(y)"
    - "ALLOCATE(rstd)"
    - "SAVE(rstd)"
  resource_events_backward:
    - "LOAD(rstd)"
  numerics_tags:
    - fp32_reduction
    - sum_of_squares_not_mean
    - no_affine_gamma
    - fla_rsqrt_semantics
capabilities:
  - fused
priority: 6
composition_notes:
  gated_delta_net: "Bill two standalone leaves (Q,K) unless scan fused variant subsumes in-kernel norm"
  mutually_exclusive_with:
    - region/rmsnorm/*
    - region/layernorm/*
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- FLA L2Norm Triton module: [`fla/modules/l2norm.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py)
- FLA L2Norm + rstd caching (DeltaNet / Comba): [flash-linear-attention PR #506](https://github.com/fla-org/flash-linear-attention/pull/506)
- Gated delta rule with `use_qk_l2norm_in_kernel`: [`fla/ops/gated_delta_rule/chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)
- Qwen3.5 reference `_l2norm` and FLA backends: [`torchtitan/models/qwen3_5/model.py`](https://github.com/pytorch/torchtitan/blob/cd8950ba/torchtitan/models/qwen3_5/model.py)
- PyTorch `F.normalize` (contrast ε semantics): [torch.nn.functional.normalize](https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)
- Zepto module: [`src/zepto/modules/l2_normalize.py`](src/zepto/modules/l2_normalize.py)
- Zepto stub provenance: [`src/zepto/analysis/lowering/implementations/regions/mixer_stub.py`](src/zepto/analysis/lowering/implementations/regions/mixer_stub.py)
- Implementation plan: [`docs/plans/step5-stateful-mixers.md`](docs/plans/step5-stateful-mixers.md) §7.6, §Phase 3 region table
- Model gap note: [`docs/model-architecture-gaps-2026-09-14.md`](docs/model-architecture-gaps-2026-09-14.md)
- Zepto cost conventions: [`docs/kernel-implementation.md`](docs/kernel-implementation.md) §1, §5 (RMSNorm analogy)

**Verification date:** 2026-09-17 (FLA `l2norm.py` on GitHub `main`; Zepto `L2Normalize` module; torchtitan Qwen3.5 reference)

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| P2 | `region/l2_normalize/fla` | §8 | A | GatedDeltaNet / Qwen3.5 cost path; stub registered in `mixer_stub.py` — needs recipe + lowering variant |
| P2 | `region/l2_normalize/reference` | §8 | A | Hardware-agnostic fused leaf matching identity provenance for tests |
| P3 | Composition with `region/gated_delta_scan` | §3 | A/C | Avoid double billing when `use_qk_l2norm_in_kernel` fuses Q/K norm into scan — architect rule TBD |
