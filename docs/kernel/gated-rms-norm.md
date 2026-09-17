# Zepto kernel research: GatedRMSNorm

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Standalone per-head normalization + SiLU output gate (boundary A); Qwen3-Next / FLA Gated DeltaNet mixer tail; prefill primary (\(S>1\)) and decode (\(S=1\)); training + inference; CUDA/XPU via FLA hub kernel
**Context:** Zepto module `GatedRMSNorm` is registered in the graph; fused region `region/gated_rms_norm` is a **stub** (`mixer_stub.py`). Default cost leaf should be **kernel-accurate** fused RMS+gate (FLA `FusedRMSNormGated` / HF `RMSNormGated` hub), not a separate `region/rmsnorm` + `region/silu` stack on the same invocation. **Not** `GatedGroupedRMSNorm` (gate-before-group-norm — Nemotron/Zamba order).

---

## Section 0: Mathematical definition

Qwen3-Next Gated DeltaNet applies **RMS normalization and learned scale on the scan value**, then multiplies by **SiLU(gate)** on a separate projected gate tensor ([`Qwen3NextRMSNormGated`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py); FLA [`FusedRMSNormGated`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py); Zepto [`gated_rms_norm.py`](../src/zepto/modules/gated_rms_norm.py)).

For value \(\mathbf y\), gate \(\mathbf z\), feature dimension \(d\) (per-head `head_dim` / `head_v_dim`), learned scale \(\boldsymbol\gamma \in \mathbb{R}^d\), and \(\varepsilon > 0\):

\[
\begin{aligned}
\mathrm{rms}_\epsilon(\mathbf y)
&= \sqrt{\frac{1}{d}\sum_{j=1}^{d} y_j^2 + \varepsilon}, \\
\widehat{\mathbf y} &= \boldsymbol\gamma \odot \frac{\mathbf y}{\mathrm{rms}_\epsilon(\mathbf y)}, \\
\operatorname{GatedRMS}(\mathbf y, \mathbf z)
&= \widehat{\mathbf y} \odot \operatorname{SiLU}(\mathbf z),
\qquad
\operatorname{SiLU}(z) = z \cdot \sigma(z).
\end{aligned}
\]

**Order is material:** normalize → ×\(\boldsymbol\gamma\) → ×\(\operatorname{SiLU}(\mathbf z)\). A raw `Multiply(normalized, z)` without SiLU is **not** checkpoint-faithful. This differs from **`GatedGroupedRMSNorm`**, which applies SiLU to \(\mathbf z\) **before** grouped RMS on \(\mathbf y \odot \operatorname{SiLU}(\mathbf z)\) ([`step5-stateful-mixers.md`](../docs/plans/step5-stateful-mixers.md) §7.8).

**I/O shapes (Gated DeltaNet output norm):**

| Tensor | Typical shape | Notes |
|--------|---------------|--------|
| Value \(\mathbf y\) | \((S, d)\) per head, or \((S, H_v, d)\) batched | Scan output before merge |
| Gate \(\mathbf z\) | same as \(\mathbf y\) | From `qkvz` projection (Qwen `z` split) |
| \(\boldsymbol\gamma\) | \((d,)\) | One weight vector **per head** (module instance) |
| Output | same as \(\mathbf y\) | bf16/fp16 activation dtype |

Let \(n = \lvert \mathbf y \rvert = \lvert \mathbf z \rvert\). Full mixer: \(n = S \cdot H_v \cdot d\) after merging heads (Zepto `GatedDeltaNet` loops per head but bills one leaf over total numel).

**Numerics policies (bytes / dtype, not FLOP constants):**
- HF / Zepto reference: variance on **value** accumulated in **fp32**; cast normalized value back to activation dtype before ×\(\boldsymbol\gamma\); SiLU on **gate** in fp32 ([`Qwen3NextRMSNormGated.forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)).
- FLA `FusedRMSNormGated`: fp32 reduction on-chip; default activation string `"swish"` aliases **SiLU** (`z * sigmoid(z)`) — Qwen hub uses `activation="silu"`.
- \(\varepsilon\): Zepto default \(10^{-6}\); Qwen3-Next config `layer_norm_epsilon` / `rms_norm_eps`.

**Structural vs eager:** Production paths fuse square + mean + normalize + ×\(\boldsymbol\gamma\) + SiLU(gate) + final multiply in one Triton kernel (FLA) or hub op. Identity lowering materializes `squared`, row `variance`, `normalized`, `sigmoid(gate)`, `silu_gate`, and output.

**Zepto identity reference:** `src/zepto/modules/gated_rms_norm.py` — `Cast(fp32)` on value/gate → RMS chain → `Cast(act)` → `ParameterScale(γ)` → `Sigmoid(gate)` → `Multiply` (SiLU) → `Multiply` (apply gate).

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | FLA **`FusedRMSNormGated`** (Triton; [`fused_norm_gate.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py)); HF Hub **`kernels-community/fla`** / `FusedRMSNormGated` via `@use_kernel_forward_from_hub("RMSNormGated")` on [`Qwen3NextRMSNormGated`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) |
| **Identity lowering** | Unfused semantic chain | Zepto `GatedRMSNorm` primitive sequence (§4.1); HF eager Python in `Qwen3NextRMSNormGated` (decomposed ops, no fused autograd) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/gated_rms_norm/fla` (recommended); **`region/gated_rms_norm`** stub today — **\(10n\)** forward, **\(14n\)** backward; elides full-rank RMS/SiLU temps |

Default Zepto estimates use the **fused region leaf**, not `region/rmsnorm` + `region/silu` + extra `Multiply` on the same `GatedRMSNorm` provenance.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF eager (Qwen3-Next)** | [`Qwen3NextRMSNormGated`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) | No | A | Any | Both (PyTorch autograd on decomposed ops) |
| **FLA Triton** | [`FusedRMSNormGated`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py) | Yes | A | CUDA (primary); used inside [`GatedDeltaNet`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py) when `use_gate=True` | Both; saves **`rstd`** fp32 per row |
| **FLA reference** | [`rms_norm_ref` + gate in `layernorm_gated.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py) (`norm_before_gate=True`) | No | A | CPU/GPU | Both |
| **HF Hub** | [`hub_kernels.py` → `RMSNormGated`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py): `kernels-community/fla` / `FusedRMSNormGated` v1 | Yes | A | CUDA + XPU entries | Both (training + inference modes registered) |
| **Chunk GDN mega-fusion** | FLA `chunk_gated_delta_rule(..., use_gate_in_kernel=True)` | Partial (scan + gate in kernel) | Mixer **B** (not this leaf) | CUDA | Both | 
| **Zepto module** | `modules/gated_rms_norm.py` | No (semantic) | A | Any | Both |
| **Zepto region** | `region/gated_rms_norm` | Stub (`fused` capability) | A | pending | TBI |

Mega-fusion in the gated-delta **scan** kernel may absorb the output gate when `use_gate_in_kernel=True`; that is a **different fusion boundary** from standalone `GatedRMSNorm` costing ([`gated-delta-scan.md`](kernel/gated-delta-scan.md)). Do not bill `region/gated_rms_norm` when the graph region is promoted to scan+gate fusion.

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **A** — standalone row normalization over last axis \(d\), extended with an elementwise SiLU gate (same boundary class as `region/rmsnorm`, not attention B/C/D).

**Mutually exclusive (same module invocation):**
- `region/gated_rms_norm/*` vs decomposed identity chain on `GatedRMSNorm` provenance (`prov-gated-rms-norm`).
- `region/gated_rms_norm` vs **`region/rmsnorm` + `region/silu` + `Multiply`** on the same `(value, gate)` outputs — double-counting.
- `region/gated_rms_norm` vs **`region/gated_grouped_rms_norm`** — different math (gate before grouped norm).
- Optional scan mega-fusion (`use_gate_in_kernel=True`) vs separate **`GatedRMSNorm`** leaf on the same tensor edge.

**Composable:**
- **Gated DeltaNet stack:** `Linear → DepthwiseCausalConv1d → L2Normalize → GatedDeltaScan → GatedRMSNorm → Linear` — each leaf bills independently unless scan fusion replaces the norm ([`gated-delta-scan.md`](kernel/gated-delta-scan.md)).
- FLA / Liger-style patches compose with other hub layers; this leaf does not fuse into `region/gqa/*`.

**Execution constraints:**
- `value` and `gate` must match shape; normalize over **last** axis; `normalized_shape == d`.
- FLA `FusedRMSNormGated`: `activation ∈ {swish, silu, sigmoid}`; Qwen uses **silu**.
- Hub: CUDA and XPU route to `kernels-community/fla`; no separate MPS entry for `RMSNormGated` in default `hub_kernels.py` (falls back to eager).
- Zepto stub requires `requested_capabilities={'fused'}` until kernel-full implements lowering.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert \mathbf y \rvert = \lvert \mathbf z \rvert\). Per row of length \(d\), with \(S_{\mathrm{row}} = n/d\) row count.

RMS block matches [`RMSNorm` §4](kernel/rmsnorm.md): **\(4d\)** per row for square + mean + normalize + ×\(\boldsymbol\gamma\). SiLU on gate matches [`SiLU` §4](kernel/silu.md): **\(5d\)** per row. Final **`Multiply(\widehat y, SiLU(z))`**: **\(d\)** per row.

### 4.1 Identity lowering (Zepto / HF eager chain)

| Step | Primitive | Tile shape | FLOPs rule | Subtotal |
|------|-----------|------------|------------|----------|
| 0–1 | `Cast` ×2 (value, gate) | \((\ldots,d)\) | 0 | 0 |
| 2 | `Multiply` (square value) | full | 1/elem | \(n\) |
| 3 | `ReduceSum` + `Divide` (/d) | reduce last | \(n - S_{\mathrm{row}}\) + \(S_{\mathrm{row}}\) | \(n\) |
| 4 | `Add` (+ε), `SquareRoot` | \((S_{\mathrm{row}},1)\) | 2/row-scalar | \(2S_{\mathrm{row}}\) |
| 5 | `Divide` (normalize value) | full | 1/elem | \(n\) |
| 6 | `Cast` (act dtype) | full | 0 | 0 |
| 7 | `ParameterScale` (×γ) | full | 1/elem | \(n\) |
| 8 | `Sigmoid` (gate) | full | 4/elem | \(4n\) |
| 9 | `Multiply` (gate × σ) | full | 1/elem | \(n\) |
| 10 | `Multiply` (× SiLU gate) | full | 1/elem | \(n\) |
| **Identity total** | | | | **\(10n + 2S_{\mathrm{row}}\)** |

Casts bill **0 FLOPs**. Row-scalar `+ε` and `sqrt` are **kept** in identity sums (same policy as `region/rmsnorm` identity oracle).

SiLU steps 8–9 bill **\(5n\)** (consistent with `region/silu` identity **\(5n\)**).

### 4.2 Fused region leaf (kernel-accurate)

FLA / hub fuse steps 2–10 without HBM temps. Count **RMS γ-only leaf** + **SiLU** + **output gate multiply**:

| Stage | Arithmetic | Per element | Subtotal |
|-------|------------|-------------|----------|
| RMS (fused) | square, mean, normalize, ×γ | 4 | \(4n\) |
| SiLU on gate | \(\sigma(z)\) + \(z \cdot \sigma(z)\) | 5 | \(5n\) |
| Apply gate | \(\widehat y \odot \operatorname{SiLU}(z)\) | 1 | \(n\) |
| **Fused leaf total** | | **10** | **\(10n\)** |

Row-scalar `+ε` / `rsqrt` (\(O(S_{\mathrm{row}})\)) **dropped** from the fused leaf (RMSNorm leaf policy).

### 4.3 Paper-comparable (if different)

Appendix E / Apertus FLOP tables count **RMSNorm (\(4n\))** in transformer blocks but **omit** SiLU gate tails on mixer outputs. For Qwen3-Next GDN, a paper-comparable stack might count **only** the RMSNorm slice (**\(4n\)**) and omit **\(6n\)** gate work — label explicitly when comparing to kernel-accurate Zepto.

**Kernel-accurate closed form:**
\[
\mathrm{FLOPs}_{\mathrm{GatedRMS,fwd}} = 10 \cdot n.
\]

**Arithmetic intensity (numeric example — Qwen3-Next GDN output norm):**

\(S{=}8192\), \(H_v{=}32\), \(d{=}128\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| \(n = S H_v d\) | \(33{,}554{,}432\) |
| FLOPs | \(10n \approx 3.36 \times 10^8\) |
| Min HBM (fused leaf, training) | read \(\mathbf y\) + read \(\mathbf z\): \(2n e \approx 64\,\mathrm{MiB}\); read \(\boldsymbol\gamma\): \(H_v d e \approx 8\,\mathrm{KiB}\); write output: \(n e \approx 64\,\mathrm{MiB}\); write/save `rstd`: \(S H_v \cdot 4 \approx 1\,\mathrm{MiB}\) |
| AI (fused, approx.) | \(\approx 3.36 \times 10^8 / (130 \times 2^{20}) \approx 2.5\,\mathrm{FLOP/byte}\) — **memory-bound** |

Unfused identity adds \(\sim 2n e\) peak from `squared` + `normalized` + `silu_gate` temps (\(\sim 128\,\mathrm{MiB}\) elidable at this shape).

---

## Section 5: Backward FLOPs — step-by-step derivation

Training context (`requires_grad=True`). Inference-only → **backward FLOPs = 0**.

### 5.1 Saved vs recomputed

| Path | Saved for backward | Recomputed |
|------|-------------------|------------|
| HF eager | Autograd saves intermediate activations on decomposed graph | Partial |
| FLA / hub fused | **`rstd`**, inputs **value**, **gate** (standard fused norm+gate autograd) | SiLU \(\sigma(z)\); normalized value |
| Zepto fused leaf | **`rstd`** via `SAVE`; inputs as upstream SAVE | SiLU factors from saved gate |

### 5.2 Fused region backward (kernel-accurate)

Let \(\widehat{\mathbf y}\) be post-RMS+γ, \(\mathbf g = \operatorname{SiLU}(\mathbf z)\), output \(\mathbf o = \widehat{\mathbf y} \odot \mathbf g\).

| Step | VJP / primitive | FLOPs/elem | Subtotal |
|------|-----------------|----------|----------|
| 1 | `Multiply` VJP: \(\partial L/\partial \widehat{\mathbf y} = \partial L/\partial \mathbf o \odot \mathbf g\) | 1 | \(n\) |
| 2 | `Multiply` VJP: \(\partial L/\partial \mathbf g = \partial L/\partial \mathbf o \odot \widehat{\mathbf y}\) | 1 | \(n\) |
| 3 | SiLU VJP on \(\mathbf z\) (recompute \(\sigma\)) | 8 | \(8n\) |
| 4 | RMSNorm+γ VJP on value (saved `rstd`) | 4 | \(4n\) |
| **Fused bwd total** | | | **\(14n\)** |

SiLU backward **\(8n\)** matches `region/silu` ([`silu.md`](kernel/silu.md) §5.2). RMS backward **\(4n\)** matches `region/rmsnorm`.

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{GatedRMS,bwd}} =
\begin{cases}
14 \cdot n & \text{if } \texttt{requires\_grad=True} \\
0 & \text{otherwise}
\end{cases}
\]

**Paper-comparable:** backward may be quoted as **\(4n\)** (RMS only) in GEMM-centric tables — omit **\(10n\)** gate/VJP work when comparing.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

Let \(n = \lvert \mathbf y \rvert\), activation element size \(e\).

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|-------------------|------------------|
| Identity lowering | `value_fp32`, `squared`, `variance`, `denom`, `normalized`, `scaled`, `sig`, `silu_gate`, `output` | \(\sim 3n e\) full-rank temps + output | — |
| Fused FLA / hub | `output`, `rstd` \((\ldots,1)\) fp32 | \(n e + S_{\mathrm{row}} \cdot 4\) | `squared`, `normalized`, `silu_gate`, σ temp |
| Inference (`requires_grad=False`) | `output` only | \(n e\) | same |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes (order) | Zepto SAVE policy |
|------|---------------|-------|---------------|-------------------|
| Fused FLA / Zepto leaf | **`rstd`** | `(*batch, 1)` fp32 | \(S_{\mathrm{row}} \cdot 4\) | `SAVE(rstd)` |
| | **value**, **gate** | same as I/O | \(2n e\) | upstream module SAVE (scan / projections) |
| Identity eager | many chain temps | various | \(\gg n e\) | per-op SAVE |
| Zepto fused (recommended) | **`rstd` only** at norm leaf | | \(S_{\mathrm{row}} \cdot 4\) | match Liger/FLA RMS policy |

Persistent: \(\boldsymbol\gamma\) shape \((d,)\) per head — `PERSIST`; \(H_v\) heads ⇒ \(H_v \cdot d \cdot e\) total weight bytes (small vs activations).

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(value_fp32) → ALLOCATE(gate_fp32) → ALLOCATE(squared) → ALLOCATE(variance)
  → ALLOCATE(denom) → ALLOCATE(normalized) → ALLOCATE(scaled)
  → ALLOCATE(sig) → ALLOCATE(silu_gate) → ALLOCATE(output)
  → SAVE(value_fp32) → SAVE(gate_fp32) → SAVE(…)  # per-op autograd
```

**Fused region leaf (`region/gated_rms_norm/fla`):**
```
ALLOCATE(output) → ALLOCATE(rstd) → SAVE(rstd)
```

**Qwen3-Next GDN trace (one mixer layer, fused, training):**

\(S{=}8192\), \(H_v{=}32\), \(d{=}128\), bf16:

- Output: \(n e \approx 64\,\mathrm{MiB}\)
- Saved `rstd`: \(S H_v \times 4 \approx 1\,\mathrm{MiB}\)
- Elided vs identity: \(\sim 128\,\mathrm{MiB}\) (three full-rank temps)
- Per decoder layer with one GDN block: **one** `GatedRMSNorm` leaf (\(n = S H_v d\)), not Apertus hidden RMSNorm count

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF eager Qwen | transformers | No | A | \(10n + 2S_{\mathrm{row}}\) | decomposed autograd | Full-rank temps | identity |
| FLA Triton | flash-linear-attention | Yes | A | \(10n\) | \(14n\) | On-chip; SAVE `rstd` | `region/gated_rms_norm/fla` (TBI) |
| HF Hub | kernels-community/fla | Yes | A | \(10n\) | \(14n\) | Same as FLA | hub-routed variant |
| Decomposed Zepto | rmsnorm + silu + mul | No | A | risk **double-count** | — | sum of leaves | **avoid** on same provenance |
| Grouped variant | Nemotron path | Yes/No | A | **different math** | — | — | `region/gated_grouped_rms_norm` |
| Zepto module | `GatedRMSNorm` | Semantic only | A | use fused leaf | use fused leaf | — | stub `region/gated_rms_norm` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: gated_rms_norm
recommended_region_ids:
  - id: region/gated_rms_norm/fla
    variant: fla
    hardware_gate: cuda
    fusion_boundary: A
    status: to_implement
  - id: region/gated_rms_norm/hub-fla
    variant: hub-fla
    hardware_gate: cuda
    fusion_boundary: A
    status: to_implement
  - id: region/gated_rms_norm/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - id: region/gated_rms_norm
    variant: default
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
pattern_rule:
  op_families:
    - cast
    - multiply
    - reduce_sum
    - divide
    - add
    - square_root
    - parameter_scale
    - sigmoid
    - multiply
    - multiply
recipe:
  forward_flops: "10 * numel(value)"
  backward_flops: "14 * numel(value) if requires_grad else 0"
  forward_flops_per_element: 10
  backward_flops_per_element: 14
  materialize_rstd: true
  save_rstd: true
  elided_temps:
    - squared
    - normalized
    - scaled_intermediate
    - sigmoid_gate
    - silu_gate
    - value_fp32_cast_temp
  saved_backward:
    - name: rstd
      shape: "(*batch_dims, 1)"
      dtype: fp32
  resource_events_forward:
    - "ALLOCATE(output)"
    - "ALLOCATE(rstd)"
    - "SAVE(rstd)"
  resource_events_backward:
    - "READ(rstd)"
  numerics_tags:
    - fp32_variance_reduction
    - norm_before_gate
    - silu_gate
    - gamma_only_affine
capabilities:
  - fused
priority: 4
composition_notes:
  mutually_exclusive_with:
    - region/rmsnorm
    - region/silu
    - region/gated_grouped_rms_norm
  parent_mixer: GatedDeltaNet
  scan_mega_fusion: "chunk_gated_delta_rule use_gate_in_kernel=True may subsume this leaf"
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Zhang & Sennrich, RMSNorm baseline: [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Ramachandran et al., SiLU / Swish: [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)
- FLA **`FusedRMSNormGated`** (Triton): [fla/modules/fused_norm_gate.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py)
- FLA **GatedDeltaNet** layer (`o_norm`): [fla/layers/gated_deltanet.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py)
- HuggingFace **Qwen3-Next** eager gated norm: [modeling_qwen3_next.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
- HF Hub **`RMSNormGated`** registry: [hub_kernels.py](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py) → `kernels-community/fla`
- Zepto module: [`src/zepto/modules/gated_rms_norm.py`](../src/zepto/modules/gated_rms_norm.py)
- Zepto mixer plan (math order vs grouped variant): [`docs/plans/step5-stateful-mixers.md`](../docs/plans/step5-stateful-mixers.md) §7.7
- Related RMSNorm / SiLU Zepto leaves: [`docs/kernel/rmsnorm.md`](kernel/rmsnorm.md), [`docs/kernel/silu.md`](kernel/silu.md)
- Qwen3.5 GDN stack notes: [gist qwen3.5-gated-deltanet-analysis](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 4 | `region/gated_rms_norm` | §8 | A | Qwen3-Next / FLA Gated DeltaNet output norm; stub registered in `mixer_stub.py`; blocks accurate GDN horizon costing until kernel-full |

**Open gaps (TBD for architect):**
- Exact backward SAVE set for `kernels-community/fla` autograd vs Zepto `SAVE(rstd)`-only policy (mirror `region/rmsnorm` Liger vs hub-XPU split if hub diverges).
- Interaction with **`use_gate_in_kernel=True`** scan fusion — mutual exclusion rules at lowering time.
- Optional **`region/gated_rms_norm/decomposed`** for debug graphs without `fused` capability (bill as identity **\(10n+2S\)** FLOPs, not separate rmsnorm+silu regions).
