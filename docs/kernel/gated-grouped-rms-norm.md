# Zepto kernel research: GatedGroupedRMSNorm

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Standalone SiLU-gate + grouped RMSNorm leaf (fusion boundary **A**); Mamba-2 mixer output norm after selective scan; prefill (\(S>1\)) and decode (\(S=1\)); training + inference; CUDA primary (HF `mamba-ssm` mega-kernel optional)
**Context:** Nemotron / Zamba-style Mamba-2 stacks gate the scan output **before** independent RMS normalization over \(G\) contiguous channel groups. Zepto module [`GatedGroupedRMSNorm`](../src/zepto/modules/gated_grouped_rms_norm.py) is registered; fused region `region/gated_grouped_rms_norm` is a **stub** in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py). **Not** [`GatedRMSNorm`](../src/zepto/modules/gated_rms_norm.py) (norm-then-gate on full last axis — Qwen3-Next GDN). Default cost leaf = kernel-accurate fused gate+grouped-RMS (**\(10n\)** forward / **\(14n\)** backward when training), not `region/silu` + `region/rmsnorm` on the same provenance.

---

## Section 0: Mathematical definition

Let batch be folded into prefix dimensions. Value \(\mathbf y\) and gate \(\mathbf z\) share shape \((S, m)\) with intermediate width \(m = h \cdot p\) (Mamba head count × head dim). Split channels into \(G\) groups of width \(r = m/G\). Define gated activations and per-group RMSNorm with learned scale \(\boldsymbol\gamma \in \mathbb{R}^m\) ([step5 §7.8](../docs/plans/step5-stateful-mixers.md); HF [`Zamba2RMSNormGated`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/zamba2/modeling_zamba2.py)):

\[
\begin{aligned}
\mathbf u &= \mathbf y \odot \operatorname{SiLU}(\mathbf z), \qquad
\operatorname{SiLU}(t) = t \cdot \sigma(t),\\
\mathbf u^{(g)} &\in \mathbb{R}^r \text{ — group } g \text{ slice of } \mathbf u \text{ along channels},\\
\widehat{\mathbf u}^{(g)}_j &=
\frac{u^{(g)}_j}{\sqrt{\frac{1}{r}\sum_{k=1}^{r}(u^{(g)}_k)^2 + \varepsilon}},\\
\operatorname{GatedGroupedRMS}(\mathbf y, \mathbf z) &=
\boldsymbol\gamma \odot \operatorname{concat}_{g=1}^{G}\widehat{\mathbf u}^{(g)}.
\end{aligned}
\]

**Order is material:** SiLU gate **before** grouped normalization. Swapping gate after normalization changes every group denominator and is checkpoint-incompatible ([`docs/kernel/gated-rms-norm.md`](../docs/kernel/gated-rms-norm.md) §0).

**I/O shapes:**

| Tensor | Shape | Role |
|--------|-------|------|
| `value` (\(\mathbf y\)) | \((S, m)\) or \((B, S, m)\) | Scan output (flattened heads × head dim) |
| `gate` (\(\mathbf z\)) | same as `value` | Mamba gate projection (Nemotron: split from `in_proj`, not scan input) |
| `weight` (\(\boldsymbol\gamma\)) | \((m,)\) | Per-channel scale |
| `output` | same as `value` | Normalized gated tensor |

**Grouped reduction:** variance is computed over the **group-width** axis (Zepto `Reshape` to \((S, G, r)\), `ReduceSum` on axis \(=2\)), producing \((S, G, 1)\) statistics — not a single \((S,1)\) statistic over full \(m\).

**Numerics policies (bytes, not FLOPs):**

- HF / Zepto: gate SiLU and grouped variance in **fp32**; cast normalized tensor back to activation dtype before ×\(\boldsymbol\gamma\) ([`gated_grouped_rms_norm.py`](../src/zepto/modules/gated_grouped_rms_norm.py), [`Zamba2RMSNormGated.forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/zamba2/modeling_zamba2.py)).
- Default \(\varepsilon = 10^{-6}\) (Nemotron H config `layer_norm_epsilon`).
- Saved **`rstd`** / inverse-RMS auxiliaries use shape \((\ldots, G, 1)\) fp32 when materialized — **not** \((\ldots, 1)\) as in full-axis `region/rmsnorm`.

**Zepto identity reference:** `Cast` ×2 → inline `Sigmoid` + `Multiply` (SiLU) → `Multiply` (gate value) → `Reshape` → square → `ReduceSum` + `/r` → `+ε` → `sqrt` → normalize → `Reshape` → `Cast` → `ParameterScale`.

**Structural vs eager:** Production Mamba-2 may fuse conv+scan+grouped gated norm+out_proj in one Triton launch (`norm_before_gate=False`); standalone billing still treats this leaf as boundary **A** unless the graph promotes mega-fusion boundary **B**.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | HF Hub **`mamba2_split_conv1d_scan_combined`** when training path fuses scan + **`Zamba2RMSNormGated`** (`norm_before_gate=False`, grouped `group_size`) ([`NemotronHMamba2Mixer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py)); otherwise eager **`Zamba2RMSNormGated`** PyTorch |
| **Identity lowering** | Unfused semantic chain | Zepto `GatedGroupedRMSNorm` primitive sequence (§4.1); HF eager decomposed ops in `Zamba2RMSNormGated` (`silu` → `view` → `pow/mean` → `rsqrt` → `×γ`) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/gated_grouped_rms_norm` (stub) — **\(10n\)** forward, **\(14n\)** backward; elides `silu_gate`, `gated`, `squared`, `normalized` HBM temps; **`SAVE(group_rstd)`** with shape \((\ldots,G,1)\) |

**Default Zepto estimates use the fused region leaf**, not `region/silu` + `region/rmsnorm` + extra `Multiply` on the same `GatedGroupedRMSNorm` provenance.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF eager (grouped gated)** | [`Zamba2RMSNormGated`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/zamba2/modeling_zamba2.py) | No | A | Any | Both (autograd on decomposed ops) |
| **Nemotron H Mamba-2 mixer** | [`NemotronHMamba2Mixer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) — `self.norm = Zamba2RMSNormGated(..., group_size=m//G)` | Partial | A / **B** | CUDA + Hub | Fused mega-path when `mamba2_split_conv1d_scan_combined` returns; else eager `self.norm(scan_output, gate)` |
| **Mamba-2 mega-kernel** | [`mamba2_split_conv1d_scan_combined`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) + `mamba_ssm` Triton | Yes (conv+scan+norm+out_proj) | **B** | CUDA (Hub) | Training prefill when enabled |
| **FLA `FusedRMSNormGated`** | [fla/modules/fused_norm_gate.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py) | Yes | A | CUDA | **Full-axis** gated norm (Qwen GDN); **not** grouped Nemotron math unless extended |
| **Zepto module (identity)** | [`gated_grouped_rms_norm.py`](../src/zepto/modules/gated_grouped_rms_norm.py) | No | A | Any | Both |
| **Zepto region (stub)** | `region/gated_grouped_rms_norm` in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py) | Cost leaf TBI | A | `requested_capabilities={'fused'}` | `status: to_implement` |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **A** — standalone normalization+gate leaf on \((\mathbf y, \mathbf z)\). Optional **B** when `mamba2_split_conv1d_scan_combined` absorbs grouped gated norm + `out_proj` on the same Nemotron layer (do not double-bill **A** + **B**).

**Mutually exclusive (same module invocation / provenance):**

- `region/gated_grouped_rms_norm/*` vs identity `GatedGroupedRMSNorm` decomposed chain.
- `region/gated_grouped_rms_norm` vs **`region/silu` + `region/rmsnorm` + `Multiply`** on the same `(value, gate)` — double-counting.
- `region/gated_grouped_rms_norm` vs **`region/gated_rms_norm`** — different math (gate-after-full-axis norm vs gate-before-grouped norm).
- Standalone norm leaf vs **`mamba2_split_conv1d_scan_combined`** mega-fusion on the same mixer slot when Hub short-circuit applies.

**Composable:**

- **Mamba2Mixer:** `Linear → DepthwiseCausalConv1d → SelectiveSSMScan → GatedGroupedRMSNorm → Linear` ([`mamba2_mixer.py`](../src/zepto/modules/mamba2_mixer.py); [`mamba2-scan.md`](../docs/kernel/mamba2-scan.md) §3).
- Scan does **not** consume gate `z`; gate applies only at this norm ([`mamba2-scan.md`](../docs/kernel/mamba2-scan.md) §0).

**Execution constraints:**

- `hidden_size % num_groups == 0`; `group_width = hidden_size // num_groups`.
- `value` and `gate` must match shape; normalize over grouped channel axis after reshape.
- Mega-kernel requires CUDA Hub path + training + no cache short-circuit; inference/decode typically calls eager `norm(scan_output, gate)`.
- `requires_grad=False` → backward FLOPs = **0**.

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert \mathbf y \rvert = \lvert \mathbf z \rvert\), \(G\) groups, \(r = m/G\), and \(S_{\mathrm{row}}\) = number of positions (product of prefix dims excluding channel). For \((S,m)\) tensor, \(S_{\mathrm{row}} = S\).

SiLU on gate matches [`region/silu`](../docs/kernel/silu.md): **\(5n\)** (`Sigmoid` **\(4n\)** + `Multiply` **\(n\)**). Grouped RMS block matches [`region/rmsnorm`](../docs/kernel/rmsnorm.md) per-element policy on all \(n\) elements with reduction fan-in \(r\): **\(4n\)** fused (square, mean, normalize, ×\(\boldsymbol\gamma\)).

### 4.1 Identity lowering (Zepto / HF eager chain)

| Step | Primitive | Tile shape | FLOPs rule | Subtotal |
|------|-----------|------------|------------|----------|
| 0–1 | `Cast` ×2 (value, gate) | \((\ldots,m)\) | 0 | 0 |
| 2 | `Sigmoid` (gate) | full | 4/elem | \(4n\) |
| 3 | `Multiply` (gate × σ) | full | 1/elem | \(n\) |
| 4 | `Multiply` (value × SiLU gate) | full | 1/elem | \(n\) |
| 5 | `Reshape` | view | 0 | 0 |
| 6 | `Multiply` (square grouped) | \((S_{\mathrm{row}},G,r)\) | 1/elem | \(n\) |
| 7 | `ReduceSum` + `Divide` (/ \(r\)) | reduce group axis | \(n - S_{\mathrm{row}}G + S_{\mathrm{row}}G\) | \(n\) |
| 8 | `Add` (+ε), `SquareRoot` | \((S_{\mathrm{row}},G,1)\) | 2/group-scalar | \(2 S_{\mathrm{row}} G\) |
| 9 | `Divide` (normalize) | full | 1/elem | \(n\) |
| 10 | `Reshape`, `Cast` | view | 0 | 0 |
| 11 | `ParameterScale` (×γ) | full | 1/elem | \(n\) |
| **Identity total** | | | | **\(10n + 2 S_{\mathrm{row}} G\)** |

SiLU steps 2–3 bill **\(5n\)**; gate multiply step 4 bills **\(n\)**. Grouped RMS steps 6–9 + 11 bill **\(4n + 2 S_{\mathrm{row}} G\)** (same row-scalar policy as `region/rmsnorm` identity **\(4n + 2S\)** with \(S \leftarrow S_{\mathrm{row}} G\)).

### 4.2 Fused region leaf (kernel-accurate)

Fused launch elides HBM temps for SiLU, gated product, squared, and normalized tensors. Count **SiLU** + **gate multiply** + **grouped RMS γ leaf**:

| Stage | Arithmetic | Per element | Subtotal |
|-------|------------|-------------|----------|
| SiLU on gate | \(\sigma(z) + z\cdot\sigma(z)\) | 5 | \(5n\) |
| Apply gate to value | \(\mathbf u = \mathbf y \odot \operatorname{SiLU}(\mathbf z)\) | 1 | \(n\) |
| Grouped RMS (fused) | square, group mean, normalize, ×γ | 4 | \(4n\) |
| **Fused leaf total** | | **10** | **\(10n\)** |

Group-scalar `+ε` / `rsqrt` (\(O(S_{\mathrm{row}} G)\)) **dropped** from fused leaf (RMSNorm leaf policy).

### 4.3 Paper-comparable (if different)

Appendix-style transformer FLOP tables often count **only** an RMSNorm-like term (**\(4n\)**) on mixer outputs and omit SiLU gate work (**\(6n\)**). When comparing to kernel-accurate Zepto, label paper totals explicitly and do **not** substitute them for the **\(10n\)** leaf.

**Kernel-accurate closed form:**
\[
\mathrm{FLOPs}_{\mathrm{GatedGroupedRMS,fwd}} = 10 \cdot n.
\]

**Arithmetic intensity (numeric example — Nemotron Mamba-2 output norm):**

Nemotron-style mixer: \(S{=}8192\), \(m{=}4096\) (\(64\) heads × \(64\) head dim), \(G{=}8\), \(r{=}512\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| \(n = S m\) | \(33{,}554{,}432\) |
| FLOPs (fused) | \(10n \approx 3.36 \times 10^8\) |
| Min HBM (fused, training) | read \(\mathbf y\) + read \(\mathbf z\): \(2n e \approx 64\,\mathrm{MiB}\); read \(\boldsymbol\gamma\): \(m e \approx 8\,\mathrm{KiB}\); write output: \(n e \approx 64\,\mathrm{MiB}\); write/save `group_rstd`: \(S G \cdot 4 \approx 256\,\mathrm{KiB}\) |
| AI (approx.) | \(\approx 3.36 \times 10^8 / (130 \times 2^{20}) \approx 2.5\,\mathrm{FLOP/byte}\) — **memory-bound** |

Unfused identity adds \(\sim 2n e\) peak from `silu_gate` + `gated` + `squared` + `normalized` temps (\(\sim 128\,\mathrm{MiB}\) elidable at this shape).

---

## Section 5: Backward FLOPs — step-by-step derivation

Training context (`requires_grad=True`). Inference-only → **backward FLOPs = 0**.

### 5.1 Saved vs recomputed

| Path | Saved for backward | Recomputed |
|------|-------------------|------------|
| HF eager | Autograd on `silu`, `pow`, `mean`, `rsqrt` chain | Partial |
| Fused leaf (target) | **`group_rstd`** fp32 \((\ldots,G,1)\); inputs **value**, **gate** (upstream SAVE) | SiLU \(\sigma(z)\); normalized grouped activations |
| Zepto fused recipe | `SAVE(group_rstd)` | SiLU factors from saved gate |

### 5.2 Fused region backward (kernel-accurate)

Let \(\widehat{\mathbf u}\) be post-grouped-RMS+γ, \(\mathbf g = \operatorname{SiLU}(\mathbf z)\), and \(\mathbf o = \widehat{\mathbf u}\) (output equals normalized gated tensor — no extra output multiply).

| Step | VJP / primitive | FLOPs/elem | Subtotal |
|------|-----------------|------------|----------|
| 1 | Grouped RMSNorm+γ VJP on gated input \(\mathbf u = \mathbf y \odot \mathbf g\) (saved `group_rstd`) | 4 | \(4n\) |
| 2 | `Multiply` VJP w.r.t. \(\mathbf y\): \(\partial L/\partial \mathbf y = \partial L/\partial \mathbf u \odot \mathbf g\) | 1 | \(n\) |
| 3 | `Multiply` VJP w.r.t. \(\mathbf g\): \(\partial L/\partial \mathbf g = \partial L/\partial \mathbf u \odot \mathbf y\) | 1 | \(n\) |
| 4 | SiLU VJP on \(\mathbf z\) (recompute \(\sigma\)) | 8 | \(8n\) |
| **Fused bwd total** | | | **\(14n\)** |

SiLU backward **\(8n\)** matches `region/silu`. Grouped RMS backward **\(4n\)** matches `region/rmsnorm` per-element recipe (reduction axis width \(r\) affects fan-in, not Zepto leaf coefficient).

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{GatedGroupedRMS,bwd}} =
\begin{cases}
14 \cdot n & \text{if } \texttt{requires\_grad=True} \\
0 & \text{otherwise}
\end{cases}
\]

**Paper-comparable:** backward may quote **\(4n\)** (grouped RMS only) — omit **\(10n\)** gate/VJP work when comparing.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

Let \(n = \lvert \mathbf y \rvert\), activation element size \(e\), \(S_{\mathrm{row}}\) positions, \(G\) groups.

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|-------------------|------------------|
| Identity lowering | `value_fp32`, `gate_fp32`, `sig`, `silu_gate`, `gated`, `grouped`, `squared`, `variance`, `denom`, `normalized`, `output` | \(\sim 3n e\) full-rank temps + output | — |
| Fused leaf (target) | `output`, `group_rstd` \((\ldots,G,1)\) fp32 | \(n e + S_{\mathrm{row}} G \cdot 4\) | `silu_gate`, `gated`, `squared`, `normalized` |
| Inference (`requires_grad=False`) | `output` only | \(n e\) | same |

`Reshape` views do not add HBM (`ALIAS` only).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes (order) | Zepto SAVE policy |
|------|---------------|-------|---------------|-------------------|
| HF eager autograd | intermediates on tape | various | \(\gg\) fused | implicit |
| Fused leaf | `group_rstd` | \((S_{\mathrm{row}}, G, 1)\) fp32 | \(4 S_{\mathrm{row}} G\) | **`SAVE(group_rstd)`** at this leaf |
| Upstream | `value`, `gate` | \((\ldots,m)\) | \(2n e\) | SAVE on producer edges / scan output |

Do **not** SAVE full-rank `gated` if `group_rstd` + inputs suffice for VJP (same policy as `region/gated_rms_norm`).

### 6.3 Resource event chains

**Identity lowering (unfused, forward):**

```
ALLOCATE(value_fp32) → ALLOCATE(gate_fp32) → ALLOCATE(sig)
  → ALLOCATE(silu_gate) → ALLOCATE(gated) → ALIAS(grouped_view)
  → ALLOCATE(squared) → ALLOCATE(variance) → ALLOCATE(denom)
  → ALLOCATE(normalized) → ALLOCATE(output)
  → SAVE(output)   # training: autograd retains gated path tensors
```

**Fused region leaf (training, target recipe):**

```
ALLOCATE(output) → ALLOCATE(group_rstd) → SAVE(group_rstd)
```

**Inference (`requires_grad=False`):**

```
ALLOCATE(output)
```

Backward (training): `RELEASE(group_rstd)` after grouped-RMS + gate VJPs complete.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF eager | `Zamba2RMSNormGated` | No | A | \(10n + 2SG\) identity | autograd | Full-rank temps | identity chain |
| Mamba mega-kernel | `mamba2_split_conv1d_scan_combined` | Yes | B | subsumed in block | subsumed | On-chip norm slice | do not bill **A** separately |
| FLA gated RMS | flash-linear-attention | Yes | A | **different math** (full axis) | **different** | `rstd` \((\ldots,1)\) | `region/gated_rms_norm` |
| Zepto module | `GatedGroupedRMSNorm` | Semantic | A | use fused leaf | use fused leaf | — | identity |
| Zepto stub | `region/gated_grouped_rms_norm` | TBI | A | **\(10n\)** | **\(14n\)** | `group_rstd` SAVE | stub |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: gated_grouped_rms_norm
recommended_region_ids:
  - id: region/gated_grouped_rms_norm/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - id: region/gated_grouped_rms_norm
    variant: default
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
  - id: region/gated_grouped_rms_norm/eager-hf
    variant: eager-hf
    hardware_gate: any
    fusion_boundary: A
    status: to_implement
pattern_rule:
  op_families:
    - Cast
    - Sigmoid
    - Multiply
    - Reshape
    - ReduceSum
    - Divide
    - Add
    - SquareRoot
    - ParameterScale
recipe:
  forward_flops: "10 * numel(value)"
  backward_flops: "14 * numel(value) if requires_grad else 0"
  forward_flops_per_element: 10
  backward_flops_per_element: 14
  materialize_group_rstd: true
  save_group_rstd: true
  elided_temps:
    - silu_gate
    - gated
    - squared
    - normalized
    - sigmoid_out
  saved_backward:
    - name: group_rstd
      shape: "(*prefix, num_groups, 1)"
      dtype: fp32
  resource_events_forward:
    - ALLOCATE(output)
    - ALLOCATE(group_rstd)
    - SAVE(group_rstd)
  resource_events_backward:
    - RELEASE(group_rstd)
  numerics_tags:
    - fp32_reduction
    - grouped_last_axis
    - silu_gate_before_norm
capabilities:
  - fused
priority: 6
mutually_exclusive_with:
  - region/gated_rms_norm
  - region/rmsnorm
  - region/silu
composition_notes:
  mamba2_mixer_stack:
    - region/linear
    - region/depthwise_causal_conv1d
    - region/mamba2_scan
    - region/gated_grouped_rms_norm
    - region/linear
  mega_fusion_boundary_B:
    kernel: mamba2_split_conv1d_scan_combined
    supersedes_standalone_norm_leaf: true
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Mamba-2 (structured state space): [Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)
- Nemotron 3 Nano architecture (hybrid Mamba-Transformer): [arXiv:2512.20848](https://arxiv.org/html/2512.20848v1)
- HF `Zamba2RMSNormGated` (grouped gated RMS reference): [modeling_zamba2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/zamba2/modeling_zamba2.py)
- HF Nemotron H Mamba-2 mixer (`norm_before_gate=False`, mega-kernel hook): [modeling_nemotron_h.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
- Mamba-2 combined Triton entrypoint: [modeling_mamba2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py)
- Zepto module contract: [step5-stateful-mixers.md §7.8](../docs/plans/step5-stateful-mixers.md)
- Zepto related leaf (distinct math): [gated-rms-norm.md](../docs/kernel/gated-rms-norm.md)
- Zepto Mamba scan composition: [mamba2-scan.md](../docs/kernel/mamba2-scan.md)

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 6 | `region/gated_grouped_rms_norm` | §8 | A | Nemotron / Mamba-2 mixer output norm; stub in `mixer_stub.py` blocks accurate Mamba-2 horizon costing until kernel-full implements lowering + recipe |

**Open gaps (§9 follow-up):**

- Dedicated CUDA Triton source line citation for **standalone** grouped gated RMS (outside mega-kernel) — eager PyTorch is authoritative for math; mega-kernel is boundary **B**.
- Optional `region/gated_grouped_rms_norm/decomposed` for graphs without `fused` capability (bill identity **\(10n + 2SG\)** FLOPs, not separate silu+rmsnorm regions).
