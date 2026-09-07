# Zepto kernel research: RMSNorm

**Date:** 2026-09-07
**Scope:** Standalone row normalization leaf (boundary A); hidden pre-norm and QK-Norm; prefill primary; training + inference; CUDA/ROCm/NPU (Liger), XPU/MPS hub variants
**Context:** Apertus-8B, bf16 primary (\(S{=}8192\), \(d{=}4096\), \(h{=}32\), \(h_{\mathrm{kv}}{=}8\), \(d_h{=}128\)); kernel-accurate Zepto leaf (\(4n\) forward/backward)

---

## Section 0: Mathematical definition

Zhang and Sennrich ([2019](https://arxiv.org/abs/1910.07467)) drop LayerNorm mean centering and keep a root-mean-square scale. HuggingFace Llama/Apertus (γ only, no β):

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \varepsilon}, \qquad
y_i = \gamma_i \cdot \frac{x_i}{\mathrm{RMS}(x)}.
\]

Equivalently, with per-row reciprocal scale \(\mathrm{rstd} = \mathrm{RMS}(x)^{-1}\):

\[
y = \gamma \odot (x \odot \mathrm{rstd}).
\]

**I/O shapes:**

| Use case | Input \(X\) | Output \(Y\) | Reduction axis | \(n = \lvert X \rvert\) |
|----------|-------------|--------------|----------------|------------------------|
| Hidden pre-norm | \((S, d)\) | \((S, d)\) | last (\(d\)) | \(Sd\) |
| QK-Norm (queries) | \((S, h, d_h)\) | same | last (\(d_h\)) | \(S h d_h\) |
| QK-Norm (keys) | \((S, h_{\mathrm{kv}}, d_h)\) | same | last (\(d_h\)) | \(S h_{\mathrm{kv}} d_h\) |

Apertus applies pre-norm around attention and MLP ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745); [Hernández-Cano et al., 2025, §2.1](https://arxiv.org/abs/2509.14233)). QK-Norm is the **same kernel** on headed tensors ([Henry et al., 2020](https://arxiv.org/abs/2010.04245); [Dehghani et al., 2023](https://arxiv.org/abs/2302.05449)).

**Numerics policies (bytes, not FLOPs):**
- HF eager (`LlamaRMSNorm` / `ApertusRMSNorm`): variance accumulation in **fp32**, cast back before ×γ ([`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)).
- Fused kernels (Liger, hub XPU/MPS): reduction in fp32 on-chip; output in activation dtype (bf16/fp16).
- \(\varepsilon\) default \(10^{-5}\) (Llama/Apertus).

**Zepto identity reference:** `src/zepto/modules/rms_norm.py` — 9-op chain: `Cast(fp32) → square → ReduceSum → /d → +ε → sqrt → /rms → Cast(act) → ×γ`.

**Structural vs eager:** Production backends fuse square + mean + normalize + scale into one launch; identity lowering materializes `squared`, `(S,1)` variance, and full-rank `normalized`.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | Liger **`LigerRMSNorm`** (Triton; CUDA/ROCm/NPU); hub **`kernels-community/rmsnorm`** (SYCL/ESIMD, XPU); **`kernels-community/mlx-rmsnorm`** (Metal, MPS); vLLM **`fused_add_rms_norm`** (residual + norm) |
| **Identity lowering** | Unfused semantic primitives | Zepto `RMSNorm`: 9-op HF fp32-variance chain; 9 `ALLOCATE`s; primitive FLOP sum \(4n + 2S\) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/rmsnorm/liger` (default CUDA/ROCm/NPU), `region/rmsnorm/hub-xpu`, `region/rmsnorm/hub-mps`, `region/rmsnorm/reference` — **\(4n\)** forward/backward; elides full-rank temps |

Default Zepto estimates use the **fused region leaf**, not the sum of identity primitives.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF eager Python** | [`LlamaRMSNorm`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | No | A | Any | Both (autograd on full chain) |
| **PyTorch native** | No dedicated `aten::rms_norm`; decomposed ops | No | A | Any | Both |
| **Liger Kernel** | [`LigerRMSNorm`](https://github.com/linkedin/Liger-Kernel) via `@use_kernel_forward_from_hub("RMSNorm")` → [`kernels-community/liger-kernels`](https://huggingface.co/kernels-community/liger-kernels) | Yes | A | CUDA, ROCm, NPU | Both; saves \(x\) + **`rstd`** fp32 |
| **HF Hub XPU** | [`kernels-community/rmsnorm`](https://huggingface.co/kernels-community/rmsnorm) SYCL/ESIMD | Yes | A | `device.type == "xpu"` | **Inference only** in [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py); autograd saves \(x, W, y, \mathrm{rstd}\) |
| **HF Hub MPS** | [`kernels-community/mlx-rmsnorm`](https://huggingface.co/kernels-community/mlx-rmsnorm) Metal (MLX lineage) | Yes | A | `device.type == "mps"` | **Inference only** in hub registry; forward writes output only — **`rstd` in threadgroup SLM** |
| **vLLM** | [`RMSNorm`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py), **`fused_add_rms_norm`** | Yes (+ residual) | A | CUDA (primary) | Inference/serving |
| **Megatron / TE** | Fused RMSNorm in Transformer Engine stacks | Yes | A | CUDA | Training |
| **Zepto (identity)** | `modules/rms_norm.py` | No | A | Any | Both |
| **Zepto (registered)** | `region/rmsnorm/{reference,liger,hub-xpu,hub-mps}` | Yes (cost leaf) | A | hardware-gated variants | Both |

Liger reports ~\(7\times\) lower kernel time and ~\(3\times\) lower **peak** memory vs HF eager at hidden 16384 by eliding full-rank temps and caching only `rstd` ([Dai et al., 2024](https://arxiv.org/abs/2410.10989)).

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **A** — standalone row normalization (same class as `region/softmax` boundary A, but over feature dim \(d\) not keys).

**Mutually exclusive (same module invocation):**
- Any `region/rmsnorm/*` variant vs the 9-op identity chain — fusion replaces the entire decomposed subgraph.
- `region/layernorm` vs `region/rmsnorm` — different math (\(5n\) vs \(4n\)); do not substitute.

**Composable:**
- Liger/hub RMSNorm patches compose with FlashAttention / Hub SDPA / Liger linear-CE per layer ([TRL kernels hub docs](https://huggingface.co/docs/trl/en/kernels_hub)).
- Pre-attn RMSNorm → RoPE → GQA is sequential; RMSNorm does not fuse into attention boundary C.
- Optional **`region/fused_add_rmsnorm`** (vLLM-style) would fuse residual add + RMSNorm — **not yet registered**; bills \(5n\) forward (\(4n\) norm + \(n\) add).

**Execution constraints:**
- Last-axis normalization; views flattened to `(B·T, H)` in hub kernels.
- Hub routing: CUDA/ROCm/NPU → Liger; XPU → `kernels-community/rmsnorm`; MPS → `mlx-rmsnorm`.
- XPU/MPS hub paths: **inference-only** in Transformers default registry; training falls back to eager.
- XPU ESIMD: hidden size often divisible by 32, \(\le 8192\) for tiled dispatch.
- MPS Metal: contiguous tensors; `axis_size > 4096` uses looped kernel variant.
- Not torch.compile-friendly (Liger / hub note in `hub_kernels.py`).

---

## Section 4: Forward FLOPs — step-by-step derivation

Let \(n = \lvert x \rvert\). Hidden RMSNorm: \(X \in \mathbb{R}^{S \times d}\), \(n = Sd\).

Per row of length \(d\): \(\mathrm{rstd} = \bigl(\frac{1}{d}\sum_i x_i^2 + \varepsilon\bigr)^{-1/2}\), \(y_i = \gamma_i x_i \mathrm{rstd}\).

### 4.1 Identity lowering (9-op HF eager chain)

| Step | Primitive | Tile shape | FLOPs rule | Subtotal |
|------|-----------|------------|------------|----------|
| 0–1 | `Cast` ×2 | \((S,d)\) | 0 (retype) | 0 |
| 2 | `Multiply` (square) | \((S,d)\) | 1/elem | \(Sd\) |
| 3 | `ReduceSum` | \((S,d)→(S,1)\) | \(Sd - S\) | \(Sd - S\) |
| 4 | `Divide` (/d) | \((S,1)\) | 1/elem | \(S\) |
| 5 | `Add` (+ε) | \((S,1)\) | 1/elem | \(S\) |
| 6 | `SquareRoot` | \((S,1)\) | 1/elem | \(S\) |
| 7 | `Divide` (normalize) | \((S,d)\) | 1/elem | \(Sd\) |
| 8 | `ParameterScale` (×γ) | \((S,d)\) | 1/elem | \(Sd\) |
| **Identity total** | | | | **\(4Sd + 2S\)** |

Mean billing: `ReduceSum` (\(Sd-S\)) + `/d` (\(S\)) = \(Sd\) — do **not** double-count.

Casts bill **0 FLOPs**. Oracle: `tests/test_rmsnorm_identity_lowering.py`.

### 4.2 Fused region leaf (kernel-accurate)

Four leading stages, one FLOP per element of \(X\) ([`RMSNormRecipe.forward_flops_per_element = 4`](../../src/zepto/analysis/lowering/recipes/rmsnorm.py)):

| Stage | Arithmetic | Per row | For \(X\) |
|-------|------------|---------|-----------|
| Square | \(x_i^2\) | \(d\) muls | \(Sd\) |
| Mean | \(\sum x_i^2\) then \(\times 1/d\) | \(d\) (reduction + divide) | \(Sd\) |
| Normalize | \(x_i \cdot \mathrm{rstd}\) | \(d\) muls | \(Sd\) |
| Scale | \(y_i \cdot \gamma_i\) | \(d\) muls | \(Sd\) |
| **Fused leaf total** | | \(4d\) | **\(4Sd = 4n\)** |

Row-scalar work (`+ε`, `sqrt`/`rsqrt`) is \(O(S)\) and **dropped** by the fused leaf — not analogous to softmax row-max (\(\Theta(hS^2)\), §10 of kernel-implementation).

QK-Norm closed form (same kernel, last axis):

\[
\mathrm{FLOPs}_{\mathrm{QK\text{-}Norm}} = 4 \cdot S \cdot (h + h_{\mathrm{kv}}) \cdot d_h.
\]

### 4.3 Paper-comparable (Appendix E / Zhang–Sennrich)

Appendix E counts the same **\(4n\)** γ-only RMSNorm leaf for Apertus pre-norm layers. No separate paper-comparable formula differs from the kernel-accurate leaf for RMSNorm (unlike softmax \(3hS^2\) paper vs \(5hS^2\) stable leaf).

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = 4 \cdot \lvert x \rvert = 4n.
\]

**Arithmetic intensity (numeric example):**

Apertus-8B, one hidden RMSNorm, \(S{=}8192\), \(d{=}4096\), bf16 (\(e{=}2\)):

- FLOPs: \(4 \times 8192 \times 4096 = 134{,}217{,}728 \approx 1.34 \times 10^8\)
- Minimum HBM traffic (fused Liger): read \(x\) (\(64\,\mathrm{MiB}\)) + read \(\gamma\) (\(8\,\mathrm{KiB}\)) + write \(y\) (\(64\,\mathrm{MiB}\)) + write `rstd` (\(S \times 4\) B fp32 \(\approx 32\,\mathrm{KiB}\)) \(\approx 128\,\mathrm{MiB}\)
- AI \(\approx 1.34 \times 10^8 / (128 \times 2^{20}) \approx 1.0\,\mathrm{FLOP/byte}\) — **memory-bound**

Unfused identity path adds \(\sim 1\text{–}2 \times n e\) peak from `squared` + `normalized` temps (\(\sim 128\,\mathrm{MiB}\) elidable per norm). Two norms per decoder block (pre-attn + pre-FFN) \(\approx 256\,\mathrm{MiB}\) elidable peak if counted simultaneously without fusion.

---

## Section 5: Backward FLOPs — step-by-step derivation

Training context with `requires_grad=True`. Fused backends save **\(x\)** (input) and **`rstd`** \((\ldots, 1)\) fp32 per row (Liger / hub-XPU convention). MPS hub inference: `requires_grad=False` → **backward FLOPs = 0**.

### 5.1 Saved vs recomputed

| Path | Saved for backward | Recomputed in backward |
|------|-------------------|------------------------|
| HF eager | Multiple chain activations + \(x\) | Partial |
| Liger / hub-XPU | \(x\), **`rstd`** | Output \(y\) not saved (Liger) |
| Hub XPU autograd | \(x\), \(W\), \(y\), **`rstd`** | — (heavier; hub inference-only today) |
| Hub MPS (if autograd) | \(x\) (+ \(W\)) only | **`rstd`** recomputed in VJP Metal kernel |
| Zepto fused leaf | **`rstd`** via `SAVE` event | Matches Liger policy; hub-mps variant omits `rstd` SAVE |

### 5.2 Fused region backward (kernel-accurate, \(4n\))

With saved \(\mathrm{rstd}\), backward mirrors forward structure ([`RMSNormRecipe.backward_flops_per_element = 4`](../../src/zepto/analysis/lowering/recipes/rmsnorm.py)):

| Step | Primitive | Per element | Subtotal |
|------|-----------|-------------|----------|
| 1 | \(\partial L/\partial y \cdot \gamma \cdot \mathrm{rstd}\) | 1 mul | \(n\) |
| 2 | Row reduction for \(\partial L/\partial \mathrm{rstd}\) | 1 op/elem contribution | \(n\) |
| 3 | \(\partial L/\partial x_i\) from normalize chain | 1 mul + 1 add | \(n\) |
| 4 | \(\partial L/\partial \gamma_i = (\partial L/\partial y_i) \cdot x_i \cdot \mathrm{rstd}\) | 1 mul | \(n\) |
| **Fused bwd total** | | **4/elem** | **\(4n\)** |

Hub-MPS recompute path would add \(\sim 2n\) for square + mean in backward; Zepto **`region/rmsnorm/hub-mps`** still bills \(4n\) when `requires_grad=True` for parity with the registered recipe, but real MPS hub defaults to inference (\(0\) backward).

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

Let \(n = \lvert x \rvert\), activation element size \(e\) (bf16: \(e{=}2\)).

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|-------------------|------------------|
| Identity (9-op) | `x_fp32`, `squared`, `variance`, `denom`, `normalized`, `normalized_act`, `y` | \(\sim 2n e\) full-rank temps + output | — |
| Fused Liger / reference | `y`, `rstd` \((S,1)\) fp32 | \(n e + S \cdot 4\) | `squared`, `normalized`, fp32 cast temps |
| Fused hub-MPS | `y` only (inference) | \(n e\) | same + no HBM `rstd` |
| vLLM fused add+norm | `y` (+ updated residual in-place) | \(n e\) | separate residual read elided |

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| Identity eager | chain intermediates + \(x\) | various | \(\gg n e\) | per-op SAVE on unfused chain |
| Liger / reference / hub-XPU | **`rstd`** | `(*batch, 1)` fp32 | \(S \cdot 4\) per norm | `SAVE rstd` |
| Hub XPU autograd | \(x, W, y, \mathrm{rstd}\) | full | \(\sim 3n e + S \cdot 4\) | heavier than Liger |
| Hub MPS inference | none (no grad) | — | 0 | no SAVE events |
| Zepto hub-mps variant | none | — | 0 | `materialize_rstd=False`, `save_rstd=False` |

Persistent weights: \(\gamma\) shape \((d,)\) — `PERSIST`, not counted in forward-lived peak.

### 6.3 Resource event chains

**Identity lowering (unfused):**
```
ALLOCATE(x_fp32) → ALLOCATE(squared) → ALLOCATE(variance) → ALLOCATE(denom)
  → ALLOCATE(normalized) → ALLOCATE(normalized_act) → ALLOCATE(y)
  → SAVE(x_fp32) → SAVE(squared) → SAVE(variance) → … (per-op autograd)
```

**Fused region leaf (Liger / reference / hub-XPU):**
```
ALLOCATE(y) → ALLOCATE(rstd) → SAVE(rstd)
```

**Fused region leaf (hub-MPS inference):**
```
ALLOCATE(y)
```

**Apertus-8B trace (one hidden RMSNorm, fused Liger, training):**
- Input \(x\): \(8192 \times 4096 \times 2 = 64\,\mathrm{MiB}\)
- Output \(y\): \(64\,\mathrm{MiB}\)
- Saved `rstd`: \(8192 \times 4 = 32\,\mathrm{KiB}\)
- Elided vs identity: \(\sim 128\,\mathrm{MiB}\) (`squared` + `normalized` not allocated)
- Decoder block: **4** RMSNorm regions (pre-attn, Q-norm, K-norm, pre-FFN) when `qk_norm=True` — see `tests/lowering/regions/test_rmsnorm.py`

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF eager | Transformers | No | A | \(4n + 2S\) | identity chain | Full-rank temps | identity 9-op |
| Liger | liger-kernels / hub | Yes | A | \(4n\) | \(4n\) | On-chip; SAVE `rstd` | `region/rmsnorm/liger` |
| Hub XPU | kernels-community/rmsnorm | Yes | A | \(4n\) | \(4n\) | ESIMD SLM; SAVE `rstd` (+ \(y\) in autograd) | `region/rmsnorm/hub-xpu` |
| Hub MPS | mlx-rmsnorm | Yes | A | \(4n\) | 0 (inference) / recompute | SLM `inv_mean`; no HBM `rstd` | `region/rmsnorm/hub-mps` |
| vLLM | vllm layernorm | Yes (+add) | A | \(5n\) | — | Fused residual | *(optional, not registered)* |
| LayerNorm | Zepto `region/layernorm` | Yes | A | \(5n\) | \(5n\) | SAVE `mean` + `inv_std` | distinct region |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: rmsnorm
recommended_region_ids:
  - id: region/rmsnorm/liger
    variant: liger
    hardware_gate: exclude_xpu_mps
    fusion_boundary: A
    status: registered
  - id: region/rmsnorm/hub-xpu
    variant: hub-xpu
    hardware_gate: xpu_only
    fusion_boundary: A
    status: registered
  - id: region/rmsnorm/hub-mps
    variant: hub-mps
    hardware_gate: mps_only
    fusion_boundary: A
    status: registered
  - id: region/rmsnorm/reference
    variant: reference
    hardware_gate: any
    fusion_boundary: A
    status: registered
pattern_rule:
  op_families:
    - cast
    - multiply
    - reduce_sum
    - divide
    - add
    - square_root
    - divide
    - cast
    - parameter_scale
recipe:
  forward_flops: "4 * numel(input)"
  backward_flops: "4 * numel(input) if requires_grad else 0"
  forward_flops_per_element: 4
  backward_flops_per_element: 4
  materialize_rstd: true
  save_rstd: true
  elided_temps:
    - squared
    - normalized
    - x_fp32_cast_temp
  saved_backward:
    - name: rstd
      shape: "(*batch_dims, 1)"
      dtype: fp32
  resource_events_forward:
    - "ALLOCATE(y)"
    - "ALLOCATE(rstd)"
    - "SAVE(rstd)"
  resource_events_forward_hub_mps:
    - "ALLOCATE(y)"
  numerics_tags:
    - fp32_variance_reduction
    - gamma_only_affine
capabilities:
  - fused
priority: 8
hub_mps_variant:
  materialize_rstd: false
  save_rstd: false
  resource_events_forward:
    - "ALLOCATE(y)"
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Zhang & Sennrich, RMSNorm: [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Xiong et al., pre-norm: [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Henry et al., QK-Norm: [arXiv:2010.04245](https://arxiv.org/abs/2010.04245)
- Dehghani et al., QK-Norm in ViT: [arXiv:2302.05449](https://arxiv.org/abs/2302.05449)
- Hernández-Cano et al., Apertus: [arXiv:2509.14233](https://arxiv.org/abs/2509.14233)
- Dai et al., Liger-Kernel (RMSNorm §): [arXiv:2410.10989](https://arxiv.org/abs/2410.10989)
- HF Transformers `LlamaRMSNorm`: [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- HF `hub_kernels.py` device map: [integrations/hub_kernels.py](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)
- Liger hub: [kernels-community/liger-kernels](https://huggingface.co/kernels-community/liger-kernels)
- Hub XPU RMSNorm: [kernels-community/rmsnorm](https://huggingface.co/kernels-community/rmsnorm)
- Hub MPS RMSNorm: [kernels-community/mlx-rmsnorm](https://huggingface.co/kernels-community/mlx-rmsnorm); MLX reference [rms_norm.metal](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal)
- vLLM layernorm: [vllm/model_executor/layers/layernorm.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py)
- Zepto module: `src/zepto/modules/rms_norm.py`
- Zepto regions: `src/zepto/analysis/lowering/implementations/regions/rmsnorm/`
- Zepto tests: `tests/test_rmsnorm_identity_lowering.py`, `tests/lowering/regions/test_rmsnorm.py`
- Domain authority: `docs/kernel-implementation.md` §5

**Verification date:** 2026-09-07 (cross-checked against registered Zepto `region/rmsnorm/*` implementations and `docs/kernel-implementation.md` §5)

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| P3 (optional) | `region/fused_add_rmsnorm` | §3 composable | A | vLLM `fused_add_rms_norm` residual fusion; bills \(5n\) forward, elides separate residual read |

Core `region/rmsnorm/*` variants are **registered** — no P1 backlog item.
