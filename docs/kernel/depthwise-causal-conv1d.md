# Zepto kernel research: Depthwise causal Conv1D (+ optional SiLU)

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Standalone depthwise causal 1-D convolution leaf with optional SiLU; prefill (\(S>1\)) and decode (\(S=1\)) with rolling \((C,K)\) conv state; training + inference; CUDA primary (Dao `causal-conv1d`), HF fallbacks
**Context:** Local filter in Mamba-2 (`Mamba2Mixer`), Qwen3-Next Gated DeltaNet (`GatedDeltaNet`), and related SSM mixers. Zepto module `DepthwiseCausalConv1d` is registered; fused region `region/depthwise_causal_conv1d` is a **stub** (`mixer_stub.py`). Default cost leaf should be **kernel-accurate** fused conv+SiLU when `activation="silu"`.

---

## Section 0: Mathematical definition

**Depthwise causal 1-D convolution** applies an independent length-\(K\) FIR filter along the time axis for each channel \(c \in \{1,\ldots,C\}\). There is **no** mixing across channels (\(\mathrm{groups}=C\) in PyTorch terms).

Let input \(x_{t,c}\) for \(t \in \{0,\ldots,S-1\}\) (Zepto rank-2 layout \((S,C)\); production stacks often use \((B,C,L)\) or \((B,L,C)\)). Zero-extend past samples \(x_{u,c}=0\) for \(u<0\) (left padding). With cross-correlation convention matching PyTorch `conv1d` + `padding=K-1` then truncate to length \(S\):

\[
a_{t,c} = b_c + \sum_{j=0}^{K-1} w_{c,j}\, x_{t-K+1+j,\,c},
\qquad
y_{t,c} = \phi(a_{t,c}),
\]

where \(W \in \mathbb{R}^{C \times K}\) (`weight[c,j]`), optional bias \(b \in \mathbb{R}^C\), and \(\phi \in \{\mathrm{id}, \mathrm{SiLU}\}\). Mamba / DeltaNet checkpoints use **SiLU** (same as Swish-1) on the conv output ([Dao-AILab `causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d); [Mamba2 module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)).

**Causality:** \(y_{t,c}\) depends only on \(x_{u,c}\) with \(u \le t\).

**I/O shapes (Zepto module):**

| Mode | `value` | `conv_state_in` | `output` | `conv_state_out` |
|------|---------|-----------------|----------|------------------|
| Prefill | \((S, C)\) | `None` | \((S, C)\) | \((C, K)\) — last \(K\) **raw input** samples per channel, left-padded if \(S<K\) |
| Decode | \((1, C)\) | \((C, K)\) | \((1, C)\) | \((C, K)\) — shifted buffer + new sample |

Persistent weights: \(K\) lag vectors `weight_j` each \((C,)\) — equivalent to checkpoint tensor \((C,1,K)\) ([`depthwise_causal_conv1d.py`](../src/zepto/modules/depthwise_causal_conv1d.py), [step5 plan](../docs/plans/step5-stateful-mixers.md)).

**Numerics policies:**

- Conv accumulations bill as **elementwise** `Multiply` + `Add` on \(C\)-vectors (not a single GEMM).
- SiLU uses Zepto special-function bucket: **5 FLOPs/element** forward, **8** backward ([`docs/kernel/silu.md`](../docs/kernel/silu.md)).
- `causal-conv1d` CUDA supports fp32, fp16, bf16; kernel widths **2, 3, 4** only ([repo README](https://github.com/Dao-AILab/causal-conv1d)).
- HF Transformers Mamba2 registers Hub fallbacks: `causal_conv1d_fn` (prefill) and `causal_conv1d_update` (decode), else `F.conv1d` eager path ([`modeling_mamba2.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py)).

**Structural vs eager (Zepto identity reference):**

1. Build history: prefill `Concat` zero pad \((K{-}1, C)\); decode roll \((C,K)\) state + new token.
2. For each output time \(t\): gather \(K\) samples → \(K\) `ParameterScale` → \((K{-}1)\) `Add` → optional bias → optional inline `Sigmoid`+`Multiply` (SiLU).
3. State tensor stores **pre-activation inputs**, not \(y\).

Production **fused** path: one kernel for depthwise causal conv ± SiLU; optional **mega-fusion** `mamba2_split_conv1d_scan_combined` absorbs conv + selective scan (boundary **B** for mixer stack, not this leaf alone).

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | [`causal_conv1d_fn`](https://github.com/Dao-AILab/causal-conv1d) / `causal_conv1d_update` (conv ± SiLU in CUDA); optional [`mamba2_split_conv1d_scan_combined`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) when entire Mamba-2 inner block is fused |
| **Identity lowering** | Unfused semantic chain | Zepto `DepthwiseCausalConv1d`: per-\(t\) `Split` window → `ParameterScale`×\(K\) → `Add` tree → optional SiLU; horizon allocates `Conv1DState` \((C,K)\) ([`state.py`](../src/zepto/analysis/horizon/state.py)) |
| **Zepto fused region** | Kernel-accurate cost leaf | `region/depthwise_causal_conv1d` (stub today) — closed-form **\(f(K,\phi)\cdot S\cdot C\)** forward FLOPs; elides per-\(t\) HBM temps; `PERSIST`/`SAVE` conv state per §6 |

**Default estimates should use the fused region leaf when `requested_capabilities` includes `fused`, not the sum of Split/Concat graph overhead (graph loop is compile-time structure, not extra arithmetic).**

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **Dao causal-conv1d** | [`pip install causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d) | Yes (conv ± act) | A (standalone leaf) | CUDA; AMD notes in README | Both; autograd via CUDA bwd |
| **PyTorch eager fallback** | `F.conv1d(..., groups=C, padding=K-1)[:,:,:S]` + `silu` | Partial (2 kernels) | A | Any | Both |
| **HF Transformers Mamba2** | `@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn")` / `causal_conv1d_update` | Yes when Hub/CUDA ext loaded | A | CUDA + Hub | Prefill vs decode paths |
| **Mamba-SSM reference** | `mamba_ssm.modules.mamba2` imports `causal_conv1d_fn`, `causal_conv1d_update` | Yes | A | CUDA | Both |
| **Mamba mega-kernel** | `mamba2_split_conv1d_scan_combined` (Hub `mamba_ssm`) | Yes (conv+scan+proj) | **B** (mixer sub-block) | CUDA | Training/inference when enabled |
| **Zepto module (identity)** | [`DepthwiseCausalConv1d`](../src/zepto/modules/depthwise_causal_conv1d.py) | No (explicit loop) | A | Any | Both; emits state ports |
| **Zepto region (stub)** | `region/depthwise_causal_conv1d` in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py) | Cost leaf TBD | A | `fused` capability gate | `status: to_implement` |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **A** — standalone depthwise conv leaf (same class as `region/silu`, `region/rmsnorm`). Not an attention softmax boundary.

**Mutually exclusive (same mixer conv slot):**

- `region/depthwise_causal_conv1d/*` vs **`mamba2_split_conv1d_scan_combined`** / full `region/mamba2_mixer` fused envelope — mega-kernel replaces separate conv + scan FLOP/memory leaves on that layer.
- Identity `DepthwiseCausalConv1d` decomposed graph vs fused `region/depthwise_causal_conv1d` — region wins when pattern + `fused` capability match (future priority > identity).

**Composable:**

- **Mamba2Mixer:** `Linear → DepthwiseCausalConv1d → Split → SelectiveSSMScan → GatedGroupedRMSNorm → Linear` — conv region bills only the conv+SiLU slice; scan is `region/mamba2_scan` (separate stub).
- **GatedDeltaNet:** `Linear → DepthwiseCausalConv1d → L2Normalize → GatedDeltaScan → …` — conv channels \(C = 2 d_{qk} + d_v\) ([`GatedDeltaNetConfig`](../src/zepto/modules/mixer_config.py)).
- SiLU inside conv is **inlined** in the module (not a separate `region/silu` on the same tensor) — avoids double-counting gate activation.

**Execution constraints:**

- Static \(K \in \{2,3,4\}\) for CUDA kernel; Zepto module allows general positive \(K\) in reference graph.
- Decode requires `conv_state_in.shape == (C,K)`; horizon `conv_state:{layer}` **PERSIST** across decode steps (related to gap **G3** KV/recurrent accounting).
- `requires_grad=False` → backward FLOPs = 0; no SAVE of pre-activations.
- Batch dimension: Zepto reference is \((S,C)\); scale \(B\) by multiplying FLOPs/bytes by batch when extending to \((B,S,C)\).

---

## Section 4: Forward FLOPs — step-by-step derivation

**Symbols:** sequence length \(S\), channels \(C\), kernel width \(K\), output elements \(n = S \cdot C\).

Define per-output-element conv core (one channel, one time index):

\[
f_{\mathrm{conv}} = K \cdot (\text{mul}) + (K-1) \cdot (\text{add}) + \mathbb{1}_{\mathrm{bias}} \cdot 1
= 2K - 1 + \mathbb{1}_{\mathrm{bias}}.
\]

SiLU add-on (when `activation="silu"`): **\(+5\)** per output element ([SiLU leaf](../docs/kernel/silu.md)).

\[
f_{\mathrm{fwd,elem}} =
\begin{cases}
2K - 1 + \mathbb{1}_{\mathrm{bias}} & \phi = \mathrm{id} \\
2K - 1 + \mathbb{1}_{\mathrm{bias}} + 5 & \phi = \mathrm{SiLU}
\end{cases}
\]

For default **\(K=4\)**, no bias (Qwen DeltaNet): \(f = 4 + 3 + 5 = 12\).  
For **\(K=4\)**, with bias (Nemotron Mamba2): \(f = 13\).

### 4.1 Identity lowering (Zepto decomposed per time step)

Per output index \((t,c)\) the module emits the same arithmetic as the fused leaf; extra `Split`/`Concat`/`Reshape`/`Transpose` ops affect **graph size** and optional temp **memory**, not the conv+SiLU FLOP count below.

| Step | Primitive | Per \((t,c)\) | FLOPs | Subtotal over all outputs |
|------|-----------|---------------|-------|---------------------------|
| 1 | `ParameterScale` × \(K\) (lag weights) | mul × \(K\) | \(K\) | \(K n\) |
| 2 | `Add` tree over \(K\) terms | add × \((K{-}1)\) | \(K{-}1\) | \((K{-}1)n\) |
| 3 | `ParameterBias` (optional) | add × 1 | 1 | \(\mathbb{1}_{\mathrm{bias}}\, n\) |
| 4 | SiLU: `Sigmoid` + `Multiply` (if enabled) | special + mul | 5 | \(5n\) if SiLU |
| **Identity total** | | | | \(f_{\mathrm{fwd,elem}} \cdot n\) |

### 4.2 Fused region leaf (kernel-accurate)

| Step | Primitive | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Fused depthwise MAC over window | \(K\) mul + \((K{-}1)\) add | \(2K{-}1\) | \((2K{-}1)n\) |
| 2 | Bias + SiLU in registers (optional) | \(1 + 5\) | up to 6 | \((\mathbb{1}_{\mathrm{bias}} + 5\mathbb{1}_{\mathrm{SiLU}}) n\) |

Fused forward FLOPs match identity arithmetic; fusion removes HBM-resident per-lag partial sums.

### 4.3 Paper-comparable (if different)

Appendix-style SSM papers often report **parameter counts** and **scan** complexity; the local conv is \(O(S \cdot C \cdot K)\) with tiny \(K\). No cheaper asymptotic formula — **paper-comparable equals kernel-accurate** for this leaf. Mega-fusion (`mamba2_split_conv1d_scan_combined`) changes **where** conv FLOPs are billed (inside boundary **B**), not the scalar \(f_{\mathrm{fwd,elem}}\).

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{fwd}} = f_{\mathrm{fwd,elem}} \cdot S \cdot C.
\]

**Decode** (\(S=1\)): \(\mathrm{FLOPs}_{\mathrm{fwd,decode}} = f_{\mathrm{fwd,elem}} \cdot C\) per layer per generated token (state update is memory traffic, §6).

**Arithmetic intensity (numeric example — Nemotron-style Mamba2 conv slice):**

| Quantity | Value |
|----------|-------|
| \(C = d_{\mathrm{ssm}} + 2 g n_{\mathrm{state}}\) | e.g. \(8192 + 2\cdot8\cdot128 = 10240\) |
| \(K\) | 4 |
| \(S\) | 8192 |
| \(f_{\mathrm{fwd,elem}}\) (SiLU + bias) | 13 |
| FLOPs | \(13 \cdot S \cdot C \approx 1.10 \times 10^{9}\) per layer |
| Min HBM (fused: read \(x\) + write \(y\), bf16 \(e=2\)) | \(2 S C e \approx 335\) MiB |
| Arithmetic intensity | \(\approx 3.3\) FLOP/byte — **memory-bound** |

At decode \(S=1\): \(\approx 1.33 \times 10^{5}\) FLOPs/layer/token — negligible vs scan/GQA layers; **state R/W** dominates bytes.

---

## Section 5: Backward FLOPs — step-by-step derivation

Training assumes SiLU on conv output; autograd saves **pre-activation** \(a\) (conv output before \(\phi\)) or recomputes from \(x,w\) depending on backend. Zepto policy: **SAVE(pre_activation)** when `requires_grad=True` and activation enabled.

**Conv backward** (per element, depthwise): gradient w.r.t. inputs and weights each mirror the forward MAC pattern → **\(\approx 2(2K-1)\)** FLOPs per output element for the conv core (input grad + weight grad contributions, elementwise depthwise).

**SiLU backward** with saved \(a\): **8 FLOPs/element** ([SiLU §5](../docs/kernel/silu.md)).

| Step | Phase | Per element | FLOPs |
|------|-------|-------------|-------|
| 1 | Conv input grad | MAC-like | \(2K-1\) |
| 2 | Conv weight grad | MAC-like | \(2K-1\) |
| 3 | SiLU backward | recompute + chain | 8 |
| **Total (SiLU, bias doesn't change bwd order)** | | | \(4K - 2 + 8 = 4K + 6\) |

For \(K=4\): **\(4K + 6 = 22\)** FLOPs per \((t,c)\).

**Closed form (training, SiLU):**
\[
\mathrm{FLOPs}_{\mathrm{bwd}} = (4K + 6) \cdot S \cdot C.
\]

**Inference / `requires_grad=False`:** backward FLOPs = **0**; no SAVE events.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

Let element size \(e\) (bf16: \(e=2\)).

### 6.1 Transient allocations (forward)

| Path | Live buffers | Peak bytes (order) | Elided by fusion |
|------|--------------|--------------------|------------------|
| Identity prefill | Extended history \((S{+}K{-}1,C)\), optional per-\(t\) partials | \(\Theta(S C e)\) + graph temps | Fused conv kernel |
| Fused `causal_conv1d_fn` | Input + output \((S,C)\) | \(2 S C e\) | History, partial sums |
| Decode update | State \((C,K)\), 1-step I/O | \((C K + 2C) e\) | Concat scratch (in-place shift in CUDA) |
| Weights | \(W \in \mathbb{R}^{C \times K}\) | \(C K e_w\) | — (persistent) |

**Persistent conv state (inference):** \(C K e\) per layer via `conv_state:{i}` — not per-token transient.

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| Fused Dao + SiLU | Pre-activation \(a\) | \((S,C)\) | \(S C e\) | `save_pre_activation: true` when SiLU |
| Fused Dao, no act | Input \(x\) or \(a\) | \((S,C)\) | \(S C e\) | `save_input: true` |
| Identity Zepto | Same as fused leaf target | \((S,C)\) | \(S C e\) | Match fused policy |

Weights \(W\) (and bias) are **PERSIST** parameters, not SAVE activations.

### 6.3 Resource event chains

**Identity lowering (unfused prefill, SiLU):**
```
PERSIST(W) → PERSIST(b?) → ALLOCATE(x) → ALLOCATE(extended_history) →
  loop t: ALLOCATE(window_t) → ... → ALLOCATE(y_t) →
ALLOCATE(y) → ALLOCATE(conv_state_out) → SAVE(pre_activation)  # training
```

**Fused region leaf (`causal_conv1d_fn` + SiLU):**
```
PERSIST(W) → PERSIST(b?) → ALLOCATE(x) → ALLOCATE(y) →
SAVE(pre_activation)   # training; omit SAVE when requires_grad=False
```

**Decode step (`causal_conv1d_update`):**
```
PERSIST(conv_state) → ALIAS(x_1) → ALLOCATE(y_1) →
RELEASE/UPDATE conv_state (in-place roll) → PERSIST(conv_state)
```

SRAM/SLM partial windows inside CUDA tiles → **no** `ALLOCATE` events.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Dao CUDA | `causal-conv1d` | Yes | A | \(f_{\mathrm{fwd,elem}} n\) | \((4K{+}6)n\) (SiLU) | No full history | `region/depthwise_causal_conv1d/cuda` (TBI) |
| PyTorch fallback | ATen | Partial | A | same | same | `conv1d` temp | identity |
| HF Hub | `kernels-community` routes | Yes | A | same | same | stateful update API | variant id TBI |
| Mamba combined | `mamba_ssm` Hub | Yes | B | bundled | bundled | fused w/ scan | `region/mamba2_mixer` |
| Zepto today | module only | No | A | same | same | explicit state | stub |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: depthwise_causal_conv1d
recommended_region_ids:
  - id: region/depthwise_causal_conv1d/decomposed
    variant: zepto-module
    hardware_gate: any
    fusion_boundary: A
    status: registered
  - id: region/depthwise_causal_conv1d/cuda
    variant: causal-conv1d
    hardware_gate: cuda
    fusion_boundary: A
    status: to_implement
  - id: region/depthwise_causal_conv1d/hub
    variant: hf-hub-causal-conv1d
    hardware_gate: cuda
    fusion_boundary: A
    status: to_implement
pattern_rule:
  op_families: [Concat, Split, ParameterScale, Add, Sigmoid, Multiply]
recipe:
  forward_flops: "f_fwd(K, bias, activation) * S * C; f_fwd = (2*K - 1 + bias + 5*silu) per output element"
  backward_flops: "(4*K + 6) * S * C when activation=silu and requires_grad; else 0"
  forward_flops_per_element: 12  # default K=4, silu, no bias
  backward_flops_per_element: 22  # K=4, silu
  materialize_history: false
  save_pre_activation: true
  save_input: false
  elided_temps: [extended_history, per_timestep_window, partial_sums]
  saved_backward:
    - name: pre_activation
      shape: "(S, C)"
  resource_events_forward:
    - "PERSIST(weight)"
    - "ALLOCATE(x)"
    - "ALLOCATE(y)"
    - "ALLOCATE(conv_state_out) on prefill boundary"
    - "PERSIST(conv_state) on decode"
  resource_events_backward:
    - "SAVE(pre_activation)"
  numerics_tags: [depthwise, causal_structural, silu_inline]
capabilities: [fused]
priority: 3
variants:
  - name: mamba2_combined
    id: region/mamba2_mixer
    fusion_boundary: B
    note: "Mutually exclusive with standalone conv leaf on same mixer forward"
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Dao-AILab **causal-conv1d** CUDA depthwise causal conv (width 2–4, optional SiLU): https://github.com/Dao-AILab/causal-conv1d
- **Mamba-2** module using `causal_conv1d_fn` / `causal_conv1d_update`: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
- **HuggingFace Transformers** Mamba2 Hub fallbacks and eager `F.conv1d` paths: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py
- Zepto **DepthwiseCausalConv1d** reference decomposition: `src/zepto/modules/depthwise_causal_conv1d.py`
- Step 5 mixer plan (state geometry, math): `docs/plans/step5-stateful-mixers.md`
- SiLU FLOP conventions (inlined activation): `docs/kernel/silu.md`

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 3 | `region/depthwise_causal_conv1d/cuda` | §8 | A | Replace `mixer_stub` with recipe matching Dao FLOPs + SAVE policy; wire provenance `prov-depthwise-causal-conv1d` |
| 3 | `region/depthwise_causal_conv1d/hub` | §8 | A | Mirror HF `@use_kernel_func_from_hub_with_fallback` routing for cost parity |
| 2 | `region/mamba2_mixer` | §3, §7 | B | Optional mega-kernel mutual exclusion vs standalone conv+scan |

**Open gaps:** Horizon recurrent byte accounting (**G3**) for `conv_state:{layer}` across long decode; batch>1 rank-3 layout not yet in reference module; non-CUDA backends use PyTorch fallback memory (two-kernel) — separate variant recipes recommended when implemented.
