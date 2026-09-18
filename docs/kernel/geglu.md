# Zepto kernel research: GeGLU (GELU-gated FFN)

**Date:** 2026-09-15
**Proposer:** Robin Girardin
**Scope:** Full gated MLP stack (`gate_proj → GELU → gate×up → down_proj`); prefill + decode; training + inference; CUDA primary; Gemma / BERT-GeGLU / ViT-style FFNs
**Context:** GeGLU is the GELU variant of the GLU family ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)), used in Gemma (`gelu_pytorch_tanh`), Muse-Glimmer, and other stacks that gate the FFN with GELU instead of SiLU. Apertus-8B uses xIELU in an ungated two-linear FFN; this report targets gated GeGLU stacks for cross-model cost modeling. Default Zepto estimates use **decomposed leaves** (`region/linear` × 3 + `region/gelu/*` + `Multiply`) until `region/geglu/*` regions are registered (see §8).

---

## Section 0: Mathematical definition

GeGLU (GELU-Gated Linear Unit) applies GELU to the gate branch and multiplies elementwise with the up projection:

\[
\mathrm{GeGLU}(x) = W_d \,\Bigl(\mathrm{GELU}(W_g x) \odot (W_u x)\Bigr),
\]

where \(\mathrm{GELU}\) is the Gaussian Error Linear Unit ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)) and \(\odot\) is elementwise multiplication.

**GELU approximation variants on the gate branch** (same I/O shape; pick one per model):

| Variant | Formula | Typical use |
|---------|---------|-------------|
| **Tanh (GPT/NewGELU)** | \(\tfrac{t}{2}\left(1 + \tanh\!\left(\sqrt{2/\pi}\,(t + 0.044715\,t^3)\right)\right)\) | Gemma `hidden_act="gelu_pytorch_tanh"`, GPT-style |
| **Exact (`erf`)** | \(\tfrac{t}{2}(1 + \mathrm{erf}(t/\sqrt{2}))\) | BERT `GELUActivation`, `F.gelu(approximate="none")` |
| **Quick (sigmoid)** | \(t \cdot \sigma(1.702\,t)\) | CLIP QuickGELU, some vision blocks |

**Weight shapes (no bias, Gemma/Llama default):**

| Matrix | Shape | Params |
|--------|-------|--------|
| \(W_g\) (`gate_proj`) | \((d,\, d_{\mathrm{ff}})\) | \(d \cdot d_{\mathrm{ff}}\) |
| \(W_u\) (`up_proj`) | \((d,\, d_{\mathrm{ff}})\) | \(d \cdot d_{\mathrm{ff}}\) |
| \(W_d\) (`down_proj`) | \((d_{\mathrm{ff}},\, d)\) | \(d \cdot d_{\mathrm{ff}}\) |

**I/O shapes (Zepto rank-2; HF/vLLM often \((B, T, d)\)):**

| Tensor | Shape |
|--------|-------|
| Input \(x\) | \((S, d)\) |
| Gate pre-activation \(g = W_g x\) | \((S, d_{\mathrm{ff}})\) |
| Up projection \(u = W_u x\) | \((S, d_{\mathrm{ff}})\) |
| Gated hidden \(h = \mathrm{GELU}(g) \odot u\) | \((S, d_{\mathrm{ff}})\) |
| Output \(y = W_d h\) | \((S, d)\) |

Let \(n = S \cdot d_{\mathrm{ff}}\) (elements in the intermediate tile).

**Numerics policies:**
- Special functions (`erf`, `tanh`, `exp`/`sigmoid`) bill **4 FLOPs/element** (Zepto coarse bucket; `docs/kernel-implementation.md` §1).
- Lone `mul` / `add` = **1 FLOP** each; `pow(t,3)` decomposes to \(t^2\) then \(t^2 \cdot t\) (2 FLOPs).
- Comparisons / branch selection: **0 FLOPs**.
- Gemma routes `hidden_act="gelu_pytorch_tanh"` → `GELUTanh` → `F.gelu(approximate="tanh")` ([`activations.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)).
- Liger GeGLU implements **tanh-GELU only** ([`geglu.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py)); erf/quick paths stay on PyTorch ATen or decomposed Zepto leaves.

**Structural vs eager decomposition (HF reference):**

```python
# GemmaMLP.forward — modeling_gemma.py
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

Eager path: **5 kernel launches** — `gate_proj`, `up_proj`, `act_fn` (GELU), elementwise `*`, `down_proj`. Liger fuses post-GEMM **tanh-GELU + mul** into one Triton kernel (`LigerGELUMulFunction`).

**Zepto identity reference (future `GeGLU` module):** `Linear(gate) ∥ Linear(up)` → `region/gelu/*` on gate → `Multiply(gelu_out, up)` → `Linear(down)`. Current `FFN` module (`src/zepto/modules/ffn.py`) is **ungated** (one up projection + xIELU); GeGLU requires a separate module with parallel gate/up branches (see `docs/model-architecture-gaps-2026-09-14.md`).

**Notation (Gemma-7B defaults where cited):**

| Symbol | Meaning | Example (Gemma-7B) |
|--------|---------|---------------------|
| \(S\) | sequence length | 8192 |
| \(d\) | hidden size | 3072 |
| \(d_{\mathrm{ff}}\) | intermediate / FFN width | 24576 |
| \(e\) | element size (bytes) | 2 (bf16) |
| \(n\) | \(S \cdot d_{\mathrm{ff}}\) | intermediate tile elements |

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | HF `GemmaMLP` eager (5 launches); PyTorch `aten::gelu` + `aten::mul`; Liger `LigerGELUMulFunction` ([`geglu.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py)) |
| **Identity lowering** | Unfused semantic chain | `region/linear(gate)` + `region/linear(up)` + `region/gelu/*` + `Multiply` + `region/linear(down)` — five structural ops, up to four intermediate HBM buffers |
| **Zepto fused region (future)** | Kernel-accurate cost leaf | `region/geglu/liger` — fuses tanh-GELU+mul post-GEMM (same **13n** elementwise FLOPs, fewer temps); parent `region/geglu` for full-stack discovery |

**Default estimates today:** sum of decomposed leaves (3× `region/linear` + `region/gelu/*` + `Multiply`). When `region/geglu/liger` is selected, **do not** also bill `region/gelu` + gate×up `Multiply` on the same edges.

**Paper-comparable (Appendix E):** `dense_mlp(..., swiglu=True)` bills **\(6 S d\, d_{\mathrm{ff}}\)** GEMM FLOPs only — three projections, **no activation/mul terms** ([`docs/kernel-implementation.md`](../docs/kernel-implementation.md) §3). GeGLU shares the same GLU topology as SwiGLU; the paper omission applies identically. That is **not** the kernel-accurate Zepto leaf total.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF Transformers eager** | [`GemmaMLP`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py) — `gate_proj`, `up_proj`, `act_fn`, `*`, `down_proj` | Partial (GELU ATen-fused) | MLP stack | Any | Both; autograd saves gate/up intermediates |
| **PyTorch ATen** | `aten::linear`, `aten::gelu`, `aten::mul` | GELU fused; GEMMs separate | MLP stack | CUDA / CPU / MPS / XPU | Both |
| **Liger GELU×mul** | [`LigerGELUMulFunction`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py) — TRL `use_liger_kernel=True` | Yes (post-GEMM, tanh-GELU) | MLP stack | CUDA (Triton) | Training-focused |
| **NVIDIA Transformer Engine** | TE Gemma/Llama-style merged gate-up (where configured) | Partial–full | MLP stack | CUDA (Hopper+) | Both |
| **Megatron-LM / TE** | Fused `linear_fc1` → `[tokens, 2·d_ff]` → GLU | Yes | MLP stack | CUDA | Training |
| **vLLM / SGLang** | Model-specific fused MLP modules (Gemma, etc.) | Partial–full | MLP stack | CUDA primary | Inference-focused |
| **HF Hub kernels** | `@use_kernel_forward_from_hub("GeLU")` on gate only | GELU only | MLP stack | Hub-routed | Both |
| **Zepto (decomposed)** | `region/linear` × 3 + `region/gelu/*` + `Multiply` | Per-leaf | MLP stack | `any` | Both |
| **Zepto (future)** | `region/geglu`, `region/geglu/liger` | Yes (cost leaf) | MLP stack | cuda (liger) / any | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **MLP stack** — not an attention fusion boundary (A/B/C/D). Classify as `mlp_stack` / `standalone_ffn`. Orthogonal to `region/gqa/*` (C) and `region/linear_ce` (D).

**Mutually exclusive (same post-GEMM edges):**
- Decomposed path: `region/gelu/*` + `Multiply(gate_act, up)` vs **`region/geglu/liger`** — pick one; Liger leaf replaces GELU+mul leaves.
- GELU variant: `region/gelu` (tanh) vs `region/gelu_erf` (exact) vs future `region/gelu/quick` — selected by approximation / pattern match; only one applies to the gate branch.
- Ungated `FFN` (xIELU) vs **GeGLU** module — different graph topology (2 vs 3 GEMMs).

**Composable:**
- **Decoder block:** `region/rmsnorm` → attention (`region/gqa/*` or decomposed) → `region/rmsnorm` → **GeGLU stack** → residual.
- Liger patches (RMSNorm, GeGLU, CE) compose with FlashAttention Hub kernels per [TRL kernels hub docs](https://huggingface.co/docs/trl/en/kernels_hub).
- Three `region/linear` leaves remain separate from GELU/mul fusion — only the activation+gate multiply fuses in Liger paths.
- `region/gelu/*` is shared with standalone GELU modules; provenance on GeGLU gate branch must not match standalone GELU outside the MLP pattern.

**Execution constraints:**
- Gate and up GEMMs share the same input \(x\); peak activation memory includes both \(g\) and \(u\) until fused GELU+mul elides \(\mathrm{GELU}(g)\) temp.
- Liger `LigerGELUMulFunction` expects separate gate \(a\) and up \(b\) tensors (same layout as SwiGLU post-GEMM path); tanh-GELU only.
- `requires_grad=False` → backward FLOPs = 0 for all leaves.
- MoE: per-expert GeGLU stacks multiply weight/param counts by routed experts; FLOPs scale with routed token count, not full expert count.
- Apertus-8B uses **xIELU**, not GeGLU; Gemma-7B numeric traces use \(d{=}3072\), \(d_{\mathrm{ff}}{=}24576\).

---

## Section 4: Forward FLOPs — step-by-step derivation

### 4.1 Identity lowering (decomposed Zepto chain)

**GEMM stages** (\(2mnk\) rule, multiply-add = 2 FLOPs):

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | `Linear(gate)` \(g = x W_g\) | \((S,d) \times (d,d_{\mathrm{ff}})\) | \(2 S d\, d_{\mathrm{ff}}\) | \(2 S d\, d_{\mathrm{ff}}\) |
| 2 | `Linear(up)` \(u = x W_u\) | \((S,d) \times (d,d_{\mathrm{ff}})\) | \(2 S d\, d_{\mathrm{ff}}\) | \(2 S d\, d_{\mathrm{ff}}\) |
| 3 | `region/gelu/*` on \(g\) | per element | 8 / 12 / 6 | \(8n\) / \(12n\) / \(6n\) |
| 4 | `Multiply` \(\mathrm{GELU}(g) \odot u\) | per element | 1 | \(n\) |
| 5 | `Linear(down)` \(y = h W_d\) | \((S,d_{\mathrm{ff}}) \times (d_{\mathrm{ff}},d)\) | \(2 S d\, d_{\mathrm{ff}}\) | \(2 S d\, d_{\mathrm{ff}}\) |

**Identity totals (kernel-accurate, by GELU variant):**

| GELU variant | Elementwise | **Total forward** |
|--------------|-------------|-------------------|
| Tanh (Gemma default) | \(12n + n = 13n\) | **\(6 S d\, d_{\mathrm{ff}} + 13n\)** |
| Exact erf | \(8n + n = 9n\) | **\(6 S d\, d_{\mathrm{ff}} + 9n\)** |
| Quick sigmoid | \(6n + n = 7n\) | **\(6 S d\, d_{\mathrm{ff}} + 7n\)** |

Cross-ref GELU per-element constants: `docs/kernel/gelu.md` §4, `src/zepto/analysis/lowering/recipes/gelu.py`.

### 4.2 Fused post-GEMM leaf (`region/geglu/liger` slice)

Liger fuses steps 3–4 into one kernel; **arithmetic unchanged** (tanh-GELU only):

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Recompute \(g^2, g^3, \tanh\) in registers | mul/add + special | 12 | \(12n\) |
| 2 | \(\mathrm{GELU}(g) \odot u\) | mul | 1 | \(n\) |
| **Fused elementwise total** | | | | **\(13n\)** |

Full stack with Liger post-GEMM fusion (tanh):
\[
\mathrm{FLOPs}_{\mathrm{fwd,Liger}} = 6 S d\, d_{\mathrm{ff}} + 13n \quad\text{(same as decomposed tanh path)}.
\]

TE merged gate-up GeGLU may reduce **GEMM count** from two to one for gate+up (\(2 S d\, d_{\mathrm{ff}}\) → one fused GEMM with same FLOPs) — FLOP total unchanged; HBM traffic for reading \(x\) may drop.

### 4.3 Paper-comparable (Appendix E)

Shazeer / Gemma training analyses often count **three linear maps only**:

\[
\mathrm{FLOPs}_{\mathrm{paper,fwd}} = 6 S d\, d_{\mathrm{ff}} \quad\text{(GEMM-only; omits GELU + mul)}.
\]

Zepto kernel-accurate leaf adds **\(13n\)** (tanh) elementwise FLOPs. At Gemma-7B (\(S{=}8192\), \(d{=}3072\), \(d_{\mathrm{ff}}{=}24576\)):

| Formula | FLOPs/layer |
|---------|-------------|
| Paper (GEMM-only) | \(\approx 3.71 \times 10^{12}\) |
| Kernel-accurate (+ tanh-GELU+mul) | \(\approx 3.71 \times 10^{12} + 2.62 \times 10^{9}\) |
| Elementwise share | **~0.07%** of GEMM FLOPs |

**Closed form (kernel-accurate, tanh default):**
\[
\mathrm{FLOPs}_{\mathrm{GeGLU,fwd}} = 6 S d\, d_{\mathrm{ff}} + 13n = 6 S d\, d_{\mathrm{ff}} + 13 S d_{\mathrm{ff}}.
\]

**Arithmetic intensity (numeric example):** Gemma-7B prefill, \(S{=}8192\), \(d{=}3072\), \(d_{\mathrm{ff}}{=}24576\), bf16 (\(e{=}2\)), tanh-GELU:

| Quantity | Value |
|----------|-------|
| GEMM FLOPs | \(6 S d\, d_{\mathrm{ff}} \approx 3.71 \times 10^{12}\) |
| Elementwise FLOPs | \(13n \approx 2.62 \times 10^{9}\) |
| Min HBM bytes (weights, persistent) | \(3 d\, d_{\mathrm{ff}} e \approx 432\) MiB |
| Min HBM bytes (activations, decomposed peak) | read \(x\) + write \(g,u,h,y\) ≈ \((d + 3 d_{\mathrm{ff}}) S e \approx 1.2\) GiB forward temps |
| GEMM arithmetic intensity | \(\approx 2d / e \approx 3072\) FLOP/byte — **compute-bound** |
| Elementwise (GELU+mul, fused) | \(\approx 13n / (2 n e) \approx 3.25\) FLOP/byte — **memory-bound** |

GeGLU FLOPs dominate in the three GEMMs; activation traffic is secondary but fusion (Liger/TE) targets the memory-bound GELU+mul tail and elision of \(\mathrm{GELU}(g)\) buffer. Tanh-GELU elementwise is **~2.2×** heavier than SwiGLU's SiLU+mul slice (\(13n\) vs \(6n\)) but remains a small fraction of total layer FLOPs.

---

## Section 5: Backward FLOPs — step-by-step derivation

Training backward decomposes into GEMM VJPs (standard `region/linear` backward ≈ **2× forward GEMM FLOPs each**) plus elementwise VJPs.

**Elementwise backward** (gate branch GELU + multiply):

| Step | Operation | Per element (tanh) | FLOPs | Subtotal |
|------|-----------|-------------------|-------|----------|
| 1 | GELU VJP on \(g\) (recompute tanh chain) | special + mul/add | 20 | \(20n\) |
| 2 | Multiply VJP w.r.t. \(u\): `grad_h * gelu(g)` | mul | 1 | \(n\) |
| 3 | Multiply VJP w.r.t. \(\mathrm{GELU}(g)\): `grad_h * u` | mul | 1 | \(n\) |
| **Elementwise backward (tanh)** | | | | **\(22n\)** |

Liger `geglu_backward` recomputes tanh-GELU from saved \((g, u)\) — same **22n** billing ([`geglu.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py) backward kernel).

**Elementwise backward by GELU variant:**

| GELU variant | GELU VJP | Multiply VJP | **Total** |
|--------------|----------|--------------|-----------|
| Tanh | \(20n\) | \(2n\) | **\(22n\)** |
| Exact erf | \(17n\) | \(2n\) | **\(19n\)** |
| Quick sigmoid | \(9n\) | \(2n\) | **\(11n\)** |

**GEMM backward** (each `region/linear`, `requires_grad=True`):

| Step | Primitive | FLOPs |
|------|-----------|-------|
| `down_proj` backward | \(dX, dW\) | \(4 S d\, d_{\mathrm{ff}}\) |
| `gate_proj` backward | \(dX, dW\) | \(4 S d\, d_{\mathrm{ff}}\) |
| `up_proj` backward | \(dX, dW\) | \(4 S d\, d_{\mathrm{ff}}\) |
| **GEMM backward total** | | **\(12 S d\, d_{\mathrm{ff}}\)** |

Gate and up both consume the same upstream \(d x\); FLOPs for both backward GEMMs are still billed.

**Identity lowering backward total (tanh default):**
\[
\mathrm{FLOPs}_{\mathrm{GeGLU,bwd,identity}} = 12 S d\, d_{\mathrm{ff}} + 22n.
\]

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{GeGLU,bwd}} = \begin{cases} 12 S d\, d_{\mathrm{ff}} + 22n & \text{tanh, if requires\_grad} \\ 12 S d\, d_{\mathrm{ff}} + 19n & \text{erf, if requires\_grad} \\ 12 S d\, d_{\mathrm{ff}} + 11n & \text{quick, if requires\_grad} \\ 0 & \text{otherwise} \end{cases}
\]

**Paper-comparable:** Appendix E / Megatron often report GEMM-only backward **\(12 S d\, d_{\mathrm{ff}}\)** (2× forward GEMMs), omitting **\(22n\)** activation VJPs — label explicitly when comparing to Zepto.

**Common mistakes:**
- Do **not** use \(2 \times (6 S d d_{\mathrm{ff}} + 13n)\) as a shortcut — mixes GEMM and elementwise scaling rules.
- Do **not** double-count `region/gelu` (20n) **and** full multiply VJP (2n) when using a fused `region/geglu/liger` leaf — bill **22n** once for the fused post-GEMM backward.
- Do **not** bill standalone `region/gelu` when `region/geglu/liger` wins on the same gate branch.

`requires_grad=False` → backward FLOPs = **0**.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers (name + shape) | Peak bytes (order) | Elided by fusion |
|------|----------------------------|--------------------|------------------|
| **HF eager decomposed** | \(g\), \(u\), \(\mathrm{GELU}(g)\), \(h\), \(y\) | up to **\(3n + S d\) \(e\)** simultaneous | — |
| **Decomposed + `region/gelu`** | \(g\), \(u\), \(h\), \(y\) (GELU temps elided) | **\((2n + S d) e\)** | \(x^2\), tanh/erf temps (\(n e\) equivalent) |
| **Liger `LigerGELUMulFunction`** | \(g\), \(u\), \(h\) (returns \(c=h\)) | **\(2n e\)** + output | \(\mathrm{GELU}(g)\) (\(n e\)) |
| **TE merged gate-up** | single GEMM out \((S, 2 d_{\mathrm{ff}})\), \(h\) | **\(3n e\)** | second GEMM output buffer |

Weights (persistent, all paths): \(3 d\, d_{\mathrm{ff}} e\) for \((W_g, W_u, W_d)\).

Input \(x\) is a boundary edge (not re-allocated by the stack).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| HF eager `GemmaMLP` | gate \(g\), up \(u\) (via linear + mul autograd) | \((S, d_{\mathrm{ff}})\) each | \(2 n e\) | per-op SAVE on linear outputs |
| **`region/gelu` + `Multiply`** | \(g\) (gelu SAVE), \(u\) (mul SAVE) | \((S, d_{\mathrm{ff}})\) | \(2 n e\) | `save_input` on gate; multiply saves both operands |
| **Liger `LigerGELUMulFunction`** | \(g\), \(u\) | \((S, d_{\mathrm{ff}})\) | \(2 n e\) | `save_gate_up: true` |
| **`Linear(down)` backward** | \(h\) (down input) | \((S, d_{\mathrm{ff}})\) | \(n e\) | standard linear SAVE(input) |

Fused GELU+mul does **not** save \(\mathrm{GELU}(g)\) or \(h\) for the mul backward — recomputed from \((g, u)\) (Liger comment: "recomputation to save memory").

Down-proj linear typically saves \(h\) for \(dW_d = h^\top d y\).

### 6.3 Resource event chains

**Identity lowering (decomposed, unfused GELU):**
```
PERSIST(W_g, W_u, W_d)
→ ALLOCATE(g) → SAVE(g)           # gate linear
→ ALLOCATE(u) → SAVE(u)           # up linear
→ ALLOCATE(x²) → SAVE(input=g)    # tanh temps (unfused gelu path)
→ ALLOCATE(gelu_g) → SAVE(...)    # gelu output
→ ALLOCATE(h) → SAVE(gate, up)    # multiply
→ ALLOCATE(y) → SAVE(h)           # down linear
```

**Decomposed with `region/gelu` (fused GELU leaf):**
```
PERSIST(W_g, W_u, W_d)
→ ALLOCATE(g) → SAVE(g)           # gate linear
→ ALLOCATE(u) → SAVE(u)           # up linear
→ ALLOCATE(gelu_g) → SAVE(input=g)  # region/gelu; elides x²/tanh temps
→ ALLOCATE(h) → SAVE(gelu_g, u)   # multiply
→ ALLOCATE(y) → SAVE(h)           # down linear
```

**Fused post-GEMM leaf (`region/geglu/liger`):**
```
PERSIST(W_g, W_u, W_d)
→ ALLOCATE(g) → SAVE(g)
→ ALLOCATE(u) → SAVE(u)
→ ALLOCATE(h) → SAVE(g, u)        # Liger GELU+mul; elides gelu_g and tanh temps
→ ALLOCATE(y) → SAVE(h)
```

Backward phase end: `RELEASE` saved tensors per leaf policy.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF GemmaMLP eager | transformers | Partial | MLP stack | \(6 S d d_{\mathrm{ff}} + 13n\) | \(12 S d d_{\mathrm{ff}} + 22n\) | SAVE \(g,u,h\) | decomposed |
| PyTorch ATen | torch | GELU fused | MLP stack | same (tanh) | same | SAVE per autograd | — |
| Liger GELU×mul | liger-kernel | Yes (post-GEMM) | MLP stack | \(6 S d d_{\mathrm{ff}} + 13n\) | \(12 S d d_{\mathrm{ff}} + 22n\) | SAVE \(g,u\); elide gelu temp | `region/geglu/liger` (future) |
| NVIDIA TE | transformer_engine | Partial–full | MLP stack | same GEMM; same 13n | same | merged gate-up buffer | — |
| Zepto decomposed | zepto | Per-leaf | MLP stack | \(6 S d d_{\mathrm{ff}} + 13n\) | \(12 S d d_{\mathrm{ff}} + 22n\) | leaf SAVE policies | `region/linear` + `region/gelu` + mul |
| Appendix E paper | Apertus §3 | N/A | MLP stack | \(6 S d d_{\mathrm{ff}}\) only | \(12 S d d_{\mathrm{ff}}\) only | not modeled | — |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: geglu
recommended_region_ids:
  - id: region/geglu/decomposed
    variant: default
    hardware_gate: any
    fusion_boundary: mlp_stack
    status: to_implement
    capabilities:
      - geglu
      - decomposed
  - id: region/geglu/decomposed-erf
    variant: erf
    hardware_gate: any
    fusion_boundary: mlp_stack
    status: to_implement
    capabilities:
      - geglu
      - decomposed
      - gelu_erf
  - id: region/geglu/liger
    variant: liger
    hardware_gate: cuda
    fusion_boundary: mlp_stack
    status: to_implement
    capabilities:
      - geglu
      - fused_post_gemm
pattern_rule:
  op_families:
    - linear_matmul   # gate_proj
    - linear_matmul   # up_proj (parallel input)
    - gelu            # region/gelu, region/gelu_erf, or region/gelu/quick
    - multiply
    - linear_matmul   # down_proj
  module_envelope: GeGLU
  edge_constraints:
    - shared_input_on_gate_up
    - gelu_on_gate_branch_only
recipe:
  forward_flops: "6 * S * d * d_ff + (gelu_fwd_per_elem + 1) * S * d_ff"
  backward_flops: "12 * S * d * d_ff + (gelu_bwd_per_elem + 2) * S * d_ff if requires_grad else 0"
  gelu_fwd_per_elem_tanh: 12
  gelu_fwd_per_elem_erf: 8
  gelu_fwd_per_elem_quick: 6
  gelu_bwd_per_elem_tanh: 20
  gelu_bwd_per_elem_erf: 17
  gelu_bwd_per_elem_quick: 9
  forward_flops_gemm: "6 * S * d * d_ff"
  forward_flops_elementwise_tanh: "13 * S * d_ff"
  backward_flops_gemm: "12 * S * d * d_ff"
  backward_flops_elementwise_tanh: "22 * S * d_ff"
  paper_forward_flops: "6 * S * d * d_ff"
  paper_backward_flops: "12 * S * d * d_ff"
  save_gate: true
  save_up: true
  save_down_input: true
  elided_temps:
    - x_squared           # x² — register-only in fused GELU
    - tanh_intermediate   # tanh(u) — register-only in fused tanh-GELU
    - erf_intermediate    # erf temps — register-only in fused erf-GELU
    - gelu_gate_output    # GELU(g) — elided when GELU+mul fused (Liger)
  saved_backward:
    - name: gate
      shape: "(S, d_ff)"
    - name: up
      shape: "(S, d_ff)"
    - name: down_input
      shape: "(S, d_ff)"
  resource_events_forward:
    - "PERSIST(W_g, W_u, W_d)"
    - "ALLOCATE(g) → SAVE(g)"
    - "ALLOCATE(u) → SAVE(u)"
    - "ALLOCATE(h) → SAVE(gate, up) when fused_post_gemm else SAVE(gelu_out, u)"
    - "ALLOCATE(y) → SAVE(h) on down linear"
  resource_events_backward:
    - "RELEASE(gate, up, h) on backward phase end"
  numerics_tags:
    - gelu_tanh_default_gemma
    - gelu_erf_bert_variant
    - gated_glu_elementwise
    - appendix_e_gemm_only_paper_compare
capabilities:
  - geglu
  - gated_ffn
priority: 8
variants:
  - id: region/geglu/decomposed
    recipe_override:
      gelu_approx: tanh
      forward_flops_per_element: 13
      backward_flops_per_element: 22
      elided_temps: [x_squared, tanh_intermediate]
  - id: region/geglu/decomposed-erf
    recipe_override:
      gelu_approx: none
      forward_flops_per_element: 9
      backward_flops_per_element: 19
      elided_temps: [erf_intermediate]
  - id: region/geglu/liger
    recipe_override:
      gelu_approx: tanh
      forward_flops_per_element: 13
      backward_flops_per_element: 22
      elided_temps: [x_squared, tanh_intermediate, gelu_gate_output]
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Shazeer (2020), GLU Variants Improve Transformer (GeGLU): https://arxiv.org/abs/2002.05202
- Hendrycks & Gimpel (2016), GELU: https://arxiv.org/abs/1606.08415
- Gemma model (GeGLU FFN, `gelu_pytorch_tanh`): https://huggingface.co/docs/transformers/model_doc/gemma
- HuggingFace `GemmaMLP`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
- HuggingFace `GELUTanh` / `gelu_pytorch_tanh`: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- PyTorch `ActivationGeluKernel.cu`: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationGeluKernel.cu
- Liger-Kernel `geglu.py` (`LigerGELUMulFunction`): https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py
- Liger-Kernel paper (Dai et al., 2024): https://arxiv.org/abs/2410.10989
- TRL Liger integration: https://huggingface.co/docs/trl/en/liger_kernel_integration
- TRL kernels hub composition rules: https://huggingface.co/docs/trl/en/kernels_hub
- Zepto GELU region (gate branch leaf): `docs/kernel/gelu.md`, `src/zepto/analysis/lowering/recipes/gelu.py`
- Zepto SwiGLU region (structural analog): `docs/kernel/swiglu.md`, `src/zepto/analysis/lowering/implementations/regions/swiglu/`
- Zepto linear region: `src/zepto/analysis/lowering/implementations/regions/linear.py`
- Cross-ref: `docs/kernel-implementation.md` §3 (Appendix E GLU \(6 S d d_{\mathrm{ff}}\)), §13 (GELU), §957 (GeGLU composition)
- Model gaps (GeGLU module missing): `docs/model-architecture-gaps-2026-09-14.md`

**Verification date:** 2026-09-15

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| — | `region/gelu` | §3 composable | standalone_activation | **Registered (recipe)** — gate nonlinearity slice only (tanh) |
| — | `region/gelu_erf` | §3 composable | standalone_activation | **Registered (recipe)** — erf gate variant |
| — | `region/linear` | §4 | MLP stack | **Registered** — gate/up/down GEMMs |
| 1 | `region/geglu/decomposed` | §8 | mlp_stack | Parent discovery wrapping 3× linear + gelu + mul |
| 2 | `region/geglu/decomposed-erf` | §8 variants | mlp_stack | BERT/erf-GELU gate variant |
| 3 | `region/geglu/liger` | §8 variants | mlp_stack | Liger post-GEMM tanh-GELU+mul fusion |
| 4 | `GeGLU` module | §0 | mlp_stack | Future — `src/zepto/modules/geglu.py` for graph provenance |

**Open gaps:**
- **`GeGLU` module not registered** — graph discovery requires future module (`docs/model-architecture-gaps-2026-09-14.md`).
- **`region/gelu*` RegionImplementations not registered** — recipes exist; decomposed GeGLU relies on leaf registration chain.
- **Liger fused gate-up for GeGLU** — Liger ships `LigerFusedGateUpSiLUMulFunction` for SwiGLU only; no GeGLU fused gate-up kernel in Liger today.
- **Quick-GELU GeGLU** — future `region/gelu/quick` + `region/geglu/decomposed-quick` variant.
- **MoE routing** — expert GeGLU FLOPs scale with tokens routed, not registered in this leaf.
- **TE merged gate-up GEMM** — same FLOPs, different HBM; no dedicated Zepto variant yet.
