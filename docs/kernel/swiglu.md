# Zepto kernel research: SwiGLU (gated FFN)

**Date:** 2026-09-15
**Proposer:** Robin Girardin
**Scope:** Full gated MLP stack (`gate_proj → SiLU → gate×up → down_proj`); prefill + decode; training + inference; CUDA primary; Llama/Granite/Qwen-style FFNs
**Context:** SwiGLU is the dominant FFN in modern LLMs (Llama, Mistral, Granite, Qwen, …). Apertus uses xIELU in an ungated two-linear FFN instead; this report targets gated stacks for cross-model cost modeling. Default Zepto estimates use **decomposed leaves** (`region/linear` × 3 + `region/silu` + `Multiply`) via registered `region/swiglu/*` variants (see §8).

---

## Section 0: Mathematical definition

SwiGLU (Swish-Gated Linear Unit) is a GLU variant ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)) used as the feed-forward block in Llama-class transformers:

\[
\mathrm{SwiGLU}(x) = W_d \,\Bigl(\mathrm{SiLU}(W_g x) \odot (W_u x)\Bigr),
\]

where \(\mathrm{SiLU}(t) = t \cdot \sigma(t)\), \(\sigma\) is the logistic sigmoid, and \(\odot\) is elementwise multiplication.

**Weight shapes (no bias, Llama default):**

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
| Gated hidden \(h = \mathrm{SiLU}(g) \odot u\) | \((S, d_{\mathrm{ff}})\) |
| Output \(y = W_d h\) | \((S, d)\) |

Let \(n = S \cdot d_{\mathrm{ff}}\) (elements in the intermediate tile).

**Numerics policies:**
- SiLU sigmoid/\(\exp\) bills **4 FLOPs/element** (Zepto special-function bucket; `docs/kernel-implementation.md` §1).
- Comparisons / branch selection: **0 FLOPs**.
- Llama `hidden_act="silu"` routes through `SiLUActivation` → `F.silu` ([`activations.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)).
- Liger optional `gate_multiplier` / `down_multiplier` scalars (GPT-OSS expert variant) — default **1.0** for standard SwiGLU; multipliers add no extra FLOPs beyond one scalar mul per element when \(\neq 1\).

**Structural vs eager decomposition (HF reference):**

```python
# LlamaMLP.forward — modeling_llama.py
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

Eager path: **5 kernel launches** — `gate_proj`, `up_proj`, `act_fn` (SiLU), elementwise `*`, `down_proj`. Liger fuses the post-GEMM **SiLU + mul** (and optionally fused gate-up layout) into one Triton kernel.

**Zepto identity reference (future `SwiGLU` module):** `Linear(gate) ∥ Linear(up)` → `region/silu(gate)` → `Multiply(silu_out, up)` → `Linear(down)`. Current `FFN` module (`src/zepto/modules/ffn.py`) is **ungated** (one up projection + xIELU); SwiGLU requires a separate module with parallel gate/up branches.

**Notation (Apertus-8B defaults where cited):**

| Symbol | Meaning | Example (Llama-8B) |
|--------|---------|---------------------|
| \(S\) | sequence length | 8192 |
| \(d\) | hidden size | 4096 |
| \(d_{\mathrm{ff}}\) | intermediate / FFN width | 14336 (Llama-3-8B) |
| \(e\) | element size (bytes) | 2 (bf16) |
| \(n\) | \(S \cdot d_{\mathrm{ff}}\) | intermediate tile elements |

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | HF `LlamaMLP` eager (5 launches); Liger `LigerSiLUMulFunction` / `LigerFusedGateUpSiLUMulFunction` ([`swiglu.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py)); NVIDIA TE merged gate-up SwiGLU ([TE Llama tutorial](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html)) |
| **Identity lowering** | Unfused semantic chain | `region/linear(gate)` + `region/linear(up)` + `region/silu` + `Multiply` + `region/linear(down)` — five structural ops, up to four intermediate HBM buffers |
| **Zepto fused region (future)** | Kernel-accurate cost leaf | `region/swiglu/liger` — fuses SiLU+mul post-GEMM (same **6n** elementwise FLOPs, fewer temps); parent `region/swiglu` for full-stack discovery |

**Default estimates today:** sum of registered decomposed leaves (3× `region/linear` + `region/silu` + `Multiply`). When `region/swiglu/liger` is selected, **do not** also bill `region/silu` + gate×up `Multiply` on the same edges.

**Paper-comparable (Appendix E):** `dense_mlp(..., swiglu=True)` bills **\(6 S d\, d_{\mathrm{ff}}\)** GEMM FLOPs only — three projections, **no activation/mul terms** ([`docs/kernel-implementation.md`](../docs/kernel-implementation.md) §3). That is **not** the kernel-accurate Zepto leaf total.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **HF Transformers eager** | [`LlamaMLP`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — `gate_proj`, `up_proj`, `act_fn`, `*`, `down_proj` | Partial (SiLU ATen-fused) | MLP stack | Any | Both; autograd saves gate/up intermediates |
| **PyTorch ATen** | `aten::linear`, `aten::silu`, `aten::mul` | SiLU fused; GEMMs separate | MLP stack | CUDA / CPU / MPS / XPU | Both |
| **Liger SiLU×mul** | [`LigerSiLUMulFunction`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py) — TRL `use_liger_kernel=True` | Yes (post-GEMM) | MLP stack | CUDA (Triton) | Training-focused |
| **Liger fused gate-up** | [`LigerFusedGateUpSiLUMulFunction`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py) — Megatron `[..., 2·d_ff]` layout | Yes (gate-up SiLU+mul) | MLP stack | CUDA (Triton) | Training; SAVE fused gate-up tensor |
| **NVIDIA Transformer Engine** | TE Llama SwiGLU — merged `gate_up_proj` + custom kernel | Yes (gate-up + SiLU) | MLP stack | CUDA (Hopper+) | Both |
| **Megatron-LM / TE** | Fused `linear_fc1` → `[tokens, 2·d_ff]` → SwiGLU | Yes | MLP stack | CUDA | Training |
| **vLLM / SGLang** | Model-specific fused MLP modules | Partial–full | MLP stack | CUDA primary | Inference-focused |
| **HF Hub kernels** | `@use_kernel_forward_from_hub("SiLU")` on gate only | SiLU only | MLP stack | Hub-routed | Both |
| **Zepto (decomposed)** | `region/linear` × 3 + `region/silu` + `Multiply` | Per-leaf | MLP stack | `any` | Both |
| **Zepto (future)** | `region/swiglu`, `region/swiglu/liger` | Yes (cost leaf) | MLP stack | cuda (liger) / any | Both |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **MLP stack** — not an attention fusion boundary (A/B/C/D). Classify as `mlp_stack` / `standalone_ffn`. Orthogonal to `region/gqa/*` (C) and `region/linear_ce` (D).

**Mutually exclusive (same post-GEMM edges):**
- Decomposed path: `region/silu` + `Multiply(gate_act, up)` vs **`region/swiglu/liger`** (or `region/swiglu/fused-gate-up`) — pick one; Liger leaf replaces SiLU+mul leaves.
- Ungated `FFN` (xIELU) vs **SwiGLU** module — different graph topology (2 vs 3 GEMMs).

**Composable:**
- **Decoder block:** `region/rmsnorm` → attention (`region/gqa/*` or decomposed) → `region/rmsnorm` → **SwiGLU stack** → residual.
- Liger patches (RMSNorm, SwiGLU, CE) compose with FlashAttention Hub kernels per [TRL kernels hub docs](https://huggingface.co/docs/trl/en/kernels_hub).
- Three `region/linear` leaves remain separate from SiLU/mul fusion — only the activation+gate multiply fuses in Liger/TE paths.
- `region/silu` is shared with standalone `SiLU` modules; provenance on SwiGLU gate branch must not match standalone SiLU outside the MLP pattern.

**Execution constraints:**
- Gate and up GEMMs share the same input \(x\); peak activation memory includes both \(g\) and \(u\) until fused SiLU+mul elides \(\mathrm{SiLU}(g)\) temp.
- Liger `LigerFusedGateUpSiLUMulFunction` expects trailing dim \(2 \cdot d_{\mathrm{ff}}\); HF uses separate projections.
- `requires_grad=False` → backward FLOPs = 0 for all leaves.
- GPT-OSS expert SwiGLU uses `gate_multiplier=1.702`, `down_multiplier`, and `(up+1)` offset — **variant**; not covered by default `region/swiglu` (see `docs/model-architecture-gaps-2026-09-14.md`).
- MoE: per-expert SwiGLU stacks multiply weight/param counts by `num_experts_per_tok`; FLOPs scale with routed token count, not full expert count.

---

## Section 4: Forward FLOPs — step-by-step derivation

### 4.1 Identity lowering (decomposed Zepto chain)

**GEMM stages** (\(2mnk\) rule, multiply-add = 2 FLOPs):

| Step | Primitive | Tile shape | FLOPs | Subtotal |
|------|-----------|------------|-------|----------|
| 1 | `Linear(gate)` \(g = x W_g\) | \((S,d) \times (d,d_{\mathrm{ff}})\) | \(2 S d\, d_{\mathrm{ff}}\) | \(2 S d\, d_{\mathrm{ff}}\) |
| 2 | `Linear(up)` \(u = x W_u\) | \((S,d) \times (d,d_{\mathrm{ff}})\) | \(2 S d\, d_{\mathrm{ff}}\) | \(2 S d\, d_{\mathrm{ff}}\) |
| 3 | `region/silu` on \(g\) | per element | 5 | \(5n\) |
| 4 | `Multiply` \(\mathrm{SiLU}(g) \odot u\) | per element | 1 | \(n\) |
| 5 | `Linear(down)` \(y = h W_d\) | \((S,d_{\mathrm{ff}}) \times (d_{\mathrm{ff}},d)\) | \(2 S d\, d_{\mathrm{ff}}\) | \(2 S d\, d_{\mathrm{ff}}\) |
| **Identity total** | | | | **\(6 S d\, d_{\mathrm{ff}} + 6n\)** |

### 4.2 Fused post-GEMM leaf (`region/swiglu/liger` slice)

Liger fuses steps 3–4 into one kernel; **arithmetic unchanged**:

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | Fused \(\sigma(g)\) in registers | special | 4 | \(4n\) |
| 2 | \(\mathrm{SiLU}(g) = g \cdot \sigma(g)\) | mul | 1 | \(n\) |
| 3 | \(\mathrm{SiLU}(g) \odot u\) | mul | 1 | \(n\) |
| **Fused elementwise total** | | | | **\(6n\)** |

Full stack with Liger post-GEMM fusion:
\[
\mathrm{FLOPs}_{\mathrm{fwd,Liger}} = 6 S d\, d_{\mathrm{ff}} + 6n \quad\text{(same as decomposed)}.
\]

TE merged gate-up SwiGLU may reduce **GEMM count** from two to one for the gate+up multiply (\(2 S d\, d_{\mathrm{ff}}\) → one \(2 S d\, (2 d_{\mathrm{ff}})\) fused GEMM with same FLOPs) — FLOP total unchanged; HBM traffic for reading \(x\) may drop.

### 4.3 Paper-comparable (Appendix E)

Shazeer / Llama training analyses often count **three linear maps only**:

\[
\mathrm{FLOPs}_{\mathrm{paper,fwd}} = 6 S d\, d_{\mathrm{ff}} \quad\text{(GEMM-only; omits SiLU + mul)}.
\]

Zepto kernel-accurate leaf adds **\(6n\)** elementwise FLOPs. At Llama-3-8B (\(S{=}8192\), \(d{=}4096\), \(d_{\mathrm{ff}}{=}14336\)):

| Formula | FLOPs/layer |
|---------|-------------|
| Paper (GEMM-only) | \(\approx 9.6 \times 10^{11}\) |
| Kernel-accurate (+ SiLU+mul) | \(\approx 9.6 \times 10^{11} + 7.1 \times 10^{8}\) |
| Elementwise share | **~0.74%** of GEMM FLOPs |

**Closed form (kernel-accurate, decomposed or Liger):**
\[
\mathrm{FLOPs}_{\mathrm{SwiGLU,fwd}} = 6 S d\, d_{\mathrm{ff}} + 6n = 6 S d\, d_{\mathrm{ff}} + 6 S d_{\mathrm{ff}}.
\]

**Arithmetic intensity (numeric example):** Llama-3-8B prefill, \(S{=}8192\), \(d{=}4096\), \(d_{\mathrm{ff}}{=}14336\), bf16 (\(e{=}2\)):

| Quantity | Value |
|----------|-------|
| GEMM FLOPs | \(6 S d\, d_{\mathrm{ff}} \approx 9.63 \times 10^{11}\) |
| Elementwise FLOPs | \(6n \approx 7.07 \times 10^{8}\) |
| Min HBM bytes (weights, persistent) | \(3 d\, d_{\mathrm{ff}} e \approx 336\) MiB |
| Min HBM bytes (activations, decomposed peak) | read \(x\) + write \(g,u,h,y\) ≈ \((d + 3 d_{\mathrm{ff}}) S e \approx 470\) MiB forward temps |
| GEMM arithmetic intensity | \(\approx 2d / e \approx 4096\) FLOP/byte — **compute-bound** |
| Elementwise (SiLU+mul, fused) | \(\approx 6n / (2 n e) \approx 1.5\) FLOP/byte — **memory-bound** |

SwiGLU FLOPs dominate in the three GEMMs; activation traffic is secondary but fusion (Liger/TE) targets the memory-bound SiLU+mul tail and elision of \(\mathrm{SiLU}(g)\) buffer.

---

## Section 5: Backward FLOPs — step-by-step derivation

Training backward decomposes into GEMM VJPs (standard `region/linear` backward ≈ **2× forward GEMM FLOPs each**) plus elementwise VJPs.

**Elementwise backward** (gate branch SiLU + multiply):

| Step | Operation | Per element | FLOPs | Subtotal |
|------|-----------|-------------|-------|----------|
| 1 | SiLU VJP on \(g\) (recompute \(\sigma\)) | special + mul/add | 8 | \(8n\) |
| 2 | Multiply VJP w.r.t. \(u\): `grad_h * silu(g)` | mul | 1 | \(n\) |
| 3 | Multiply VJP w.r.t. \(\mathrm{SiLU}(g)\): `grad_h * u` | mul | 1 | \(n\) |
| **Elementwise backward** | | | | **\(10n\)** |

Liger `swiglu_backward` recomputes \(\sigma\) and SiLU from saved \((g, u)\) — same **10n** billing ([`swiglu.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py) backward kernel).

**GEMM backward** (each `region/linear`, `requires_grad=True`):

| Step | Primitive | FLOPs |
|------|-----------|-------|
| `down_proj` backward | \(dX, dW\) | \(4 S d\, d_{\mathrm{ff}}\) |
| `gate_proj` backward | \(dX, dW\) | \(4 S d\, d_{\mathrm{ff}}\) |
| `up_proj` backward | \(dX, dW\) | \(4 S d\, d_{\mathrm{ff}}\) |
| **GEMM backward total** | | **\(12 S d\, d_{\mathrm{ff}}\)** |

Gate and up both consume the same upstream \(d x\); in practice \(d x = d x_{\mathrm{gate}} + d x_{\mathrm{up}}\) (accumulated). FLOPs for both backward GEMMs are still billed.

**Identity lowering backward total:**
\[
\mathrm{FLOPs}_{\mathrm{SwiGLU,bwd,identity}} = 12 S d\, d_{\mathrm{ff}} + 10n.
\]

**Closed form:**
\[
\mathrm{FLOPs}_{\mathrm{SwiGLU,bwd}} = \begin{cases} 12 S d\, d_{\mathrm{ff}} + 10n & \text{if requires\_grad} \\ 0 & \text{otherwise} \end{cases}
\]

**Paper-comparable:** Appendix E / Megatron often report GEMM-only backward **\(12 S d\, d_{\mathrm{ff}}\)** (2× forward GEMMs), omitting **\(10n\)** activation VJPs — label explicitly when comparing to Zepto.

**Common mistakes:**
- Do **not** use \(2 \times (6 S d d_{\mathrm{ff}} + 6n)\) as a shortcut — mixes GEMM and elementwise scaling rules.
- Do **not** double-count `region/silu` (8n) **and** full multiply VJP (2n) when using a fused `region/swiglu/liger` leaf — bill **10n** once for the fused post-GEMM backward.

`requires_grad=False` → backward FLOPs = **0**.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers (name + shape) | Peak bytes (order) | Elided by fusion |
|------|----------------------------|--------------------|------------------|
| **HF eager decomposed** | \(g\), \(u\), \(\mathrm{SiLU}(g)\), \(h\), \(y\) | up to **\(3n + S d\) \(e\)** simultaneous | — |
| **Decomposed + `region/silu`** | \(g\), \(u\), \(h\), \(y\) (SiLU temp elided) | **\((2n + S d) e\)** | \(\mathrm{SiLU}(g)\) (\(n e\)) |
| **Liger `LigerSiLUMulFunction`** | \(g\), \(u\), \(h\) (returns \(c=h\)) | **\(2n e\)** + output | \(\mathrm{SiLU}(g)\) |
| **Liger fused gate-up** | fused \([g; u]\) (\(2n e\)), \(h\) | **\(2n e + n e = 3n e\)** | separate \(g,u\) if merged upstream |
| **TE merged gate-up** | single GEMM out \((S, 2 d_{\mathrm{ff}})\), \(h\) | **\(3n e\)** | second GEMM output buffer |

Weights (persistent, all paths): \(3 d\, d_{\mathrm{ff}} e\) for \((W_g, W_u, W_d)\).

Input \(x\) is a boundary edge (not re-allocated by the stack).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes | Zepto SAVE policy |
|------|---------------|-------|-------|-------------------|
| HF eager `LlamaMLP` | gate \(g\), up \(u\) (via linear + mul autograd) | \((S, d_{\mathrm{ff}})\) each | \(2 n e\) | per-op SAVE on linear outputs |
| **`region/silu` + `Multiply`** | \(g\) (silu SAVE), \(u\) (mul SAVE) | \((S, d_{\mathrm{ff}})\) | \(2 n e\) | `save_input` on gate; multiply saves both operands |
| **Liger `LigerSiLUMulFunction`** | \(g\), \(u\) | \((S, d_{\mathrm{ff}})\) | \(2 n e\) | `save_gate_up: true` |
| **Liger fused gate-up** | fused gate-up tensor | \((S, 2 d_{\mathrm{ff}})\) | \(2 n e\) | `save_fused_gate_up: true` |
| **`Linear(down)` backward** | \(h\) (down input) | \((S, d_{\mathrm{ff}})\) | \(n e\) | standard linear SAVE(input) |

Fused SiLU+mul does **not** save \(\mathrm{SiLU}(g)\) or \(h\) for the mul backward — recomputed from \((g, u)\).

Down-proj linear typically saves \(h\) for \(dW_d = h^\top d y\).

### 6.3 Resource event chains

**Identity lowering (decomposed, unfused SiLU):**
```
PERSIST(W_g, W_u, W_d)
→ ALLOCATE(g) → SAVE(g)           # gate linear
→ ALLOCATE(u) → SAVE(u)           # up linear
→ ALLOCATE(σ(g)) → SAVE(input=g)  # sigmoid temp (unfused silu path)
→ ALLOCATE(silu_g) → SAVE(...)    # silu output
→ ALLOCATE(h) → SAVE(gate, up)    # multiply
→ ALLOCATE(y) → SAVE(h)           # down linear
```

**Decomposed with `region/silu` (fused SiLU leaf):**
```
PERSIST(W_g, W_u, W_d)
→ ALLOCATE(g) → SAVE(g)           # gate linear
→ ALLOCATE(u) → SAVE(u)           # up linear
→ ALLOCATE(silu_g) → SAVE(input=g)  # region/silu; elides σ(g)
→ ALLOCATE(h) → SAVE(silu_g, u)   # multiply
→ ALLOCATE(y) → SAVE(h)           # down linear
```

**Fused post-GEMM leaf (`region/swiglu/liger`):**
```
PERSIST(W_g, W_u, W_d)
→ ALLOCATE(g) → SAVE(g)
→ ALLOCATE(u) → SAVE(u)
→ ALLOCATE(h) → SAVE(g, u)        # Liger SiLU+mul; elides silu_g and σ(g)
→ ALLOCATE(y) → SAVE(h)
```

**Fused gate-up variant (`region/swiglu/liger-fused-gate-up`):**
```
PERSIST(W_g, W_u, W_d)
→ ALLOCATE(gate_up) → SAVE(gate_up)   # [g; u] fused layout, 2n e
→ ALLOCATE(h) → SAVE(gate_up)         # SiLU+mul from fused buffer
→ ALLOCATE(y) → SAVE(h)
```

Backward phase end: `RELEASE` saved tensors per leaf policy.

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| HF LlamaMLP eager | transformers | Partial | MLP stack | \(6 S d d_{\mathrm{ff}} + 6n\) | \(12 S d d_{\mathrm{ff}} + 10n\) | SAVE \(g,u,h\) | decomposed |
| PyTorch ATen | torch | SiLU fused | MLP stack | same | same | SAVE per autograd | — |
| Liger SiLU×mul | liger-kernel | Yes (post-GEMM) | MLP stack | \(6 S d d_{\mathrm{ff}} + 6n\) | \(12 S d d_{\mathrm{ff}} + 10n\) | SAVE \(g,u\); elide silu temp | `region/swiglu/liger` (future) |
| Liger fused gate-up | liger-kernel | Yes | MLP stack | same | same | SAVE \([g;u]\) | `region/swiglu/liger-fused-gate-up` (future) |
| NVIDIA TE | transformer_engine | Yes (gate-up+SiLU) | MLP stack | same GEMM; same 6n | same | merged gate-up buffer | — |
| Zepto decomposed | zepto | Per-leaf | MLP stack | \(6 S d d_{\mathrm{ff}} + 6n\) | \(12 S d d_{\mathrm{ff}} + 10n\) | leaf SAVE policies | `region/linear` + `region/silu` + mul |
| Appendix E paper | Apertus §3 | N/A | MLP stack | \(6 S d d_{\mathrm{ff}}\) only | \(12 S d d_{\mathrm{ff}}\) only | not modeled | — |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: swiglu
recommended_region_ids:
  - id: region/swiglu/decomposed
    variant: default
    hardware_gate: any
    fusion_boundary: mlp_stack
    status: to_implement
    capabilities:
      - swiglu
      - decomposed
  - id: region/swiglu/liger
    variant: liger
    hardware_gate: cuda
    fusion_boundary: mlp_stack
    status: to_implement
    capabilities:
      - swiglu
      - fused_post_gemm
  - id: region/swiglu/liger-fused-gate-up
    variant: liger-fused-gate-up
    hardware_gate: cuda
    fusion_boundary: mlp_stack
    status: to_implement
    capabilities:
      - swiglu
      - fused_gate_up
pattern_rule:
  op_families:
    - linear_matmul   # gate_proj
    - linear_matmul   # up_proj (parallel input)
    - silu            # or region/silu
    - multiply
    - linear_matmul   # down_proj
  module_envelope: SwiGLU
  edge_constraints:
    - shared_input_on_gate_up
    - silu_on_gate_branch_only
recipe:
  forward_flops: "6 * S * d * d_ff + 6 * S * d_ff"
  backward_flops: "12 * S * d * d_ff + 10 * S * d_ff if requires_grad else 0"
  forward_flops_gemm: "6 * S * d * d_ff"
  forward_flops_elementwise: "6 * S * d_ff"
  backward_flops_gemm: "12 * S * d * d_ff"
  backward_flops_elementwise: "10 * S * d_ff"
  paper_forward_flops: "6 * S * d * d_ff"
  paper_backward_flops: "12 * S * d * d_ff"
  save_gate: true
  save_up: true
  save_fused_gate_up: false
  save_down_input: true
  elided_temps:
    - sigmoid_gate        # σ(g) — register-only in fused SiLU
    - silu_gate_output    # SiLU(g) — elided when SiLU+mul fused (Liger)
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
    - "ALLOCATE(h) → SAVE(gate, up) when fused_post_gemm else SAVE(silu_out, u)"
    - "ALLOCATE(y) → SAVE(h) on down linear"
  resource_events_backward:
    - "RELEASE(gate, up, h) on backward phase end"
  numerics_tags:
    - silu_special_function_4_flops
    - gated_glu_elementwise
    - appendix_e_gemm_only_paper_compare
capabilities:
  - swiglu
  - gated_ffn
priority: 8
variants:
  - id: region/swiglu/decomposed
    recipe_override:
      save_fused_gate_up: false
      elided_temps: [sigmoid_gate]
  - id: region/swiglu/liger
    recipe_override:
      save_gate: true
      save_up: true
      elided_temps: [sigmoid_gate, silu_gate_output]
  - id: region/swiglu/liger-fused-gate-up
    recipe_override:
      save_fused_gate_up: true
      save_gate: false
      save_up: false
      elided_temps: [sigmoid_gate, silu_gate_output, separate_gate_buffer, separate_up_buffer]
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Shazeer (2020), GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- Ramachandran et al. (2017), Swish / SiLU: https://arxiv.org/abs/1710.05941
- Touvron et al., Llama 2 / Llama 3 (SwiGLU FFN): https://huggingface.co/docs/transformers/model_doc/llama
- HuggingFace `LlamaMLP`: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- HuggingFace `SiLUActivation` / `hidden_act=silu`: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
- Liger-Kernel `swiglu.py` (`LigerSiLUMulFunction`, `LigerFusedGateUpSiLUMulFunction`): https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
- Liger-Kernel paper (Dai et al., 2024): https://arxiv.org/abs/2410.10989
- TRL Liger integration: https://huggingface.co/docs/trl/en/liger_kernel_integration
- TRL kernels hub composition rules: https://huggingface.co/docs/trl/en/kernels_hub
- NVIDIA TE Llama SwiGLU tutorial: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
- PyTorch `Activation.cpp` (SiLU backward reference): https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Activation.cpp
- Zepto SiLU region (gate branch leaf): `docs/kernel/silu.md`, `src/zepto/analysis/lowering/implementations/regions/silu/`
- Zepto linear region: `src/zepto/analysis/lowering/implementations/regions/linear.py`
- Zepto FFN module (ungated reference): `src/zepto/modules/ffn.py`
- Cross-ref: `docs/kernel-implementation.md` §3 (Appendix E SwiGLU \(6 S d d_{\mathrm{ff}}\)), §458 (Liger composition)
- Model gaps (GPT-OSS variant, MoE): `docs/model-architecture-gaps-2026-09-14.md`

**Verification date:** 2026-09-15

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| — | `region/silu` | §3 composable | standalone_activation | **Registered** — gate nonlinearity slice only |
| — | `region/linear` | §4 | MLP stack | **Registered** — gate/up/down GEMMs |
| — | `region/swiglu/decomposed` | §8 | mlp_stack | **Registered** — parent discovery wrapping 3× linear + silu + mul |
| — | `region/swiglu/liger` | §8 variants | mlp_stack | **Registered** — Liger post-GEMM fusion |
| — | `SwiGLU` module | §0 | mlp_stack | **Registered** — `src/zepto/modules/swiglu.py` |
| — | `region/swiglu/liger-fused-gate-up` | §8 variants | mlp_stack | **Registered** — Megatron/TE fused gate-up SAVE policy |

**Open gaps:**
- **`SwiGLU` module registered** — GPT-OSS expert variant still deferred (`docs/model-architecture-gaps-2026-09-14.md`).
- **GPT-OSS expert activation** (`SiLU(1.702×gate) * (up+1)`) — variant; not default `region/swiglu`.
- **MoE routing** — expert SwiGLU FLOPs scale with tokens routed, not registered in this leaf.
- **TE merged gate-up GEMM** — same FLOPs, different HBM; no dedicated Zepto variant yet.
