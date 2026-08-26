# Kernel implementations for Zepto cost modeling

This document is a reference for **how production backends implement the kernels that appear in Apertus / Llama-style decoder stacks**, with enough mathematics and cost structure to derive Zepto **implementations** (including **fused kernels**) without copying a particular CUDA source file.

It is not a Zepto API spec. Pair it with:

- [`CONTEXT.md`](../CONTEXT.md) — glossary (theoretical FLOP, fused kernel, resource event, peak allocated VRAM)
- [`apertus-transformers-implementation.md`](../apertus-transformers-implementation.md) — HuggingFace / vLLM wiring for Apertus and Zepto gaps
- ADR-0001 / ADR-0002 — structural graph vs lowered graph; context-driven implementation selection

**Authority split.** Checkpoint configs and library source decide *what* is computed. Papers decide *why* a fused kernel has a given FLOP and memory complexity. HuggingFace eager Python is the semantic reference; Liger, FlashAttention, vLLM, and optional CUDA wheels are execution strategies that Zepto should model as alternative **implementations** of the same structural operations.

---

## 1. Cost conventions

Zepto’s **theoretical FLOP** counts a multiply-add as **two FLOPs** ([`CONTEXT.md`](../CONTEXT.md)). That matches the Apertus technical report’s Appendix E GEMM convention (`2 * m * n * k` for an \(m \times k\) by \(k \times n\) matmul) rather than the coarse \(6ND\) training approximation ([Hernández-Cano et al., 2025, App. E](https://arxiv.org/abs/2509.14233)).

Notation used throughout (Apertus-8B numbers in parentheses when an example is needed):

| Symbol | Meaning | 8B example |
|--------|---------|------------|
| \(S\) | sequence length (prefill) | 8192 or 65536 |
| \(d\) | hidden size | 4096 |
| \(d_{\mathrm{ff}}\) | MLP width | 21504 |
| \(h\) | query heads | 32 |
| \(h_{\mathrm{kv}}\) | key/value heads | 8 |
| \(d_h\) | head dim | 128 |
| \(L\) | layers | 32 |
| \(V\) | vocab size | 131072 |
| \(e\) | element size in bytes | 2 (bf16), 4 (fp32) |
| \(M\) | on-chip SRAM bytes per SM (FlashAttention analysis) | hardware-specific |

**Two different “costs” must not be mixed:**

1. **Arithmetic intensity / theoretical FLOPs** — closed-form operation counts. Fusion rarely changes the leading GEMM term; it can change elementwise constants and *backward* FLOPs when the kernel recomputes instead of storing.
2. **HBM traffic and peak allocated VRAM** — bytes moved and bytes live. Fusion *does* change this: intermediates that stay in registers/SRAM are not `ALLOCATE`d as global tensors ([Dao et al., 2022](https://arxiv.org/abs/2205.14135); [Dai et al., 2024](https://arxiv.org/abs/2410.10989)).

Apertus Appendix E counts **GEMM + softmax + RMSNorm** and **omits** RoPE trigonometry, residual adds, embedding lookup, and xIELU. Zepto kernel leaves should keep Appendix E numbers for those counted ops, and add explicit elementwise leaves for the omitted kernels when the estimation context is kernel-accurate rather than “paper-FLOP comparable”.

**Worked VRAM unit.** \(1\,\mathrm{GiB} = 2^{30}\) bytes. A bf16 tensor of shape \((h,S,S)\) at \(h{=}32\), \(S{=}8192\) is exactly \(4\,\mathrm{GiB}\).

---

## 2. How to derive a Zepto fused kernel from this file

For each production kernel:

1. Write the **mathematical definition** (this is the structural-graph meaning).
2. Take the **eager PyTorch** decomposition as the unfused baseline (every intermediate is a tensor → one `ALLOCATE` per temp).
3. Take the **fused CUDA/Triton/vLLM** path as a `RegionImplementation`: same public outputs, **elided** internal temps, **explicit** saved backward values (`rstd`, softmax row stats, ReLU mask, …).
4. Set `forward_flops` from the paper/Appendix E leaf, not from summing unfused primitive estimates, unless the backend is literally the eager chain.
5. Record HBM/IO complexity when the kernel is memory-bound (FlashAttention, fused RMSNorm). Wall-clock speedup is *not* a Zepto output; IO complexity explains *why* peak VRAM drops.

Registered Zepto regions today: `region/linear`, `region/layernorm`, `region/relu`. Apertus-critical missing regions are called out per kernel below.

---

## 3. Linear (GEMM)

### Mathematics

A bias-free linear layer (Apertus, PaLM-style: no biases — [Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311); [Hernández-Cano et al., 2025, §2.1](https://arxiv.org/abs/2509.14233)):

\[
Y = X W, \qquad X \in \mathbb{R}^{S \times n_{\mathrm{in}}},\; W \in \mathbb{R}^{n_{\mathrm{in}} \times n_{\mathrm{out}}}.
\]

HuggingFace `nn.Linear` stores \(W^\top\) as `(out, in)` and computes \(X W^\top\). Zepto `Linear` stores `(in, out)` and uses `linear_matmul`. Both realize the same product; layout is an accounting convention, not a FLOP change.

### Programmatic implementations

| Backend | Path | Fusion |
|---------|------|--------|
| HuggingFace eager | `nn.Linear` → cuBLAS / ATen GEMM | Single GEMM kernel; no activation fusion on Apertus linears |
| PyTorch | `F.linear` / `torch.addmm` | Same |
| vLLM | `ColumnParallelLinear` / `RowParallelLinear` / fused `QKVParallelLinear` | Tensor-parallel split; **Q/K/V merged into one GEMM** |
| Hub | A `Linear` hub entry exists in Transformers mappings; **not** attached to Apertus projections | — |

vLLM’s Apertus attention uses one `QKVParallelLinear` then `split`, instead of three `nn.Linear`s ([`vllm/.../apertus.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/apertus.py)). Arithmetic FLOPs are identical to three GEMMs; **kernel-launch and read of \(X\)** drop from three passes to one.

### FLOPs

\[
\mathrm{FLOPs}(Y = XW) = 2\, S\, n_{\mathrm{in}}\, n_{\mathrm{out}}.
\]

Apertus-8B per layer (prefill):

| GEMM | Shape | FLOPs |
|------|--------|-------|
| \(W_Q\) | \(S \times d \times d\) | \(2 S d^2\) |
| \(W_K, W_V\) | \(S \times d \times (h_{\mathrm{kv}} d_h)\) each | \(2 \times 2 S d (h_{\mathrm{kv}} d_h)\) |
| \(W_O\) | \(S \times d \times d\) | \(2 S d^2\) |
| MLP up | \(S \times d \times d_{\mathrm{ff}}\) | \(2 S d\, d_{\mathrm{ff}}\) |
| MLP down | \(S \times d_{\mathrm{ff}} \times d\) | \(2 S d\, d_{\mathrm{ff}}\) |

Ungated MLP total: \(4 S d\, d_{\mathrm{ff}}\) — Appendix E `dense_mlp(..., swiglu=False)`. SwiGLU would be \(6 S d\, d_{\mathrm{ff}}\) (three projections). Apertus uses the ungated formula because xIELU is **not** a gated unit; the paper scales \(d_{\mathrm{ff}}\) by \(1.5\times\) vs SwiGLU to match parameter/FLOP budget ([Hernández-Cano et al., 2025, §2.4](https://arxiv.org/abs/2509.14233)).

### Memory

- **Weights (persistent):** \(n_{\mathrm{in}} n_{\mathrm{out}} e\) per matrix. Dominant model-size term; not fusion-sensitive.
- **Activations:** input \(S n_{\mathrm{in}} e\), output \(S n_{\mathrm{out}} e\). Unfused bias-add would add a temp; Apertus has no bias.
- **Backward:** save \(X\) (and \(W\)) for \(dX = dY W^\top\), \(dW = X^\top dY\). Fused linear+activation (not used on Apertus projections) would save a smaller activation or a mask instead of both GEMM output and activation output.

**Zepto:** `region/linear` already wraps one `linear_matmul`. Do not invent extra temps. Fused QKV is a *future* region that replaces three linears with one leaf whose FLOPs still sum to the three GEMMs but whose peak activation traffic is one read of \(X\).

---

## 4. Embedding lookup

### Mathematics

\[
Y_{s} = E_{:,\, t_s} \quad\text{or}\quad E_{t_s,\,:},
\]

a gather from a table \(E\). HuggingFace `nn.Embedding` is \((V, d)\). Zepto `Embedding` stores \((d, V)\) so an untied LM head can share the layout convention of `Linear`.

### FLOPs and memory

Lookup is **zero theoretical FLOPs** (no arithmetic). Cost is **memory**:

- Table: \(V d e\) persistent (8B: \(131072 \times 4096 \times 2 \approx 1.00\,\mathrm{GiB}\)).
- Output activations: \(S d e\).
- Backward: scatter-add into \(dE\); no extra \(S \times V\) dense Jacobian.

Untied embeddings duplicate this table at the LM head ([Hernández-Cano et al., 2025, §2.1](https://arxiv.org/abs/2509.14233)): **+1 GiB** bf16 for 8B vs tied weights.

---

## 5. RMSNorm

### Mathematics

Zhang and Sennrich ([2019](https://arxiv.org/abs/1910.07467)) drop LayerNorm’s mean centering and keep a root-mean-square scale. HuggingFace Llama/Apertus (γ only):

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \varepsilon}, \qquad
y = \gamma \odot \frac{x}{\mathrm{RMS}(x)}.
\]

Eager HF upcasts to fp32 for the variance, then casts back ([`LlamaRMSNorm` / generated `ApertusRMSNorm`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)). Apertus uses pre-norm around attention and MLP ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745); [Hernández-Cano et al., 2025, §2.1](https://arxiv.org/abs/2509.14233)).

QK-Norm is the **same kernel** on the last axis of headed tensors, with \(n = d_h\) ([Henry et al., 2020](https://arxiv.org/abs/2010.04245); [Dehghani et al., 2023](https://arxiv.org/abs/2302.05449)).

### Programmatic implementations

| Backend | Implementation | What is fused |
|---------|----------------|---------------|
| HF eager | `pow2 → mean → rsqrt → mul γ` in Python | Nothing; full-size temps in autograd |
| HF hub (`USE_HUB_KERNELS`) | `@use_kernel_forward_from_hub("RMSNorm")` → `kernels-community/liger-kernels` **`LigerRMSNorm`** (CUDA/ROCm/NPU); XPU `kernels-community/rmsnorm`; MPS `mlx_rmsnorm` | Normalize + scale in one Triton kernel; **cache RMS / rstd** for backward ([Dai et al., 2024, §RMSNorm](https://arxiv.org/html/2410.10989v1)) |
| vLLM | `RMSNorm` CUDA/IR op; **`fused_add_rms_norm`** when a residual is passed | Residual add + RMSNorm; returns `(y, residual')` so the next block does not re-read a separate residual buffer ([`vllm/.../layernorm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py)) |

Liger reports ~\(7\times\) lower kernel time and ~\(3\times\) lower **peak** memory vs HuggingFace RMSNorm at hidden size 16384, by avoiding materializing intermediates and caching only the RMS statistic ([Dai et al., 2024](https://arxiv.org/abs/2410.10989)).

### FLOPs

Appendix E / Zhang–Sennrich γ-only leaf:

\[
\mathrm{FLOPs}_{\mathrm{RMSNorm}} = 4 \cdot \lvert x \rvert
\]

(`square, mean, rsqrt, scale`). Unfused Zepto decomposition (`square → reduce_sum → divide → add → sqrt → divide → ×γ`) over-counts to \(\approx 5\text{–}6 \cdot \lvert x \rvert\) and **allocates** `squared` and `normalized` at full rank.

QK-Norm Appendix E:

\[
\mathrm{FLOPs}_{\mathrm{QK\text{-}Norm}} = 4\, S\, (h + h_{\mathrm{kv}})\, d_h.
\]

### Memory (forward-lived and saved)

Let \(n = \lvert x \rvert\) (e.g. \(S d\) for hidden RMSNorm).

| Quantity | Eager / unfused | Fused Liger-style | vLLM fused add+norm |
|----------|-----------------|-------------------|---------------------|
| Output \(y\) | \(n e\) | \(n e\) | \(n e\) |
| Full-size temps | \(\sim 1\text{–}2 \times n e\) | **0** (on-chip) | **0** |
| Saved for backward | \(x\) and/or several activations | \(x\) (if grad) + **`rstd` of shape \((S,1)\)** (or per-token) | same + residual already in the returned pair |
| Residual extra read | separate `x + dx` kernel | separate | **elided** |

Apertus-8B, \(S{=}8192\), one hidden RMSNorm: \(S d e = 64\,\mathrm{MiB}\) bf16. Two full-size temps on pre-attn **and** pre-FFN ≈ **256 MiB** of elidable peak if counted as simultaneous — fused `region/rmsnorm` must **replace** the primitive chain so those temps never enter the resource-event stream.

**Zepto:** add `region/rmsnorm` mirroring `region/layernorm` (which already saves `mean` and `inv_std` and uses \(5n\) FLOPs). RMSNorm should save **`rstd` only** (no mean) and use **\(4n\)** FLOPs. Optional `region/fused_add_rmsnorm` for vLLM-comparable residual fusion.

---

## 6. LayerNorm (non-Apertus, existing Zepto region)

Ba et al. ([2016](https://arxiv.org/abs/1607.06450)):

\[
y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta.
\]

One extra reduction (mean) and a shift vs RMSNorm. Zepto `region/layernorm` already uses \(5n\) FLOPs and saves `mean` + `inv_std`. Keep RMSNorm and LayerNorm as **distinct implementations**; do not reuse the LayerNorm leaf for Apertus.

---

## 7. Residual add

\[
z = x + f(x).
\]

Elementwise: \(\mathrm{FLOPs} = \lvert x \rvert\) adds. Memory: if unfused, \(z\) is a new tensor; if fused into RMSNorm (vLLM) or into a decoder-block kernel, \(z\) may alias the residual register/buffer.

Appendix E **does not** count residual adds. Include them in Zepto when matching a backend that launches a separate `add` kernel; omit them inside `fused_add_rms_norm`.

---

## 8. Rotary positional embeddings (RoPE)

### Mathematics

Su et al. ([2021](https://arxiv.org/abs/2104.09864)) rotate pairs of features by a position-dependent angle \(\theta_m = m \omega_i\), \(\omega_i = \Theta^{-2i/d_h}\). HuggingFace eager:

\[
\mathrm{rotate\_half}(x) = (-x_{d_h/2:},\; x_{:d_h/2}), \qquad
\mathrm{RoPE}(x) = x \odot \cos + \mathrm{rotate\_half}(x) \odot \sin.
\]

**Llama 3.1 / Apertus inference frequencies** (`rope_type: llama3` in Transformers `_compute_llama3_parameters`): start from base inverse frequencies, then wavelength-piecewise scale by `factor` (8), `low_freq_factor` (1), `high_freq_factor` (4), `original_max_position_embeddings` (8192). Low-frequency components are divided by `factor`; mid-band interpolates; high-frequency bands are left unchanged. `attention_scaling = 1.0` for this type (unlike YaRN).

**Paper vs checkpoint (do not conflate).** Pretraining uses \(\Theta = 5 \times 10^5\) at \(S{=}4096\) ([Hernández-Cano et al., 2025, §2.1](https://arxiv.org/abs/2509.14233)). Long-context extension **raises \(\Theta\) each stage**; Table 5 ends at **\(\Theta = 12 \times 10^6\)** for 64k context. Released Instruct-2509 configs therefore have `rope_theta: 12000000` plus Llama-3 scaling. Kernel costing uses the **checkpoint** frequencies; wrong \(\Theta\) does not change tensor *shapes* or FLOPs, only numerical angles (Zepto gap G1).

The paper describes this scaling as NTK-aware following the Llama 3 implementation in Transformers ([Peng et al., 2023](https://arxiv.org/abs/2309.00071); [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)).

### Programmatic implementations

| Stage | HF | Hub / CUDA | vLLM |
|-------|----|------------|------|
| **Materialize** \(\cos,\sin\) | `ApertusRotaryEmbedding`: fp32 `inv_freq @ position_ids`, `cat`, `cos`/`sin` | Not hub-replaced (Python) | `get_rope(..., rope_parameters=config.rope_parameters)` builds a cache |
| **Apply** | `apply_rotary_pos_emb` | `@use_kernel_forward_from_hub("rotary_pos_emb")` → `kernels-community/rotary` `apply_rotary_transformers` (CUDA/XPU; ROCm `aiter-rope`) | fused inside the RoPE module on Q/K |

Liger fuses Q and K rotation into one Triton kernel and reports large speedups from removing `rotate_half` temps ([Dai et al., 2024, §RoPE](https://arxiv.org/html/2410.10989v1)).

### FLOPs

Materialize (once per forward, shared across layers):

\[
\underbrace{S \cdot (d_h/2)}_{\text{outer product}} + \underbrace{2 S d_h}_{\sin/\cos} \quad \text{(trig counted as elementwise)}.
\]

Apply per tensor (Q and K):

\[
\approx 6\, h_{\star}\, S\, d_h
\]

(two muls, one add, plus rotate-half as a view/negate/concat — fused kernels fold rotate-half into the same load).

Appendix E **omits** RoPE. Zepto should still bill apply-RoPE when the backend launches it; bill materialize once at model scope (`RoPEMaterialize`).

### Memory

- Persistent / forward-lived cache: \(\cos,\sin\) each \((S, d_h)\) → \(2 S d_h e\). At \(S{=}65536\), \(d_h{=}128\), bf16: **32 MiB** total, independent of \(L\) if shared (HF model-level `rotary_emb`; Zepto `RoPEMaterialize`).
- Unfused apply: temps for `rotate_half` of shape \((h,S,d_h)\). Fused rotary **elides** them.
- Init of `inv_freq` is outside the structural graph (host/setup).

**Zepto:** optional `region/rope_apply` over `split/neg/concat/mul/add`. Do not fuse materialize into apply: HF and vLLM keep a cache.

---

## 9. Grouped-query attention (structure)

### Mathematics

Vaswani et al. ([2017](https://arxiv.org/abs/1706.03762)) scaled dot-product attention:

\[
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V.
\]

Grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) shares each KV head across \(h / h_{\mathrm{kv}}\) query heads. Multi-query attention ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) is the extreme \(h_{\mathrm{kv}}{=}1\). GQA’s purpose is **decode-time HBM**: loading \(K,V\) from cache is bandwidth-bound; fewer KV heads cut that traffic by \(h / h_{\mathrm{kv}}\) (Apertus-8B: **\(4\times\)**). Prefill GEMM FLOPs stay \(\Theta(h S^2 d_h)\) because each query head still attends with dimension \(d_h\).

HF / Zepto eager order for Apertus:

1. Q/K/V projections  
2. Reshape/transpose to \((B, h_\star, S, d_h)\) (Zepto: rank-3 \((h,S,d_h)\))  
3. **QK-Norm** on last dim  
4. **RoPE**  
5. `repeat_kv` (eager) or implicit broadcast (FlashAttention GQA)  
6. Score / softmax / context  
7. Merge + \(W_O\)

### Repeat-KV

Eager: `repeat_interleave` along the head axis, shape \((h_{\mathrm{kv}},S,d_h) \to (h,S,d_h)\). **FLOPs ≈ 0** (copy). **VRAM:** extra full-size K and V at query-head count if materialized. FlashAttention-2 / vLLM `Attention` **do not** materialize the repeat; they index the KV head inside the kernel ([Dao, 2023, §3.1.2](https://arxiv.org/abs/2307.08691)).

**Zepto:** eager `repeat_kv` should remain a copy-cost node unless `region/gqa` / Flash is selected.

---

## 10. Softmax and masked softmax

### Mathematics

Stable softmax along keys:

\[
m_i = \max_j z_{ij}, \quad
p_{ij} = \frac{e^{z_{ij}-m_i}}{\sum_k e^{z_{ik}-m_i}}.
\]

HF eager: `softmax(..., dtype=torch.float32).to(query.dtype)` after `QK^\top * scale + mask` ([`eager_attention_forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)). The fp32 upcast is a **numerics** policy, not extra GEMM FLOPs; it **doubles** the bytes of the softmax tile while that kernel runs.

### FLOPs

Appendix E:

\[
\mathrm{FLOPs}_{\mathrm{softmax}} = 3\, h\, S^2
\]

(exp, sum, divide). Unfused Zepto (`scale → exp → reduce_sum → divide`) is the same leading term plus an optional scale (\(h S^2\)). Mask add is \(h S^2\) extra if not fused.

### Memory

| Path | Live attention matrices | Saved for backward |
|------|-------------------------|--------------------|
| Eager / math SDPA | scores, (optional scaled), exp, weights: **\(\Theta(h S^2)\)** | typically \(P\) (\((h,S,S)\)) |
| Fused masked softmax | **one** \((h,S,S)\) output | \(P\) |
| FlashAttention | **no** \((h,S,S)\) in HBM | row stats \((m,\ell)\) of size \(\Theta(h S)\) ([Dao et al., 2022, Thm. 1](https://arxiv.org/abs/2205.14135)) |

At \(S{=}8192\), \(h{=}32\), bf16: one \((h,S,S)\) = **4.0 GiB**. fp32 softmax tile = **8.0 GiB**. Unfused accounting of 2–3 such buffers is the dominant **false peak** in Zepto today (gap G4b).

**Zepto:** `region/softmax` for standalone `Softmax`; `region/masked_softmax` spanning `add(mask) → exp → sum → div` inside GQA provenance. Tag `numerics="stable_fp32"` without inserting Cast ops unless comparing to HF eager byte-for-byte.

---

## 11. Attention kernels: eager, SDPA, FlashAttention

### 11.1 Standard (eager) attention

Two GEMMs plus softmax:

\[
\mathrm{FLOPs}_{\mathrm{attn, core}}
= \underbrace{2 h S^2 d_h}_{QK^\top}
+ \underbrace{3 h S^2}_{\mathrm{softmax}}
+ \underbrace{2 h S^2 d_h}_{PV}
= 4 h S^2 d_h + 3 h S^2.
\]

HBM: \(\Theta(S d_h + S^2)\) per head because \(S \times S\) is written and reread ([Dao et al., 2022, §2–3](https://arxiv.org/abs/2205.14135)). Peak VRAM is dominated by \(P\) (and often \(S = QK^\top\)).

HF `eager_attention_forward` is this algorithm with `repeat_kv` and fp32 softmax.

### 11.2 Memory-efficient attention (PyTorch SDPA `mem_efficient`)

Rabe and Staats ([2021](https://arxiv.org/abs/2112.05682)) show that the \(S \times S\) matrix need not be stored in HBM: softmax can be accumulated in tiles (online softmax). PyTorch `F.scaled_dot_product_attention` **dispatches** among:

- `math` — eager-like, \(\Theta(S^2)\) memory  
- `flash` — FlashAttention when shapes/dtypes/mask allow  
- `mem_efficient` — xFormers-style tiled attention  

**Implication for Zepto:** an `sdpa` flag is **not** a unique cost model. If the dispatcher picks flash, use §11.3; if math, use §11.1. Worst-case SDPA = eager memory unless the estimation context pins the sub-backend.

### 11.3 FlashAttention (exact, IO-aware)

Dao et al. ([2022](https://arxiv.org/abs/2205.14135)) **tile** \(Q,K,V\) into SRAM-resident blocks, run **online softmax**, and **never write** \(S\) or \(P\) to HBM. Backward **recomputes** attention tiles from saved row statistics \((m,\ell)\) instead of storing \(P\).

**Theorem 1 (Dao et al., 2022):** the algorithm returns exact \(\mathrm{softmax}(QK^\top)V\) with \(\mathcal{O}(S^2 d_h)\) FLOPs and \(\mathcal{O}(S)\) **additional** memory beyond inputs and output.

**IO complexity:** standard attention \(\Theta(S d_h + S^2)\) HBM accesses; FlashAttention \(\Theta(S^2 d_h^2 / M)\). The speedup is from **less HBM traffic**, not fewer GEMM FLOPs. Recomputation **increases** backward FLOPs (extra \(QK^\top\) and softmax) but still wins on wall-clock because HBM is the bottleneck.

Causal masking is applied **inside the tile loop** (skip or mask future keys); a dense \(S \times S\) mask tensor is unnecessary when `is_causal=True`.

### 11.4 FlashAttention-2

Dao ([2023](https://arxiv.org/abs/2307.08691)) keeps the same **linear extra memory** and exactness, and:

- reduces **non-matmul** FLOPs in the inner loop (GPUs’ tensor cores make non-GEMM expensive relative to matmul),
- parallelizes along **sequence** as well as batch/heads (better occupancy at long \(S\), small batch),
- partitions work across warps to cut shared-memory round trips,
- supports **MQA/GQA without repeating KV**.

Reported: ~\(2\times\) vs FlashAttention-1, 50–73% of peak FLOP/s on A100, up to 10–20\(\times\) memory saving vs standard attention at long \(S\).

HuggingFace `attn_implementation="flash_attention_2"` (and v3/v4 when wired) maps to this family. vLLM `Attention` selects a FlashAttention-style backend for Apertus.

### 11.5 Cost table (one layer, Apertus-8B prefill, bf16)

| Backend | Core attn FLOPs | Extra HBM tensors | Peak attn activation (order) |
|---------|-----------------|-------------------|------------------------------|
| Eager HF | \(4 h S^2 d_h + 3 h S^2\) | \(M\) if materialized; \(S,P\) | \(\Theta(h S^2)\) |
| SDPA math | same | same | \(\Theta(h S^2)\) |
| SDPA flash | same leading GEMM | row stats | \(\Theta(h S d_h)\) |
| FlashAttention-2 | same GEMM; fewer scalar FLOPs | \((m,\ell)\): \(\Theta(h S)\) | \(\Theta(h S d_h)\) tiles + QKV |
| vLLM Attention | same | KV cache + tiles | decode: \(\Theta(h \cdot 1 \cdot S_{\mathrm{cache}} d_h)\) compute |

At \(S{=}8192\): dropping \((h,S,S)\) removes **4 GiB per matrix**. Naive unfused estimates with scores + weights + exp can overstate peak by **8–12 GiB per layer**.

**Zepto:** `attention_backend` on the estimation context selects identity-chain vs `region/gqa`. Flash-style region: FLOPs ≈ Appendix E attention (optionally drop the \(3 h S^2\) if folded into the kernel leaf as a non-dominant term), VRAM **without** \((h,S,S)\) `ALLOCATE`s, save \(\Theta(h S)\) stats for training backward if required.

---

## 12. Causal mask

HF `create_causal_mask` builds an **additive** mask (0 / \(-\infty\)) that is batch-, padding-, and cache-length-aware, often rank-4. Zepto `MaterializedCausalMask` is a static \((1,S,S)\) buffer shared across layers.

**FLOPs:** 0 to materialize; \(h S^2\) adds if applied as a separate kernel.

**VRAM:**

\[
\mathrm{bytes}(M) = S^2 e_{\mathrm{mask}}
\]

if fully materialized. At \(S{=}65536\), bf16: **8.0 GiB**; fp32: **16.0 GiB**. Flash / SDPA causal flag: **0** extra \(S \times S\) storage. Padding masks in eager HF may still force a dense additive tensor.

**Zepto:** bill a persistent model-level buffer for the eager path; **do not** multiply by \(L\). For Flash regions, do not allocate \(M\).

---

## 13. xIELU

### Mathematics

Huang and Schlag ([2025](https://arxiv.org/abs/2411.13010)) derive xIELU by integrating affine-transformed ELU gradients. Apertus writes ([Hernández-Cano et al., 2025, §2.1](https://arxiv.org/abs/2509.14233)):

\[
\mathrm{xIELU}(x) =
\begin{cases}
\alpha_p x^{2} + 0.5\, x & x > 0, \\
\alpha_n (e^{x} - 1) - \alpha_n x + 0.5\, x & x \le 0,
\end{cases}
\]

with **per-layer scalar** \(\alpha_p, \alpha_n\). Equivalently, with \(\beta = 0.5\):

\[
\alpha_n (e^{x}-1-x) + \beta x
\quad (x \le 0).
\]

HuggingFace / vLLM eager Python ([`XIELUActivation`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)):

```text
α_p = softplus(α_p_param)
α_n = β + softplus(α_n_param)
y = where(x > 0, α_p x² + β x,
          (expm1(min(x, ε)) - x) α_n + β x)
```

\(\varepsilon = -10^{-6}\) clamps the exponential for stability (Huang & Schlag, §3.4). Parameters are stored in **inverse-softplus** space: \(\alpha_p\) init \(0.8\) as \(\log(\mathrm{expm1}(0.8))\); \(\alpha_n\) param is \(\log(\mathrm{expm1}(\alpha_n^{\mathrm{init}}-\beta))\) because the forward adds \(\beta\) after softplus.

xIELU is **not** a GLU: one activation, two linears. Hidden width is scaled \(1.5\times\) vs SwiGLU to match FLOPs/parameters ([Huang and Schlag, 2025, §3.5](https://arxiv.org/abs/2411.13010); Apertus §2.4).

### Programmatic implementations

| Backend | Path |
|---------|------|
| HF / vLLM Python | `torch.where` + `expm1` (always available) |
| Optional CUDA | `pip install git+https://github.com/nickjbrowning/XIELU` → `torch.classes.xielu.XIELU().forward` when the tensor is CUDA |
| Hub kernels | **Not registered** in Transformers `hub_kernels.py` |

Huang & Schlag §3.5: scalars have negligible memory; quadratic and exponential terms are computed **on the fly** (like GELU/SiLU); eager PyTorch is slower than fused GELU because of branching and multiple kernels; **a fused CUDA kernel** is expected to close that gap. Memory footprint of activations/gradients is comparable to GELU — **one output of size \(\lvert H \rvert\)**, not a persistent extra buffer, if fused.

### FLOPs

Appendix E **omits** xIELU (MLP leaf is GEMM-only). A fused elementwise leaf consistent with the arithmetic is on the order of

\[
8 \cdot S \cdot d_{\mathrm{ff}}
\]

(square, several muls/adds, `expm1` on the negative branch). Unfused Zepto (`where` + `exp` + `minimum` + many `multiply`) reports \(\approx 10\text{–}12 \cdot S d_{\mathrm{ff}}\) plus **full-size branch temps**.

At \(S{=}8192\), \(d_{\mathrm{ff}}{=}21504\): \(\lvert H \rvert \approx 1.76 \times 10^8\) → one bf16 buffer ≈ **336 MiB**. Fused CUDA: output only; optional mask of \(\lvert H \rvert\) bytes if the backward needs a sign mask (same pattern as Zepto `region/relu`).

**Zepto:** optional `region/xielu`. Inference estimates may fold `softplus` into the scalars (current graph inputs `effective_alpha_p` / `effective_alpha_n`). Training estimates may add two scalar `softplus` ops (negligible FLOPs).

---

## 14. LM head

Untied `Linear(d \to V)`: FLOPs \(2 S d V\). Appendix E `final_logits`. Memory: weight \(V d e\) (another ~1 GiB bf16 at 8B) plus logits \(S V e\). At \(S{=}8192\), \(V{=}131072\), bf16 logits ≈ **2.0 GiB** — often comparable to a single attention matrix at the same \(S\). No Apertus-specific fused kernel; Liger `FusedLinearCrossEntropy` exists for **training loss** (not required for forward-only costing).

---

## 15. KV cache (decode, vLLM / HF `use_cache`)

Persistent state, not a fused compute kernel:

\[
\mathrm{bytes}(\mathrm{KV}) = 2\, L\, h_{\mathrm{kv}}\, S_{\max}\, d_h\, e.
\]

Apertus-8B, \(S_{\max}{=}65536\), bf16: **8.0 GiB**. GQA vs MHA: multiply by \(h_{\mathrm{kv}}/h = 1/4\).

Prefill attention FLOPs \(\Theta(h S^2 d_h)\); decode step \(\Theta(h \cdot 1 \cdot S_{\mathrm{cache}} d_h)\) plus GEMMs on one token. FlashAttention extra memory stays linear in the *current* query length \(\times\) cache length tiles, not a new \((S,S)\) matrix.

**Zepto:** model as persistent tensors with `semantic_type="kv_cache"` on a decode **invocation**, not as temps inside prefill lowering (gap G3).

---

## 16. End-to-end backend map (Apertus CUDA inference)

Typical `from_pretrained(..., attn_implementation="flash_attention_2")` with hub kernels:

```text
Embedding            gather (ATen)
RoPE materialize     PyTorch fp32 freq → cos/sin
for each layer:
  RMSNorm            LigerRMSNorm  or  eager fp32 variance
  QKV GEMM           cuBLAS  (vLLM: one fused QKV GEMM)
  Q/K RMSNorm        LigerRMSNorm
  RoPE apply         rotary hub CUDA  or  eager rotate_half
  Attention          FlashAttention-2  (not S×S matmul)
  O GEMM             cuBLAS
  RMSNorm            Liger  or  vLLM fused_add_rms_norm
  Up GEMM            cuBLAS
  xIELU              nickjbrowning CUDA  or  Python where
  Down GEMM          cuBLAS
Final RMSNorm        Liger
LM head              cuBLAS
```

Eager attention replaces the FlashAttention line with `matmul + fp32 softmax + matmul`. Other hub substitutions may still apply.

---

## 17. Recommended Zepto leaves (summary)

Use these closed forms when a fused **implementation** is selected. Unfused identity lowering may keep primitive sums for debugging, but must not be compared to vLLM/HF-flash totals.

| Kernel | FLOP leaf (forward) | Elide in fused VRAM | Save (training) |
|--------|---------------------|---------------------|-----------------|
| Linear / GEMM | \(2 S n_{\mathrm{in}} n_{\mathrm{out}}\) | none (already one kernel) | \(X\) |
| Fused QKV | sum of three GEMMs | two extra reads of \(X\) | \(X\) |
| RMSNorm | \(4 \lvert x \rvert\) | `squared`, full `normalized` | `rstd` \((S,1)\) |
| Fused add+RMSNorm | \(4 \lvert x \rvert + \lvert x \rvert\) | residual temp | `rstd` |
| Softmax | \(3 h S^2\) | exp buffer | \(P\) or Flash stats |
| Masked softmax | \(3 h S^2\) (+ mask add if unfused) | pre-softmax logits | \(P\) |
| Flash GQA | \(4 h S^2 d_h\) (+ small softmax term) | all \((h,S,S)\) | \((m,\ell)\) |
| RoPE apply | \(\approx 6 h_\star S d_h\) | rotate-half temps | — |
| xIELU | \(\approx 8 S d_{\mathrm{ff}}\) | branch/exp temps | output or sign mask |
| Repeat-KV | 0 | full repeated K/V if Flash | — |
| Causal mask (eager) | 0 | — | persistent \(S^2\) |
| Causal (Flash) | 0 | **no** \(S^2\) tensor | — |

---

## 18. Bibliography

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. (2023). GQA: Training generalized multi-query transformer models from multi-head checkpoints. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).

Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).

Chowdhery, A., et al. (2022). PaLM: Scaling language modeling with Pathways. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311).

Dai, Y., Kothapalli, V., Song, Q., Tang, S., Zhu, S., Shimizu, S., Sahni, S., Ning, H., and Chen, Y. (2024). Liger Kernel: Efficient Triton kernels for LLM training. [arXiv:2410.10989](https://arxiv.org/abs/2410.10989). Source: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel). Hub mapping: [transformers `hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py).

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. NeurIPS. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).

Dehghani, M., et al. (2023). Scaling vision transformers to 22 billion parameters. [arXiv:2302.05449](https://arxiv.org/abs/2302.05449).

Grattafiori, A., et al. (2024). The Llama 3 herd of models. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).

Henry, A., Dachapally, P. R., Pawar, S., and Chen, Y. (2020). Query-key normalization for transformers. [arXiv:2010.04245](https://arxiv.org/abs/2010.04245).

Hernández-Cano, A., Hägele, A., Huang, A. H., et al. (2025). Apertus: Democratizing open and compliant LLMs for global language environments. [arXiv:2509.14233](https://arxiv.org/abs/2509.14233). Appendix E: FLOP script. Table 5: long-context RoPE \(\Theta\) schedule.

Huang, A. H., and Schlag, I. (2025). Deriving activation functions using integration. [arXiv:2411.13010](https://arxiv.org/abs/2411.13010). CUDA: [nickjbrowning/XIELU](https://github.com/nickjbrowning/XIELU).

Peng, B., Quesnelle, J., Fan, H., and Shippole, E. (2023). YaRN: Efficient context window extension of large language models. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071). (NTK-aware interpolation lineage used by Llama-3-style RoPE in Transformers.)

Rabe, M. N., and Staats, C. (2021). Self-attention does not need \(O(n^2)\) memory. [arXiv:2112.05682](https://arxiv.org/abs/2112.05682).

Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150).

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

Vaswani, A., et al. (2017). Attention is all you need. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

Xiong, R., et al. (2020). On layer normalization in the transformer architecture. ICML. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745).

Zhang, B., and Sennrich, R. (2019). Root mean square layer normalization. NeurIPS. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).

**Library sources (implementation, not papers):**

- HuggingFace Apertus modular / generated: [`transformers/.../apertus/`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/apertus)
- HuggingFace `XIELUActivation`: [`activations.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)
- Llama-3 RoPE init: [`modeling_rope_utils.py` `_compute_llama3_parameters`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)
- vLLM Apertus: [`vllm/model_executor/models/apertus.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/apertus.py)
- Checkpoints: [Apertus-8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509), [Apertus-70B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509)
- PyTorch SDPA: [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

*Kernel formulas cross-checked 2026-08-26 against Transformers `main`, vLLM `main`, swiss-ai Instruct-2509 configs, and the papers above.*
