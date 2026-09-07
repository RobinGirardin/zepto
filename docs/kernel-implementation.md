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

Registered Zepto regions today: `region/linear`, `region/layernorm`, `region/relu`, `region/rmsnorm`, `region/xielu` (optional; gated on `requested_capabilities={'fused'}`). Apertus-critical missing regions are called out per kernel below.

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
| HF hub — CUDA / ROCm / NPU | `@use_kernel_forward_from_hub("RMSNorm")` → [`kernels-community/liger-kernels`](https://huggingface.co/kernels-community/liger-kernels) **`LigerRMSNorm`** (Triton; training + inference) | Normalize + scale in one Triton kernel; **cache `rstd`** for backward ([Dai et al., 2024, §RMSNorm](https://arxiv.org/html/2410.10989v1)) |
| HF hub — **Intel XPU** | Same decorator → [`kernels-community/rmsnorm`](https://huggingface.co/kernels-community/rmsnorm) **`RMSNorm`** (SYCL/ESIMD C++; **inference only** in [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)) | Same math in one native XPU kernel; **cache `rstd`**; see §5.1 |
| HF hub — **Apple MPS** | Same decorator → [`kernels-community/mlx-rmsnorm`](https://huggingface.co/kernels-community/mlx-rmsnorm) **`RMSNorm`** (Metal; **inference only** in [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py); legacy repo id `mlx_rmsnorm`) | MLX-lineage Metal kernel on PyTorch MPS; **`rstd` stays in threadgroup SLM** (not written to HBM); see §5.2 |
| vLLM | `RMSNorm` CUDA/IR op; **`fused_add_rms_norm`** when a residual is passed | Residual add + RMSNorm; returns `(y, residual')` so the next block does not re-read a separate residual buffer ([`vllm/.../layernorm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py)) |

Liger reports ~\(7\times\) lower kernel time and ~\(3\times\) lower **peak** memory vs HuggingFace RMSNorm at hidden size 16384, by avoiding materializing intermediates and caching only the RMS statistic ([Dai et al., 2024](https://arxiv.org/abs/2410.10989)).

#### 5.1 HF Hub: `kernels-community/rmsnorm` (Intel XPU fallback)

Transformers routes **`RMSNorm` on `device.type == "xpu"`** to this repo when `USE_HUB_KERNELS=YES`. CUDA/ROCm/NPU stay on Liger; Apple Silicon MPS uses **`kernels-community/mlx-rmsnorm`** (§5.2). This is **not** a second CUDA backend — it is the **default XPU inference** kernel in the hub registry.

| Property | Detail |
|----------|--------|
| **Hub repo** | [`kernels-community/rmsnorm`](https://huggingface.co/kernels-community/rmsnorm) (Apache-2.0; built with [`kernel-builder`](https://github.com/huggingface/kernels)) |
| **Load path** | `from kernels import get_kernel` → `get_kernel("kernels-community/rmsnorm")` → `apply_rms_norm_forward` / `apply_rms_norm_backward` |
| **Transformers wiring** | `hub_kernels.py`: `"xpu": { Mode.INFERENCE: LayerRepository(repo_id="kernels-community/rmsnorm", layer_name="RMSNorm", version=1) }` — **no `Mode.TRAINING` entry**; XPU training falls back to eager PyTorch |
| **Implementation** | Precompiled **SYCL ESIMD** extension (`torch.ops._rmsnorm_xpu_*::apply_rms_norm`), not Triton. Builds exist for Torch 2.10–2.29 × XPU 2025.x (Linux/Windows) plus **CPU** wheels (likely CI/dev fallback). Lineage overlaps Intel XPU norm work (e.g. [`intel/llm-scaler` omni_xpu_kernel](https://github.com/intel/llm-scaler) ESIMD RMSNorm / `fused_add_rms_norm`). |
| **Public API** | `RMSNorm` module + `RMSNormFunction` autograd wrapper; ops return `(output, rstd)` from forward |
| **Fusion scope** | Single kernel: per-row \(\sum x_i^2 \to \mathrm{rstd} \to y = x \cdot \mathrm{rstd} \cdot \gamma\) with on-chip reduction (SLM/sub-group), no full-rank `squared` / `normalized` HBM temps |
| **vs Liger on XPU** | Liger added XPU Triton tuning ([Liger-Kernel #653](https://github.com/linkedin/Liger-Kernel/pull/653): `grf_mode`, warp/stage counts), but Transformers **still defaults XPU inference to `kernels-community/rmsnorm`**, not `LigerRMSNorm`. Zepto should treat them as **device-routed variants of the same fused leaf** (\(4n\) FLOPs, elided temps), not competing math. |

**Autograd saved tensors (hub XPU vs Liger):**

| Kernel | Saved for backward (training) | Notes |
|--------|-------------------------------|-------|
| HF eager | \(x\), intermediate activations along the 7-op chain | Highest HBM footprint |
| Liger | \(x\) + **`rstd`** per row (fp32); optional \(W\) if affine | Does **not** save output \(y\) |
| **`kernels-community/rmsnorm`** | **`hidden_states`, `weight`, `output`, `rstd`** (four tensors) | Saves **full output \(y\)** in addition to \(x\) and `rstd` — heavier than Liger for training, though Transformers only registers this path for **XPU inference** today |

**Execution constraints:** last-axis normalization on 2D views `(B·T, H)` or `(B, T, H)`; dtypes fp16/bf16/fp32 per build; contiguity assumed by custom op. Intel ESIMD paths in related code often require `hidden_size` divisible by 32 and \(\le 8192\) for tiled dispatch (model-specific one-WI paths exist for odd sizes like QK-Norm \(d_h{=}128\)).

**Zepto variant note:** Same closed-form **`region/rmsnorm/liger`** recipe applies (\(4 \lvert x \rvert\) forward FLOPs, ALLOCATE `y` + `rstd`, SAVE `rstd` only). A future `region/rmsnorm/hub-xpu` descriptor id is optional if backend routing must distinguish XPU hub from CUDA Liger; cost model is identical, only the **saved-tensor policy in real PyTorch autograd** differs (see table above).

#### 5.2 HF Hub: `kernels-community/mlx-rmsnorm` (Apple MPS / Metal fallback)

Transformers routes **`RMSNorm` on `device.type == "mps"`** to this repo when `USE_HUB_KERNELS=YES`. CUDA/ROCm/NPU stay on Liger; Intel XPU uses `kernels-community/rmsnorm` (§5.1). This is **not** a native MLX-array backend — it is a **PyTorch MPS custom op** whose Metal shader is ported from [MLX `rms_norm.metal`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal).

| Property | Detail |
|----------|--------|
| **Hub repo** | [`kernels-community/mlx-rmsnorm`](https://huggingface.co/kernels-community/mlx-rmsnorm) (MIT; built with [`kernel-builder`](https://github.com/huggingface/kernels) **`metal`** backend). Source: [`kernels-community/mlx-rmsnorm`](https://github.com/huggingface/kernels-community/tree/main/mlx-rmsnorm). Older underscore id `kernels-community/mlx_rmsnorm` is deprecated ([Transformers #46293](https://github.com/huggingface/transformers/pull/46293)). |
| **Load path** | `from kernels import get_kernel` → `get_kernel("kernels-community/mlx-rmsnorm", version=1)` → `rmsnorm_forward` / `rmsnorm_backward` or `layers.RMSNorm` |
| **Transformers wiring** | `hub_kernels.py`: `"mps": { Mode.INFERENCE: LayerRepository(repo_id="kernels-community/mlx_rmsnorm", layer_name="RMSNorm", version=1) }` — **no `Mode.TRAINING` entry**; MPS training falls back to eager PyTorch (same policy as XPU hub) |
| **Implementation** | Precompiled **Metal** extension (`torch.ops._mlx_rmsnorm_metal_*::launch_forward_kernel` / `launch_backward_kernel`, `at::kMPS` dispatch). Wheels under `build/torch*-metal-aarch64-darwin/`. Embedded metallib from `mlx_rmsnorm/rmsnorm.metal` + Objective-C++ launcher (`metal_rmsnorm.mm`). |
| **Public API** | `rmsnorm_forward(x, weight, epsilon)` returns **output only** (no `(output, rstd)` tuple). `rmsnorm_backward` recomputes the row normalizer inside the VJP Metal kernel. |
| **Fusion scope** | One dispatch per row batch: \(\sum x_i^2\) via `simd_sum` → `local_inv_mean[0] = rsqrt(acc/axis_size + eps)` in **threadgroup SLM** → \(y_i = w_i \cdot x_i \cdot \mathrm{inv\_mean}\). No full-rank `squared` / `normalized` HBM temps. |
| **Dispatch variants** | `axis_size ≤ 4096`: `rms{float32\|float16\|bfloat16}` / `vjp_rms*`. Larger last dim: `rms_looped*` / `vjp_rms_looped*` (same math, tiled loop over the axis). |
| **vs Liger on MPS** | Liger targets CUDA/ROCm/NPU Triton; Transformers **does not** route MPS to `LigerRMSNorm`. Zepto should treat MLX-hub Metal and Liger as **device-routed variants of the same fused leaf** (\(4n\) forward FLOPs, elided temps), not competing math. |

**Autograd / saved activations (hub MPS vs Liger vs XPU hub):**

| Kernel | Saved for backward (training) | Notes |
|--------|-------------------------------|-------|
| Liger | \(x\) + **`rstd`** per row (fp32) | Writes `rstd` to HBM for backward |
| **`kernels-community/rmsnorm`** (XPU) | \(x\) + **`rstd`** + **`y`** | Heavier autograd surface; hub registry inference-only on XPU |
| **`kernels-community/mlx-rmsnorm`** (MPS) | **`x`** (and **`weight`** if affine grad) only | Forward keeps **`inv_mean` in threadgroup memory only** — **no `rstd` tensor in HBM**. Backward **recomputes** \(\sum x_i^2\) and `rsqrt` from saved \(x\) in the VJP kernel (recompute-style, like FlashAttention backward). Hub registry is **MPS inference-only** today; backward exists in the package but is not selected by default `hub_kernels.py`. |

**Execution constraints:** tensors must be on **`mps`** device; launcher copies to **contiguous** storage if needed. Supported dtypes: fp16, bf16, fp32. Last-axis normalization on views flattened to `(B·T, H)` inside `rmsnorm_forward`. Not torch.compile-friendly (same note as Liger RMSNorm in `hub_kernels.py`).

**Zepto variant note:** Same closed-form **`region/rmsnorm/liger`** recipe applies for FLOPs and elided forward temps (\(4 \lvert x \rvert\), no `squared`/`normalized` `ALLOCATE`s). Zepto’s fused leaf still models **`SAVE rstd`** for training backward (\(\approx S \times 4\) bytes fp32 per row) because that is the Liger/hub-XPU convention and matches `RMSNormRecipe.rstd_shape`. A future `region/rmsnorm/hub-mps` descriptor id is optional if backend routing must distinguish MPS Metal from CUDA Liger; **peak VRAM during real MPS inference** is one output buffer only (no `rstd` HBM write on the forward path).

### FLOPs

Let \(n = \lvert x \rvert\). For a hidden RMSNorm, \(X \in \mathbb{R}^{S \times d}\) so \(n = Sd\). Per row of length \(d\):

\[
\mathrm{rstd} = \Bigl(\tfrac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon\Bigr)^{-1/2},
\qquad
y_i = \gamma_i \cdot x_i \cdot \mathrm{rstd}.
\]

**Appendix E / Zhang–Sennrich γ-only leaf** (what fused `region/rmsnorm` bills — `RMSNormRecipe.forward_flops_per_element = 4`):

\[
\mathrm{FLOPs}_{\mathrm{RMSNorm}} = 4 \cdot \lvert x \rvert = 4Sd.
\]

Four *leading* stages, each one FLOP per element of \(X\):

| Stage | Arithmetic | Per row | For \(X\) |
|-------|------------|---------|-----------|
| **Square** | \(x_i^2\) | \(d\) muls | \(Sd\) |
| **Mean** | \(\sum x_i^2\) then \(\times 1/d\) | \(d-1\) adds + 1 divide \(= d\) | \(Sd\) |
| **rsqrt applied** | \(x_i \cdot \mathrm{rstd}\) | \(d\) muls | \(Sd\) |
| **Scale** | \(y_i \cdot \gamma_i\) | \(d\) muls | \(Sd\) |
| **Total (leaf)** | | \(4d\) | \(4Sd\) |

**Mean is \(Sd\), not \(Sd + S\).** Summing \(d\) numbers takes \(d-1\) additions; the `/d` is one divide per row. They cancel:

\[
S(d-1) + S = Sd.
\]

Billing the reduction as \(d\) adds *and* adding the divide would double-count the \(+S\). Zepto’s primitives already match this: `ReduceSum` is `numel(input) - numel(output)` \(= Sd - S\); the following `Divide` on the \((S,1)\) variance is \(S\). Together they are exactly \(Sd\).

The mnemonic’s “rsqrt” is the **normalize multiply** \(x \odot \mathrm{rstd}\), not the scalar reciprocal-sqrt. `rstd` itself is one value per row.

**Row-scalar work dropped by the leaf.** Add \(\varepsilon\) and `sqrt`/`rsqrt` are \(O(S)\), not \(O(Sd)\). Identity lowering of the HF eager chain in [`rms_norm.py`](../src/zepto/modules/rms_norm.py) (`cast → square → reduce_sum → divide → add → sqrt → divide → cast → ×γ`) therefore sums to:

\[
4Sd + 2S
\]

(casts bill 0 FLOPs). Breakdown:

| Primitive | Shape billed | FLOPs |
|-----------|--------------|-------|
| square (`Multiply`) | \((S,d)\) | \(Sd\) |
| `ReduceSum` | \(Sd - S\) | \(Sd - S\) |
| mean `Divide` (`/d`) | \((S,1)\) | \(S\) |
| `Add`(\(\varepsilon\)) | \((S,1)\) | \(S\) |
| `SquareRoot` | \((S,1)\) | \(S\) |
| normalize `Divide` | \((S,d)\) | \(Sd\) |
| \(\times\gamma\) | \((S,d)\) | \(Sd\) |
| Casts | — | \(0\) |
| **Identity total** | | \(4Sd + 2S\) |

The extra \(2S\) is `add(ε)` + `sqrt` on the \((S,1)\) variance — **not** the mean’s `/d`. The fused leaf keeps the \(4Sd\) leading term and drops those \(O(S)\) scalars. **Do not** analogize this drop to softmax’s row-max: `sqrt`/`rstd` is one scalar per row, while softmax `max` is a full-rank reduction over keys (\(\Theta(h S^2)\), §10). Oracle: identity sum in `tests/test_rmsnorm_identity_lowering.py`; fused \(4n\) in `tests/lowering/regions/test_rmsnorm.py`.

An older 7-op decomposition that billed some of those scalars as full-rank ops is what earlier notes called \(\approx 5\text{–}6 \cdot \lvert x \rvert\). Unfused execution still **allocates** `squared` and `normalized` at full rank; that is a VRAM difference, not a reason to change the fused leaf’s FLOP count. Fusion (Liger / hub) does **not** go below \(4n\) arithmetic.

QK-Norm is the same kernel on the last axis, so still \(4\cdot\lvert x \rvert\):

\[
\mathrm{FLOPs}_{\mathrm{QK\text{-}Norm}} = 4\, S\, (h + h_{\mathrm{kv}})\, d_h.
\]

LayerNorm is \(5n\) because it adds mean-centering \(x - \mu\) (\(n\) more subtracts). RMSNorm has no mean, so one less.

### Memory (forward-lived and saved)

Let \(n = \lvert x \rvert\) (e.g. \(S d\) for hidden RMSNorm).

| Quantity | Eager / unfused | Fused Liger-style (CUDA/ROCm/NPU) | HF hub XPU (`kernels-community/rmsnorm`) | HF hub MPS (`kernels-community/mlx-rmsnorm`) | vLLM fused add+norm |
|----------|-----------------|-----------------------------------|------------------------------------------|----------------------------------------------|---------------------|
| Output \(y\) | \(n e\) | \(n e\) | \(n e\) | \(n e\) | \(n e\) |
| Full-size temps | \(\sim 1\text{–}2 \times n e\) | **0** (on-chip) | **0** (ESIMD SLM) | **0** (Metal threadgroup SLM) | **0** |
| Saved for backward | \(x\) and/or several activations | \(x\) (if grad) + **`rstd`** \((S,1)\) fp32 per row | \(x\) + **`rstd`** + **`y`** (+ \(W\) if grad) in autograd; hub registry is inference-only on XPU | **`x`** only if autograd enabled; **`rstd` not materialized** (recomputed in VJP); hub registry is inference-only on MPS | same + residual already in the returned pair |
| Residual extra read | separate `x + dx` kernel | separate | separate | separate | **elided** |

**Arithmetic intensity.** RMSNorm is **memory-bound** at LLM hidden sizes: forward reads \(x\) and \(\gamma\), writes \(y\), and (for Liger / XPU hub) writes one fp32 scalar per row for `rstd`. Fused kernels win on wall-clock by keeping the reduction and scale in **SRAM/SLM** (Liger Triton registers; XPU ESIMD SLM + sub-group reduce; MPS Metal `threadgroup float local_inv_mean[1]`), not by lowering theoretical FLOPs below \(4n\).

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

In GQA, masked softmax is the step that turns attention **logits** \(z\) into **weights** \(P\) along the key axis. It sits after \(QK^\top\) (and optional scale + mask) and before the \(PV\) GEMM. Production stacks rarely expose it as a standalone op: it is either decomposed in eager Python, fused as **scale + mask + softmax** in NVIDIA training kernels, or absorbed entirely inside **FlashAttention / SDPA** (online softmax with structural causal masking).

### Mathematics

Attention logits (one query row \(i\), keys \(j\)):

\[
z_{ij} = \frac{(QK^\top)_{ij}}{\sqrt{d_h}} + M_{ij}, \qquad
m_i = \max_j z_{ij}, \qquad
p_{ij} = \frac{e^{z_{ij}-m_i}}{\sum_k e^{z_{ik}-m_i}}.
\]

\(M\) is an **additive** mask: causal attention sets forbidden positions to a large negative value (HF uses \(-\infty\) in fp32 softmax; in practice a dtype minimum or \(-10^9\)). Tensor shape is \((h, S, S)\) or batched \((B, h, S, S)\); softmax reduces along the **last** axis (keys).

HF eager runs `softmax(..., dtype=torch.float32).to(query.dtype)` on masked logits ([`eager_attention_forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)). The fp32 upcast is a **numerics** policy, not extra GEMM FLOPs; it **doubles** the bytes of the softmax tile while that kernel runs.

### Programmatic implementations

Four **fusion boundaries** recur below. Hub and serving kernels plug into these; there is no Hub `kernels-community/*softmax*` that is a standalone *attention* softmax.

| Boundary | What is fused | Still writes \(P\) / logits? | Zepto region |
|----------|---------------|------------------------------|--------------|
| **A** Standalone softmax | max / exp / sum / div on a materialized tile | **Yes** | `region/softmax` |
| **B** Scaled masked softmax | scale + additive mask + A | **Yes** (\(P\) only) | `region/masked_softmax` |
| **C** Full attention (online softmax) | \(QK^\top\) + softmax + \(PV\) | **No** \((h,S,S)\) | `region/gqa` variants |
| **D** LM-head fused CE | GEMM + vocab softmax + CE | **No** full \((S,V)\) | `region/linear_ce` |

| Backend | Path | Boundary | Mask handling | Standalone masked softmax? |
|---------|------|----------|---------------|----------------------------|
| **PyTorch `F.softmax` (CUDA)** | `torch.nn.functional.softmax` → ATen `aten::softmax` (native `SoftMax.cu`; cuDNN optional/legacy) | A | N/A | Yes — fused kernel, not Python `exp→sum→div` |
| **PyTorch eager (decomposed)** | Manual `exp → sum → div` or Zepto primitives | A (unfused) | Materialized mask if used in GQA | Yes |
| **HF Transformers eager** | `matmul → add(causal_mask) → softmax(fp32)` | A after separate mask add | Materialized causal mask buffer | Yes |
| **PyTorch SDPA** | `F.scaled_dot_product_attention` — **dispatcher**, not one kernel (§11.2) | C if `flash` / `mem_efficient`; A+GEMM if `math` | Structural causal when `is_causal=True` | No when fused |
| **Megatron-LM / Apex** | `scaled_masked_softmax_cuda`, `scaled_upper_triang_masked_softmax_cuda`, also `ScaledSoftmax` (scale only) and `SoftmaxOne` (attention-sink) ([`fused_softmax.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py)) | B | Fused scale + mask + softmax; CUDA kernel has **shape/seq-len fallbacks** to `torch.softmax` | Yes (fused training primitive) |
| **NVIDIA TransformerEngine** | `te_softmax::scaled_masked_softmax_fwd` ([TE softmax](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/softmax.py)) | B | Same as Megatron family | Yes |
| **FlashAttention 2/3/4** | Tiled \(Q,K,V\) with **online softmax** | C | Causal in tile loop; **no** dense \(S \times S\) ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) | No |
| **HF Hub — CUDA Flash** | `kernels-community/flash-attn2`, `flash-attn3`, `flash-attn4`, `vllm-flash-attn3` | C | Same as FlashAttention | No |
| **HF Hub — MPS Flash** | [`kernels-community/metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa) | C | Structural causal; GQA/MQA; varlen; see §11.8 | No |
| **HF Hub — paged decode** | [`kernels-community/paged-attention`](https://huggingface.co/kernels-community/paged-attention) | C (decode) | Paged KV; \(S_q{=}1\); see §11.9 | No |
| **HF Hub — ROCm softmax** | [`kernels-community/aiter-kernels`](https://huggingface.co/kernels-community/aiter-kernels) `softmax` | A | No fused causal mask in this op; Flash lives in `aiter-flash-attn`; see §10.1 | Yes (ROCm) |
| **HF Hub — SageAttention** | [`kernels-community/sage-attention`](https://huggingface.co/kernels-community/sage-attention) / `sage-blackwell` | C (quantized QK) | Same elision as Flash; INT8/FP8 bytes; see §11.11 | No |
| **FlashInfer** | vLLM / SGLang serving kernels ([intro](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)) | C (+ optional vocab sampling softmax) | Prefill / decode / append; paged KV; fused RoPE; see §11.9 | No |
| **Liger Kernel (attention)** | No dedicated masked-softmax Triton op | — | Speedups from **FlashAttention integration** ([Liger-Kernel](https://github.com/linkedin/Liger-Kernel)) | N/A |
| **Liger `FusedLinearCrossEntropy`** | Token-chunked GEMM + vocab softmax + CE ([`fused_linear_cross_entropy.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)) | D | Vocab axis \(V\), not keys \(S\); **chunk peak ≠ 0**; see §14 | No |
| **TRL Hub fused linear CE** | [`trl-lib/fused-linear-ce`](https://huggingface.co/trl-lib/fused-linear-ce) | D | Same op as Liger; **vocab-tiled** rather than token-chunked; see §14.1 | No |
| **Zepto (today)** | `MatMul → Add(mask) → Softmax` in `GroupedQueryAttention` ([`gqa.py`](../src/zepto/modules/gqa.py), [`softmax.py`](../src/zepto/modules/softmax.py)) | A unfused | Materialized causal mask | Decomposed primitives |
| **Zepto (registered)** | `region/softmax` (boundary A) | A | Elide `exp_scores`; stable 5× leaf | Fused region leaf |
| **Zepto (planned)** | `region/masked_softmax`, `region/gqa/*`, `region/linear_ce` | B / C / D | Elide pre-softmax logits or the whole \((h,S,S)\) tile | Fused region leaves |

**Composition rules.** `region/masked_softmax` (boundary B) and `region/gqa` / Flash / Metal-Flash / paged / Sage (boundary C) are **mutually exclusive** on the same attention layer — C replaces score + softmax + context. Liger patches (RMSNorm, CE, SwiGLU, …) compose with FlashAttention Hub kernels per [TRL kernels hub docs](https://huggingface.co/docs/trl/en/kernels_hub); they do not replace attention masked softmax. Boundary **D** (`LigerFusedLinearCrossEntropyLoss`, `trl-lib/fused-linear-ce`) is orthogonal: vocab-axis training CE, not causal masking over keys (§14). AITER `softmax` is boundary **A** on ROCm and does **not** replace C (`aiter-flash-attn`).

#### PyTorch fused `F.softmax` vs decomposed eager

Calling `F.softmax(x, dim=-1)` on CUDA does **not** execute three separate ATen ops. PyTorch lowers to a single **`aten::softmax`** dispatch ([native `SoftMax.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu); cuDNN is optional/legacy). That is **one kernel launch**, not necessarily one DRAM pass.

Naive stable softmax walks each row **three times** in HBM (max, exp+sum, normalize). Triton’s fused-softmax tutorial measures naive PyTorch traffic as **read \(5MN+2M\), write \(3MN+2M\)** — about \(4\times\) a kernel that keeps the row in SRAM ([Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)). ATen’s CUDA softmax still often **reloads the row**; the “read \(z\) once, write \(P\) once” bound holds only when the row fits in SRAM.

Worked row size (bf16, last dim \(= S\)): \(S{=}8192\) → **16 KiB** (fits typical SRAM); \(S{=}65536\) → **128 KiB** (may spill). Do **not** bill `aten::softmax` as a guaranteed 1-pass kernel at 64k context.

Zepto’s unfused `Softmax` module still models **separate** `Exp` / `ReduceSum` / `Divide` primitives. That chain also **omits stable max-subtract** (`src/zepto/modules/softmax.py`), so it is not numerically the same as ATen/HF softmax — only a structural decomposition. `F.softmax` is already a fused softmax **kernel**, just not fused with the preceding mask add or \(QK^\top\) matmul.

#### Implicit causal mask (`is_causal=True`) — concrete example

**Eager path** materializes an additive mask \(M \in \mathbb{R}^{S \times S}\) and adds it before softmax. For \(S=4\):

\[
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}, \qquad
z = \frac{QK^\top}{\sqrt{d_h}} + M, \qquad P = \mathrm{softmax}(z).
\]

Query row \(i=2\) (third token) attends only to keys \(j \in \{0,1,2\}\); positions \(j > i\) are \(-\infty\) so \(p_{ij} = 0\).

**SDPA / FlashAttention** with `is_causal=True` never allocates \(M\). The backend encodes causality **structurally**:

1. **Skip work:** for query row \(i\), only key columns \(j \le i\) participate in the online softmax loop; future columns are not loaded from HBM.
2. **In-register mask:** when a tile straddles the diagonal, the kernel applies \(-\infty\) (or a large negative) to forbidden entries inside SRAM — same numerics as adding \(M\), but **no \(S^2\) mask tensor**.

Example: query row \(i=2\), keys \(j=0..3\). Eager computes four logits then zeros \(j=3\) via mask add. SDPA-flash computes softmax over \(\{z_{2,0}, z_{2,1}, z_{2,2}\}\) only — mathematically identical to `softmax([z20, z21, z22, -inf])` without ever storing \(-\infty\) in global memory.

**Zepto accounting:** eager GQA bills **persistent** \((1,S,S)\) mask storage + \(h S^2\) mask-add FLOPs; SDPA/Flash regions bill **0** mask bytes and **0** mask-add FLOPs (causality is control-flow inside the attention kernel).

### FLOPs — forward derivation

Work on a logits tile of shape \((h, S, S)\). Count **one FLOP per elementwise op** on that tile; reductions bill like RMSNorm (sum over \(S\) keys per \((h,S)\) row).

**Unfused Zepto GQA score path** (today):

| Step | Primitive | Per \((h,S,S)\) element | FLOPs |
|------|-----------|-------------------------|-------|
| 1. Scale | `Multiply` by \(1/\sqrt{d_h}\) | 1 mul | \(h S^2\) |
| 2. Mask | `Add` materialized mask | 1 add | \(h S^2\) |
| 3. Exp | `Exp` | 1 exp (1 FLOP in Zepto leaf convention) | \(h S^2\) |
| 4. Sum | `ReduceSum` over keys | \(S-1\) adds per row → \(S\) per row (same reduction billing as §5 mean) | \(h S^2\) |
| 5. Normalize | `Divide` by row sum | 1 div | \(h S^2\) |
| **Unfused total (score + softmax)** | | | **\(5 h S^2\)** |

The \(QK^\top\) GEMM (\(2 h S^2 d_h\) FLOPs) is separate; it lives in §11, not in the softmax leaf. This identity chain **does not subtract a row-max** — it matches Zepto `Softmax`, not ATen/HF stable softmax.

**Stable softmax is five leading stages, not three.** The row-max \(m_i\) is a reduction over \(S\) keys per query row — **\(h S^2\) comparisons**, billed like `ReduceSum` (§5 mean rule: \(S-1\) ops per row → \(S\) per row). It is **not** \(O(hS)\). The RMSNorm `sqrt` analogy is wrong: `sqrt`/`rstd` is one scalar per row; softmax max is full-rank. Subtracting \(m_i\) is another \(h S^2\) elementwise op.

| Stage | Arithmetic | FLOPs |
|-------|------------|-------|
| **Max** | \(m_i = \max_j z_{ij}\) | \(h S^2\) |
| **Subtract** | \(z_{ij} - m_i\) | \(h S^2\) |
| **Exp** | \(e^{z_{ij}-m_i}\) | \(h S^2\) |
| **Sum** | \(\sum_k e^{z_{ik}-m_i}\) (reduction billing) | \(h S^2\) |
| **Divide** | normalize by row sum | \(h S^2\) |
| **Stable softmax total** | | **\(5 h S^2\)** |

**Fused `region/softmax` (boundary A)** bills **\(5 h S^2\)** — those five stages. That is the kernel-accurate leaf for ATen/AITER/Triton fused softmax.

**Fused `region/masked_softmax` (boundary B)** bills **\(5 h S^2\)** as well: scale + mask + exp + sum + div, matching the unfused GQA score-path total above. Those five ops run **inside one kernel** (Megatron / TE `scaled_masked_softmax`): they are real arithmetic, not free, but they do **not** increment the leaf beyond \(5 h S^2\) and they do **not** allocate extra \((h,S,S)\) tensors. A Megatron kernel that is also *numerically stable* still executes max+subtract on-chip (\(+2 h S^2\) hardware work, no extra ALLOCATE). Zepto’s billed fused leaf stays **\(5 h S^2\)**; the unfused identity path already counted scale+mask as two of those five.

Apertus Appendix E writes \(\mathrm{FLOPs}_{\mathrm{softmax}} = 3 h S^2\) (exp + sum + div only) inside the attention core (§11.1). That paper formula is **not** the Zepto fused-region leaf.

**HF eager + fp32 softmax:** same arithmetic FLOPs as the fused leaf; the fp32 cast is **0 FLOPs**. Bytes: the softmax **working set** is \(2\times\) a bf16 tile (8.0 GiB at \(h{=}32\), \(S{=}8192\)). That 8 GiB is tile size, not a unique extra peak — if scores (4 GiB bf16), fp32 weights (8 GiB), and a downcast \(P\) (4 GiB) overlap, peak can approach **~16 GiB**.

### FLOPs — backward

VJP with saved output \(P\): \(dZ = P \odot (dY - \langle P, dY \rangle)\) — a row dot product, a subtract, and a multiply:

\[
\mathrm{FLOPs}_{\mathrm{softmax,bwd}} \approx 4\, h S^2.
\]

(The older \(2\times\) forward heuristic \(6 h S^2\) is a bound, not this derivation.) If pre-softmax logits \(z\) are saved instead of \(P\), the order is the same. **FlashAttention** does not store \(P\): backward **recomputes** attention tiles from saved row statistics ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)). Saved-state memory drops to \(\Theta(h S)\), but backward FLOPs **increase** (extra \(QK^\top\) and softmax passes inside tiles) — a deliberate trade for HBM bandwidth.

### Arithmetic intensity

Softmax on the full \((h,S,S)\) tile is **memory-bound** (HBM bandwidth limited, not Tensor Core limited).

Kernel-accurate fused softmax (bf16, \(e=2\)), **ideal** 1-pass (row in SRAM):

| Quantity | Value |
|----------|-------|
| FLOPs | \(5 h S^2\) |
| Minimum HBM traffic (read \(z\), write \(P\)) | \(2 \times h S^2 \times e = 4 h S^2\) bytes |
| Arithmetic intensity | \(5/4 = 1.25\) FLOP/byte |

The previous \(3/4 = 0.75\) figure used the three-stage leaf and the same 1-pass traffic; it is a **lower bound**, not `aten::softmax` as shipped (see 3-pass vs SRAM-resident above). Unfused eager with separate `exp` and `sum` buffers: **multiple read/write passes** over \(\Theta(h S^2)\) tensors → effective AI **drops**. Fusion wins on wall-clock by keeping max / exp / sum / div (and optionally scale / mask) in **SRAM / registers**, not by reducing theoretical FLOPs.

Vocab-axis softmax (LM head, tile \((S,V)\)) has the same 1-pass bound: \(\mathrm{AI} = 3/(2e) = 0.75\) FLOP/byte at bf16 if billed at \(3SV\) FLOPs, or \(5/(2e)\) if billed at \(5SV\). Do **not** write \(3/(4e)\) — that applies \(e\) twice.

### Memory — forward-lived tensors

Let one \((h,S,S)\) bf16 tensor at \(h{=}32\), \(S{=}8192\) be **4.0 GiB** ([§1 worked unit](#1-cost-conventions)).

| Path | Live \((h,S,S)\)-scale buffers during forward | Notes |
|------|-----------------------------------------------|-------|
| **Unfused Zepto / HF eager** | `scores`, optional `scaled_scores`, `exp_scores`, `weights` (\(P\)) | **2–3×** peak vs output alone if temps overlap in the resource stream |
| **HF eager + fp32 softmax** | Same + fp32 working set | **8.0 GiB** tile bytes; peak can exceed that if bf16 scores/\(P\) still live |
| **Fused `region/masked_softmax`** | **`P` only** | Pre-softmax logits elided on-chip |
| **Megatron / TE fused softmax** | **`P` only** (same accounting) | Scale + mask fused in CUDA |
| **AITER `softmax` (ROCm)** | **`P` only** | Boundary A; see §10.1 |
| **FlashAttention / SDPA-flash / Metal-Flash / Sage** | **none** | Output is \((h,S,d_h)\) context; online softmax in SRAM (§11.8–§11.11) |
| **Paged attention / FlashInfer decode** | **none** of size \(S\times S\) | Softmax over \(S_{\mathrm{cache}}\) with \(S_q{=}1\); see §11.9 |

**Identity lowering of Zepto `Softmax`** ([`softmax.py`](../src/zepto/modules/softmax.py)): optional `scale → exp → reduce_sum → divide`. Each intermediate is a full-rank \((h,S,S)\) `ALLOCATE` unless a fused region replaces the chain.

**GQA provenance** ([`gqa.py`](../src/zepto/modules/gqa.py)): `MatMul(Q,K^T) → Add(causal_mask) → Softmax`. Unfused peak can naively sum **scores + masked_scores + exp + weights** — the dominant **false peak** in Zepto cost models today (gap **G4b**).

### Memory — saved activations (training backward)

| Path | Saved for autograd | Size (order) |
|------|-------------------|--------------|
| Eager / fused masked softmax | Output **\(P\)** | \(\Theta(h S^2)\) |
| HF eager (typical) | **\(P\)** after fp32 softmax | \(\Theta(h S^2)\) |
| FlashAttention-style (boundary C) | Row stats per query row | \(\Theta(h S)\) ([Dao et al., 2022, Thm. 1](https://arxiv.org/abs/2205.14135)); see §11.6 |

Materialized causal mask \(M\): **persistent** \((1,S,S)\) or \((S,S)\) in eager HF (not recomputed each forward); Flash / structural causal paths bill **0** extra \(S^2\) mask storage (see §17 summary table).

### Numerical example — Apertus-8B GQA, one layer, prefill

Config: \(h{=}32\), \(h_{\mathrm{kv}}{=}8\), \(d_h{=}128\), \(S{=}8192\), bf16 (\(e{=}2\)). Formulas omit batch \(B\) unless stated; at \(B{=}4\), \(S{=}4096\) multiply \((h,S,S)\) bytes by \(B\).

| Quantity | Calculation | Result |
|----------|-------------|--------|
| One \((h,S,S)\) buffer | \(32 \times 8192^2 \times 2\) B | **4.0 GiB** |
| fp32 softmax working set (HF eager) | \(4.0\,\mathrm{GiB} \times 2\) | **8.0 GiB** |
| Unfused peak (2–3 simultaneous \((h,S,S)\) temps) | \(2\text{–}3 \times 4.0\,\mathrm{GiB}\) | **8–12 GiB** false peak |
| Fused `region/masked_softmax` forward | \(P\) only | **4.0 GiB** |
| Boundary-C attention (Flash / Metal-Flash / Sage) | no \((h,S,S)\) | **0** for scores/\(P\) |

At \(S{=}4096\), \(B{=}4\): one batched \((B,h,S,S)\) buffer is \(4 \times 32 \times 4096^2 \times 2\) B **≈ 4 GiB**; unfused GQA can account for **12+ GiB** per attention step before fusion — comparable in severity to RMSNorm temp inflation (§5), but scaling as **\(S^2\)** instead of \(S d\).

### Zepto regions

| Region | Scope | FLOP leaf | VRAM elision |
|--------|-------|-----------|--------------|
| `region/softmax` | Standalone `Softmax` / ATen / AITER | **\(5 h S^2\)** (stable softmax) | `exp` buffer; do not assume 1 HBM pass at \(S{=}64k\) |
| `region/masked_softmax` | GQA score step (Megatron/TE) | **\(5 h S^2\)** (scale+mask+exp+sum+div) | pre-softmax logits (`scores`, `scaled_scores`, `exp_scores`) |
| `region/gqa` / Flash / Metal-Flash / Sage | Full attention (boundary C) | §11.3–§11.11 | all \((h,S,S)\) temps |
| `region/gqa/paged` | Decode / paged KV | §11.9 | no \(S\times S\); softmax over \(S_{\mathrm{cache}}\) |
| `region/linear_ce` | LM-head training CE (boundary D) | \(2 S d V + 3 S V\) | chunk logits, **not** 0; §14 |

Tag `numerics="stable_fp32"` on HF-comparable estimates without inserting Cast ops unless comparing byte-for-byte to eager Transformers. **Implementation order** (gap G4): `region/rmsnorm` → `region/softmax` + `region/masked_softmax` → golden vs Atto at \(S \in \{8192, 65536\}\) → `region/gqa` backend variants (§11.8–§11.11) → `region/linear_ce`.

#### 10.1 HF Hub: `kernels-community/aiter-kernels` (`softmax`)

ROCm Triton package vendored from [ROCm/aiter](https://github.com/ROCm/aiter) ([Hub](https://huggingface.co/kernels-community/aiter-kernels), [source README](https://github.com/huggingface/kernels-community/blob/main/aiter-kernels/README.md)). This is the **standalone softmax Liger does not ship**. Flash Attention is **intentionally excluded** and lives in `kernels-community/aiter-flash-attn` / `aiter-flash-attn-ck`.

| Property | Detail |
|----------|--------|
| **Fusion boundary** | **A** — fused row softmax; still writes \(P\) |
| **vs `F.softmax`** | Same leaf (\(5 h S^2\), elide `exp`); device = ROCm (verified gfx942 MI300X, gfx950 MI355X) |
| **vs Megatron B** | No fused causal mask in `aiter_kernels.softmax`; mask stays a separate kernel unless AITER Flash (C) is selected |
| **vs Liger** | Liger has no attention softmax; AITER has softmax **and** a separate Flash package |
| **Zepto** | Route `region/softmax` here when `context.backend` is ROCm; do **not** treat it as `region/gqa` |

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

The \(3 h S^2\) softmax term is the **Apertus Appendix E** paper formula (exp + sum + div). Kernel-accurate fused softmax is **\(5 h S^2\)** (§10). Paper-comparable attention totals in this section keep Appendix E; `region/softmax` / `region/masked_softmax` use the §10 leaves.

HF `eager_attention_forward` is this algorithm with `repeat_kv` and fp32 softmax.

### 11.2 PyTorch SDPA is a dispatcher, not one cost model

Rabe and Staats ([2021](https://arxiv.org/abs/2112.05682)) show that the \(S \times S\) matrix need not be stored in HBM: softmax can be accumulated in tiles (online softmax). PyTorch `F.scaled_dot_product_attention` **dispatches** among several backends; an `sdpa` flag is **not** a unique cost model ([PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)):

| Sub-backend | Fusion boundary | Memory class | Zepto leaf |
|---------------|-----------------|--------------|------------|
| `math` | Eager GEMM + softmax | \(\Theta(S^2)\) — use §11.1 | `region/gqa` eager / identity |
| `flash` | C (FlashAttention) | no \((h,S,S)\) — use §11.3 | `region/gqa/flash2` |
| `mem_efficient` | C (xFormers-style tiled) | no dense \(P\); not identical to Flash IO | `region/gqa` mem-efficient variant if pinned |
| cuDNN FMHA | C on NVIDIA | same asymptotic class as Flash when selected | pin explicitly; do not assume |
| FlexAttention | C (compiled block-mask) | depends on mask sparsity | out of scope until requested |

**Implication for Zepto:** worst-case SDPA = eager memory unless the estimation context pins the sub-backend (`attention_backend` per [ADR-0009](adr/0009-backend-profiles-and-attention-selection.md); optional `sdpa_mode` in `state`). On **MPS**, the dispatcher often falls through to `math` / `mem_efficient` unless [`kernels-community/metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa) is loaded (§11.8).

### 11.3 FlashAttention (exact, IO-aware)

Dao et al. ([2022](https://arxiv.org/abs/2205.14135)) **tile** \(Q,K,V\) into SRAM-resident blocks, run **online softmax**, and **never write** \(S\) or \(P\) to HBM. Backward **recomputes** attention tiles from saved row statistics \((m,\ell)\) instead of storing \(P\).

**Theorem 1 (Dao et al., 2022):** the algorithm returns exact \(\mathrm{softmax}(QK^\top)V\) with \(\mathcal{O}(S^2 d_h)\) FLOPs and \(\mathcal{O}(S)\) **additional** memory beyond inputs and output.

**IO complexity:** standard attention \(\Theta(S d_h + S^2)\) HBM accesses; FlashAttention \(\Theta(S^2 d_h^2 / M)\). The speedup is from **less HBM traffic**, not fewer GEMM FLOPs. Recomputation **increases** backward FLOPs (extra \(QK^\top\) and softmax) but still wins on wall-clock because HBM is the bottleneck.

Causal masking is applied **inside the tile loop** (skip or mask future keys); a dense \(S \times S\) mask tensor is unnecessary when `is_causal=True`. Paper-comparable FLOPs in §11.1 still count the full \(S\times S\) (eager also forms the upper triangle then masks). Kernel-accurate causal Flash **executes** about half the QK/PV tiles; optional causal factor \(\approx 1/2\) on `region/gqa/flash*` when the estimation context is kernel-accurate rather than paper-comparable.

### 11.4 FlashAttention-2

Dao ([2023](https://arxiv.org/abs/2307.08691)) keeps the same **linear extra memory** and exactness, and:

- reduces **non-matmul** FLOPs in the inner loop (GPUs’ tensor cores make non-GEMM expensive relative to matmul),
- parallelizes along **sequence** as well as batch/heads (better occupancy at long \(S\), small batch),
- partitions work across warps to cut shared-memory round trips,
- supports **MQA/GQA without repeating KV**.

Reported: ~\(2\times\) vs FlashAttention-1, 50–73% of peak FLOP/s on A100, up to 10–20\(\times\) memory saving vs standard attention at long \(S\).

HuggingFace `attn_implementation="flash_attention_2"` maps to this family. vLLM `Attention` selects a FlashAttention-style backend for Apertus.

### 11.5 FlashAttention-3

Shah et al. ([2024](https://arxiv.org/abs/2407.08608); [PyTorch blog](https://pytorch.org/blog/flashattention-3)) reimplements the FlashAttention tiling algorithm for **NVIDIA Hopper** (H100/H800): warp-specialized producer–consumer pipelining (TMA + WGMMA), inter-/intra-warpgroup overlap of block GEMMs with softmax, and optional **FP8** forward via block quantization and incoherent processing. Package path: [`Dao-AILab/flash-attention` `hopper/`](https://github.com/Dao-AILab/flash-attention) (`flash_attn_3.flash_attn_interface`) or Hub [`kernels-community/flash-attn3`](https://huggingface.co/kernels-community/flash-attn3).

**Memory model.** FA3 inherits the **same asymptotic memory** as FA1/FA2 (§11.3, Theorem 1): exact attention with \(\mathcal{O}(S)\) **additional** HBM beyond Q, K, V, and output. FA3 does **not** introduce a new memory-complexity class; it reduces **HBM bytes moved** and **wall-clock** on Hopper by better utilizing SRAM and overlapping scalar softmax with Tensor Core GEMMs. Reported H100 throughput: ~1.5–2.0\(\times\) vs FA2 in FP16/BF16 (~740 TFLOPs/s forward); FP8 forward approaches ~1.2 PFLOPs/s ([Shah et al., 2024](https://arxiv.org/abs/2407.08608)). **Requirements:** Hopper GPU, CUDA \(\ge\) 12.3; FP16/BF16 forward+backward; FP8 **forward only** in the current beta.

For Zepto costing, treat **FlashAttention-2 and FlashAttention-3 as one fused leaf** (`region/gqa` / `attention_backend=flash`): identical FLOP leaf and VRAM elision rules (§11.6–11.7). Do not bill FA3-specific speedup as extra VRAM savings beyond FA2. **FlashAttention-4** (`kernels-community/flash-attn4`, `flash_attn_func` / `flash_attn_varlen_func`) is the same **VRAM class** (boundary C, no \((h,S,S)\)); treat it as another `region/gqa` variant, not a new memory-complexity class. Hub [`vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3) is FA3 packaged for vLLM (varlen / serving layouts) — same bf16 elision, distinct packaging.

### 11.6 Memory accounting — eager vs FlashAttention-2/3

This section derives the **peak allocated VRAM** and **saved autograd state** used in §11.7 and in `region/gqa` lowering. Two quantities must stay separate ([§1](#1-cost-conventions)):

1. **Peak allocated VRAM** — global tensors live in HBM at some point during forward or backward.
2. **HBM IO (bytes moved)** — total read/write traffic; FlashAttention lowers this via tiling even when input/output tensor sizes match eager.

#### 11.6.1 Eager HF GQA — what gets allocated

HF `eager_attention_forward` runs `matmul → add(causal_mask) → softmax(fp32) → matmul` with `repeat_kv` ([§9](#9-grouped-query-attention-structure), [§10](#10-softmax-and-masked-softmax)). Per layer, prefill, batch \(B\):

**Persistent (model scope, not × \(L\)):**

| Tensor | Shape | Bytes |
|--------|-------|-------|
| Causal mask \(M\) | \((1,S,S)\) or \((B,1,S,S)\) | \(S^2 e_{\mathrm{mask}}\) |

At \(S{=}8192\), bf16 mask: **128 MiB** ([§12](#12-causal-mask)). At \(S{=}65536\): **8.0 GiB** bf16.

**Forward inputs/outputs (linear in \(S\)):**

| Tensor | Shape | Order |
|--------|-------|-------|
| Q | \((B,h,S,d_h)\) | \(B h S d_h e\) |
| K, V (pre-`repeat_kv`) | \((B,h_{\mathrm{kv}},S,d_h)\) | \(2 B h_{\mathrm{kv}} S d_h e\) |
| K, V (post-`repeat_kv`) | \((B,h,S,d_h)\) each | **+\(2 B (h - h_{\mathrm{kv}}) S d_h e\)** copy vs compact KV |
| Context | \((B,h,S,d_h)\) | \(B h S d_h e\) |

**Forward temps — dominant \(\Theta(B h S^2)\):**

| Tensor | Shape | Role |
|--------|-------|------|
| Scores \(QK^\top\) | \((B,h,S,S)\) | Pre-softmax logits |
| Masked scores | \((B,h,S,S)\) | Optional separate buffer after mask add |
| fp32 softmax tile | \((B,h,S,S)\) fp32 | HF numerics during `softmax(..., dtype=float32)` — **2×** bytes vs bf16 tile ([§10](#10-softmax-and-masked-softmax)) |
| Weights \(P\) | \((B,h,S,S)\) | Attention probabilities before \(PV\) |

**Worked unit.** One \((h,S,S)\) bf16 buffer at \(h{=}32\), \(S{=}8192\): \(32 \times 8192^2 \times 2\) B = **4.0 GiB** ([§1](#1-cost-conventions)).

Naive peak if 2–3 \((h,S,S)\) temps overlap in the resource stream: **8–12 GiB per layer** ([§10 numerical example](#numerical-example--apertus-8b-gqa-one-layer-prefill)).

**Saved for backward (training):**

| Path | Saved | Size |
|------|-------|------|
| HF eager (typical) | Output **\(P\)** after fp32 softmax | \(\Theta(B h S^2)\) |
| Alternative | Pre-softmax logits \(z\) | same order |

Autograd needs the full \(S \times S\) tile (or equivalent) for the softmax VJP.

**HBM IO (leading order, per layer):**

\[
\mathrm{IO}_{\mathrm{eager}} = \Theta(B S d_h + B h S^2)
\]

The \(S^2\) term comes from writing scores, reading for softmax, writing \(P\), reading \(P\) for \(PV\) ([Dao et al., 2022, §2–3](https://arxiv.org/abs/2205.14135)).

#### 11.6.2 FlashAttention-2/3 — what gets allocated

FA2 and FA3 share the **same tiling algorithm** for memory (§11.3). Q, K, V blocks reside in SRAM; **online softmax** maintains running row max \(m_i\) and sum \(\ell_i\); dense scores and \(P\) are **never written to HBM**.

**Persistent:**

| Item | Eager | Flash FA2/FA3 |
|------|-------|---------------|
| Causal mask \(M\) | \((1,S,S)\) in HBM | **0** — structural `is_causal=True` in tile loop ([§10](#10-softmax-and-masked-softmax)) |

**Forward inputs/outputs:**

| Tensor | Shape | Notes |
|--------|-------|-------|
| Q | \((B,h,S,d_h)\) | Same as eager |
| K, V | \((B,h_{\mathrm{kv}},S,d_h)\) | **No `repeat_kv`** — GQA indexes KV head inside kernel ([Dao, 2023, §3.1.2](https://arxiv.org/abs/2307.08691)) |
| Context | \((B,h,S,d_h)\) | Same as eager |
| Scores / \(P\) | — | **0 HBM** — tile buffers in SRAM only |

**Saved for backward (training, FP16/BF16):**

| Path | Saved | Size |
|------|-------|------|
| FlashAttention | Row stats **\((m,\ell)\)** per query row (fp32) | \(\Theta(B h S)\) |
| | **Not** \(P\) | Backward **recomputes** tiles from \((m,\ell)\) ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) |

\[
\mathrm{bytes}_{(m,\ell)} = B \cdot h \cdot S \cdot 2 \cdot 4
\]

(two fp32 scalars per query row). At \(h{=}32\), \(S{=}8192\), \(B{=}1\): **≈ 2 MiB** ([§10 table](#numerical-example--apertus-8b-gqa-one-layer-prefill)).

**FP8 forward (FA3 beta):** block-quantization scales add \(\Theta(\text{blocks})\) metadata in HBM; still **no** \((S,S)\) tensor. FP8 backward is not in the current beta — training backward uses FP16/BF16 FA3 path when enabled.

**HBM IO:**

\[
\mathrm{IO}_{\mathrm{Flash}} = \Theta\!\left(\frac{B h S^2 d_h^2}{M}\right)
\]

where \(M\) = on-chip SRAM bytes per SM ([§1 notation](#1-cost-conventions)). FA3 improves **utilization** of this tiled path on Hopper; the complexity class matches FA2.

#### 11.6.3 Side-by-side summary (one layer, prefill)

| Category | Eager HF GQA | FlashAttention-2/3 |
|----------|--------------|---------------------|
| Persistent causal mask | \(\Theta(S^2)\) | **0** |
| Q, K, V, context | \(\Theta(B h S d_h)\) | \(\Theta(B h S d_h)\); K/V at \(h_{\mathrm{kv}}\) (smaller) |
| `repeat_kv` expansion | **+\(\Theta(B h S d_h)\)** if materialized | **0** |
| Forward \((h,S,S)\) temps | \(\Theta(B h S^2)\) — scores, \(P\), fp32 tile | **0** in HBM |
| Saved backward state | \(\Theta(B h S^2)\) — typically \(P\) | \(\Theta(B h S)\) — \((m,\ell)\) |
| Peak attn activation (order) | \(\Theta(B h S^2)\) | \(\Theta(B h S d_h)\) |

**Scaling with \(S\).** Eager peak attn memory grows as **\(S^2\)** (dominated by scores + \(P\)). Flash peak attn memory grows as **\(S\)** (Q/K/V/context + row stats). At \(S{=}65536\), one \((h,S,S)\) bf16 matrix at \(h{=}32\) is **256 GiB**; Flash-style attention at the same \(S\) keeps attention-specific peak at **\(\Theta(h S d_h)\)** (hundreds of MiB for Q+K+V+output at \(B{=}1\)), which is why long-context stacks require Flash-style kernels.

#### 11.6.4 Numerical example — Apertus-8B GQA, one layer

Config: \(B{=}1\), \(h{=}32\), \(h_{\mathrm{kv}}{=}8\), \(d_h{=}128\), \(S{=}8192\), bf16 (\(e{=}2\)). Cross-checks [§10 numerical example](#numerical-example--apertus-8b-gqa-one-layer-prefill).

| Item | Eager | FA2/FA3 |
|------|-------|---------|
| Q | \(32 \times 8192 \times 128 \times 2\) → **64 MiB** | **64 MiB** |
| K, V at \(h_{\mathrm{kv}}\) | **32 MiB** | **32 MiB** |
| K, V after `repeat_kv` | **+96 MiB** (128 MiB total KV) | **0** (no repeat) |
| One \((h,S,S)\) bf16 buffer | **4.0 GiB** | **0** |
| fp32 softmax tile (transient working set) | **8.0 GiB** | **0** |
| Naive peak (2–3 \((h,S,S)\) temps) | **8–12 GiB** | **0** |
| Causal mask (persistent, all layers) | **128 MiB** | **0** |
| Saved backward | **≈ 4.0 GiB** (\(P\)) | **≈ 2 MiB** (\(m,\ell\)) |

**Per-layer delta at this \(S\):** Flash elides **~8–12 GiB** of forward \((h,S,S)\) peak and **~4 GiB** of saved \(P\) vs eager. Across \(L{=}32\) layers, eager backward can retain **\(P\) per layer** → **~\(\,128\,\mathrm{GiB}\)** of \(S^2\) activation storage at this \(S\); Flash saves **~\(\,64\,\mathrm{MiB}\)** total row stats (same order per layer, \(\Theta(hS)\) not \(\Theta(hS^2)\)).

#### 11.6.5 What FlashAttention-3 does *not* save

These are **unchanged** vs eager and must still be billed:

- **Weight matrices** \(W_Q, W_K, W_V, W_O\) — persistent model weights.
- **Q, K, V activations** after projections — FA3 still reads them; K/V are smaller at \(h_{\mathrm{kv}}\) without `repeat_kv`.
- **Context output** \((B,h,S,d_h)\).
- **KV cache** in decode — separate from FA3 kernel memory; persistent \(\Theta(L \cdot h_{\mathrm{kv}} \cdot S_{\max} \cdot d_h)\) ([§15](#15-kv-cache-decode-vllm--hf-use_cache)).

FA3 removes the **quadratic-in-\(S\)** activation footprint (scores, weights, fp32 softmax tile, saved \(P\)), not linear Q/K/V/output storage.

#### 11.6.6 Zepto `region/gqa` rules (from this accounting)

1. **Eager path:** keep `MatMul → Add(mask) → Softmax` with every \((h,S,S)\) intermediate as an `ALLOCATE` unless a narrower fused region replaces the chain (gap **G4b** in [`apertus-transformers-implementation.md`](../apertus-transformers-implementation.md)).
2. **Flash path (FA2 or FA3):** **no** \((h,S,S)\) `ALLOCATE`s; bill causal mask **0** bytes; omit `repeat_kv` copy; save \(\Theta(hS)\) \((m,\ell)\) for training backward, not \(P\).
3. **Do not** multiply eager causal mask by \(L\); **do not** allocate \(M\) for Flash regions ([§12](#12-causal-mask)).
4. **FLOPs** stay at Appendix E attention (\(4 h S^2 d_h + 3 h S^2\)) for both paths — fusion changes **VRAM**, not leading GEMM FLOPs ([§11.3](#113-flashattention-exact-io-aware)).

### 11.7 Cost table (one layer, Apertus-8B prefill, bf16)

| Backend | Core attn FLOPs | Extra HBM tensors | Peak attn activation (order) |
|---------|-----------------|-------------------|------------------------------|
| Eager HF | \(4 h S^2 d_h + 3 h S^2\) | \(M\) if materialized; scores, \(P\) | \(\Theta(h S^2)\) |
| SDPA math | same | same | \(\Theta(h S^2)\) |
| SDPA flash | same leading GEMM | \((m,\ell)\): \(\Theta(h S)\) | \(\Theta(h S d_h)\) |
| FlashAttention-2 | same GEMM; fewer scalar FLOPs | \((m,\ell)\): \(\Theta(h S)\) | \(\Theta(h S d_h)\) |
| **FlashAttention-3** | **same as FA2** | **same as FA2** | **same as FA2** |
| **FlashAttention-4** | **same as FA2** (bf16 class) | **same as FA2** | **same as FA2** |
| vLLM Attention / FlashInfer prefill | same | KV cache + tiles | decode: \(\Theta(h \cdot 1 \cdot S_{\mathrm{cache}} d_h)\) compute |
| **Metal-Flash SDPA** | **same as FA2** | **same as FA2** (MPS) | **same as FA2** |
| **Paged attention** | decode \(\Theta(h S_{\mathrm{cache}} d_h)\) | paged KV; no \(S\times S\) | \(\Theta(h S_{\mathrm{cache}} d_h)\) |
| **SageAttention** | same leading GEMM | no \(P\); quantized QK bytes | \(\Theta(h S d_h)\) |

At \(S{=}8192\): dropping \((h,S,S)\) removes **4 GiB per matrix**. Naive unfused estimates with scores + weights + exp can overstate peak by **8–12 GiB per layer** ([§11.6.4](#1164-numerical-example--apertus-8b-gqa-one-layer)).

**Zepto:** `attention_backend` on the estimation context selects identity-chain vs `region/gqa`. Flash-style region (FA2 / FA3 / FA4 / Metal-Flash / Sage): FLOPs ≈ Appendix E attention (optionally drop the \(3 h S^2\) if folded into the kernel leaf as a non-dominant term), VRAM per §11.6 — **without** \((h,S,S)\) `ALLOCATE`s, save \(\Theta(h S)\) stats for training backward if required. Paged decode is a **different work tile** (§11.9), not a drop-in for prefill Flash.

### 11.8 HF Hub: `kernels-community/metal-flash-sdpa` (Apple MPS)

Metal Flash-style SDPA for PyTorch MPS ([Hub](https://huggingface.co/kernels-community/metal-flash-sdpa), [source README](https://github.com/huggingface/kernels-community/blob/main/metal-flash-sdpa/README.md)). Inspired by FlashAttention; some shaders from MLX. Transformers serving already references this repo. This is the MPS analogue of “SDPA picked `flash`” — **not** a standalone softmax, and **not** `mlx-rmsnorm` (that is RMSNorm only, §5.2).

| Property | Detail |
|----------|--------|
| **Fusion boundary** | **C** — varlen Flash SDPA; online softmax; no dense \((h,S,S)\) |
| **API** | `flash_attention_varlen` / `flash_attn_varlen_func` (`cu_seqlens_q/k`, causal flag, optional softcap) |
| **Features** | GQA/MQA; causal; fp16/bf16/fp32; head dims 32, 64, 72, 80, 96, **128**, 256 |
| **FLOPs** | Same leading terms as FA2 (§11.1 / §11.3): \(4 h S^2 d_h + 3 h S^2\) paper-comparable; no extra softmax leaf |
| **VRAM** | Same class as FA2: output \((h,S,d_h)\); **0** scores/\(P\) in HBM; **0** materialized causal \(M\) when `do_causal=True` |
| **vs PyTorch SDPA on MPS** | Unpinned SDPA on MPS often falls through to `math` / `mem_efficient` and **can still materialize** scores. Without this variant, an MPS estimate stays on the 8–12 GiB eager false peak (§10 G4b) |
| **vs `region/masked_softmax`** | Mutually exclusive — this region replaces the whole score path |
| **Zepto** | `region/gqa/metal-flash` (or `attention_backend` MPS-flash profile). Same elision rules as `region/gqa/flash2`; device tag `mps` |

### 11.9 Paged decode: Hub `paged-attention` and FlashInfer

Prefill Flash (FA2/Metal-Flash) softmaxes an \(S\times S\) tile in SRAM. **Decode** softmaxes **one new query** against **\(S_{\mathrm{cache}}\)** keys. That is a different work tile; prefill \(4 h S^2 d_h\) does not apply.

#### Hub `kernels-community/paged-attention`

vLLM / mistral.rs paged-attention CUDA ([Hub](https://huggingface.co/kernels-community/paged-attention), [README](https://github.com/huggingface/kernels-community/blob/main/paged-attention/README.md)). Query shape `(num_seqs, h, d_h)` (\(S_q{=}1\)); K/V in **pages** `(num_blocks, h, d_h, block_size)` plus `block_tables` / `seq_lens`. Ops: `paged_attention_v1` / `v2`, `reshape_and_cache`, `copy_blocks`, `swap_blocks`, FP8 helpers.

**FLOPs (one decode step, one layer),** counting a mul-add as two FLOPs like GEMM:

\[
\mathrm{FLOPs}_{\mathrm{paged, step}} \approx 4\, h\, S_{\mathrm{cache}}\, d_h
\]

(\(QK^\top\) and \(PV\) with \(S_q{=}1\)). Softmax over \(S_{\mathrm{cache}}\) is \(\Theta(h S_{\mathrm{cache}})\) — dominated by the GEMMs at LLM \(d_h\). No \(S\times S\) matrix.

**VRAM:** no \((h,S,S)\) ALLOCATE. Persistent KV is the paged cache in §15, not a kernel temp. Output is `(num_seqs, h, d_h)`.

**vs FA2 prefill:** same online-softmax idea; **different shapes**. **vs eager GQA:** no `repeat_kv`, no causal \(M\), no \(P\). Pair with §15: persistent KV bytes × this FLOP leaf.

#### FlashInfer (vLLM / SGLang)

Serving library ([FlashInfer intro](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html); [NVIDIA blog](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer)). Three attention stages the catalog otherwise folds into one “FA2” row:

| Stage | Query length | Bound | Zepto leaf |
|-------|--------------|-------|------------|
| Prefill | \(S\) | compute-ish | reuse `region/gqa` Flash class |
| Decode | 1 | **IO-bound** (operational intensity \(\approx O(1)\)) | same as paged-attention |
| Append / spec-decode | few tokens | mixed | missing from eager/FA2 prefill costing |

Extras vs FA2: fused RoPE inside attention, GQA without `repeat_kv`, quantized KV, cascade attention for shared prefixes. Also a **logits-processor softmax** (temperature → softmax → top-p → sample): vocab-axis **inference**, not Liger training CE (§14). Bill that as a small \(3 S V\) (usually \(S{=}1\)) sampling leaf if the invocation samples; do not elide LM-head logits on the inference path.

**Zepto:** `region/gqa/paged` for decode invocations (`semantic_type="kv_cache"`). Do not use the prefill Flash leaf for \(S_q{=}1\).

### 11.10 Hub serving FA3/FA4 packaging

| Hub repo | Relation to §11.5 | Zepto |
|----------|-------------------|--------|
| `kernels-community/flash-attn3` | FA3 reference Hub build | `region/gqa/flash3` — same VRAM class as FA2 |
| `kernels-community/vllm-flash-attn3` | FA3 **for vLLM** (varlen, extra LSE workspace) | same class; distinct id if packaging must be attributed |
| `kernels-community/flash-attn4` | FA4 (`flash_attn_func`, varlen) | `region/gqa/flash4` — same bf16 elision; different Hopper/Blackwell schedule |
| `kernels-community/sgl-flash-attn3` | SGLang FA3 build | same class |

FP8 FA3/FA4 **forward** halves Q/K/V element size versus bf16; that is a different **IO** class, not extra VRAM savings beyond “no \(P\)”.

### 11.11 SageAttention (quantized boundary C)

[`kernels-community/sage-attention`](https://huggingface.co/kernels-community/sage-attention) / `sage-blackwell`: `sageattn`, `per_block_int8`, etc. Still **no dense \(P\)** (boundary C). QK (and sometimes PV accumulators) use INT8/FP8, so **HBM bytes** change while the FLOP order stays \(4 h S^2 d_h\). Optional `region/gqa/sage` variant; not `region/masked_softmax`.

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
\alpha_p x^{2} + \beta x & x > 0, \\
\alpha_n (e^{\min(x,\varepsilon)} - 1 - x) + \beta x & x \le 0,
\end{cases}
\]

with **per-layer scalar** \(\alpha_p, \alpha_n\), \(\beta = 0.5\), and \(\varepsilon = -10^{-6}\). Learned parameters map through softplus:

\[
\alpha_p = \mathrm{softplus}(\alpha_{p,\mathrm{param}}), \qquad
\alpha_n = \beta + \mathrm{softplus}(\alpha_{n,\mathrm{param}}).
\]

HuggingFace / vLLM eager Python ([`XIELUActivation`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)):

```text
α_p = softplus(α_p_param)
α_n = β + softplus(α_n_param)
y = where(x > 0, α_p x² + β x,
          (expm1(min(x, ε)) - x) α_n + β x)
```

\(\varepsilon\) clamps the exponential for stability (Huang & Schlag, §3.4). Parameters are stored in **inverse-softplus** space: \(\alpha_p\) init \(0.8\) as \(\log(\mathrm{expm1}(0.8))\); \(\alpha_n\) param is \(\log(\mathrm{expm1}(\alpha_n^{\mathrm{init}}-\beta))\) because the forward adds \(\beta\) after softplus.

xIELU is **not** a GLU: one activation, two linears (`Up GEMM → xIELU → Down GEMM`). Hidden width is scaled \(1.5\times\) vs SwiGLU to match FLOPs/parameters ([Huang and Schlag, 2025, §3.5](https://arxiv.org/abs/2411.13010); Apertus §2.4).

### Programmatic implementations

| Backend | Path | Fused? | Notes |
|---------|------|--------|-------|
| HF eager Python | [`XIELUActivation`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py): `torch.where` + `expm1` | No | Default; always available |
| vLLM | [`CustomOp` `XIELU`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) | Partial | CUDA wheel if installed, else Python fallback |
| SGLang | [`BaseFusedOp` `XIELU`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/activation.py) | Partial | Same optional CUDA path as HF/vLLM |
| nickjbrowning / rubber-duck-debug CUDA | `pip install git+https://github.com/nickjbrowning/XIELU` → `torch.classes.xielu.XIELU().forward` | Yes | CUDA only (cc 6.0+); `torch.compile` via `allow_in_graph`; optional `with_vector_loads` |
| llama.cpp / ggml | [`ggml_cuda_op_xielu`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/unary.cu) | Yes | Apertus GGUF inference; F16/F32 contiguous tensors |
| Liger Kernel | — | — | **No xIELU Triton kernel** |
| HF Hub kernels | — | — | **Not registered** in [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py) |
| Zepto (today) | [`src/zepto/modules/xielu.py`](../src/zepto/modules/xielu.py): `Where` + `Exp` + `Minimum` + `Multiply` … | No | Decomposed; `effective_alpha_p` / `effective_alpha_n` fold `softplus` at inference |
| Zepto (planned) | `region/xielu` | Yes | Atto-parity leaf; elide branch temps |

**Delivery & integration.** HF, vLLM, and SGLang share the same pattern: try `import xielu.ops` at init; on success bind `torch.classes.xielu.XIELU()` and route CUDA tensors through `_xielu_cuda`, otherwise emit a one-time warning and use `_xielu_python`. The CUDA path expects **3D** tensors \((B, T, H)\) and reshapes rank-2 inputs. llama.cpp embeds a standalone elementwise CUDA kernel with \(\alpha_p, \alpha_n, \beta, \varepsilon\) passed as op params (per-layer scalars from GGUF). There is **no** Liger monkey-patch or Hub dynamic loader for xIELU today.

Huang & Schlag §3.5: scalars have negligible memory; quadratic and exponential terms are computed **on the fly** (like GELU/SiLU); eager PyTorch is slower than fused GELU because of branching and multiple kernel launches; **a fused CUDA kernel** closes that gap by keeping the piecewise logic in registers. Memory footprint of activations/gradients is comparable to GELU — **one output of size \(\lvert H \rvert\)**, not a persistent extra buffer, when fused.

### Interoperability and composition

- **Layer scope:** xIELU runs **sequentially** inside the MLP (`Up GEMM → xIELU → Down GEMM`). It does **not** overlap or collide with RMSNorm, RoPE, FlashAttention, or other hub kernels.
- **Execution constraints:** contiguous tensors; bf16 / fp16 / fp32; CUDA wheel requires NVIDIA GPU with compute capability \(\ge 6.0\). XPU and MPS backends use the Python `where` path only (see §16.1–16.2).
- **Training vs inference:** inverse-softplus parameter storage; two scalar `softplus` ops per forward (negligible FLOPs). nickjbrowning CUDA provides a full backward via custom autograd; HF/vLLM Python path uses standard autograd through `where` / `expm1`.

### FLOPs

Appendix E **omits** xIELU (MLP leaf is GEMM-only). The fused-leaf constants below are **derived**, not copied from Appendix E.

**Accounting** matches §1 and [`CONTEXT.md`](../CONTEXT.md): a multiply-add is **two FLOPs**, so a lone mul or add is **1**. Special functions (`exp` / `expm1`) use the same coarse bucket as SiLU’s sigmoid: **≈ 4 FLOPs**, not one hardware instruction. The comparison \(x > 0\) is **not** billed as an arithmetic FLOP (same policy as ReLU). Formulas are **per layer** (this file’s convention); Atto writes the same leading terms with extra factors \(B\) and \(L\).

**Sources.** Op mix: Huang & Schlag ([2025, §3.5](https://arxiv.org/abs/2411.13010)). Billed fused leaf (Atto, sibling repo): `docs/cost-estimation-framework/50 - activation/55 - xielu.md`, implemented as `XIELU.forward_flops` / `backward_flops` in `atto/src/atto/ops/elementwise.py`, restated by the golden in `atto/tests/analyze/xielu/derivation.md`.

**Forward (fused leaf).** A fused kernel evaluates **one** branch per element.

Positive branch \(\alpha_p x^{2} + \beta x\):

| Step | FLOPs |
|------|-------|
| \(x^{2}\) | 1 mul |
| \(\alpha_p \cdot x^{2}\) | 1 mul |
| \(\beta \cdot x\) | 1 mul |
| add | 1 add |
| **Total** | **4** |

Negative branch \(\alpha_n(\mathrm{expm1}(\min(x,\varepsilon)) - x) + \beta x\):

| Step | FLOPs |
|------|-------|
| clamp + \(\mathrm{expm1}\) (SiLU-style special-function bucket) | 4 |
| \(\mathrm{expm1} - x\) | 1 |
| \(\times \alpha_n\) | 1 |
| \(\beta \cdot x\) | 1 |
| add | 1 |
| **Total** | **8** |

Huang & Schlag §3.5 list “one exponentiation, four multiplications, four additions, and one conditional” and call that **on par with SiLU**. Mapping the exp to the 4-FLOP special-function slot and the remaining arithmetic to 4 yields the same **8**. The fused leaf takes the **expensive branch as an upper bound** so mixed signs are not undercounted:

\[
\mathrm{FLOPs}_{\mathrm{xIELU,fwd}} = 8 \cdot S \cdot d_{\mathrm{ff}}.
\]

**Caveats on 8 (still the right leading constant):**

- Positive-only tensors are closer to **4**, not 8.
- Eager `torch.where` **computes both branches**, so the unfused Zepto decomposition (`Where` + `Exp` + `Minimum` + many `Multiply`) reports \(\approx 10\text{–}12 \cdot S d_{\mathrm{ff}}\) plus **full-size branch temps**. Fusion does not change the math; it stops paying for the unused side plus HBM temps.
- Warp divergence can execute both sides; 8 is then slightly optimistic. Order of magnitude is unchanged.
- Two scalar `softplus`s per layer are \(\approx 18\) FLOPs (\(c_{\mathrm{sp}} \approx 9\) each: \(\ln(1+e^{x})\)). They are \(O(1)\) per layer, not \(O(S d_{\mathrm{ff}})\), and are correctly dropped from the leading term. Inference graphs that pass `effective_alpha_p` / `effective_alpha_n` omit them entirely. Atto’s full billed formula is \(8 B S d_{\mathrm{ff}} L + 18 L\).

**Backward (fused leaf).** Local VJP, with pre-activation \(H\) **saved** (xIELU cannot recover the Jacobian from \(Z\) alone — unlike ReLU):

\[
\frac{\partial\mathcal{L}}{\partial H}
=\frac{\partial\mathcal{L}}{\partial Z}\odot
\begin{cases}
2\alpha_p H+\beta & H>0\\
\alpha_n(e^{H}-1)+\beta & H\le 0
\end{cases}
\]

| Branch | Work | FLOPs |
|--------|------|-------|
| \(H > 0\) | \(2\alpha_p H + \beta\), then \(\odot g\) | **4** (2 mul + 1 add + 1 mul by upstream) |
| \(H \le 0\) | recompute \(e^{H}\) (\(\approx 4\)) + \(\alpha_n(e^{H}-1)+\beta\) + \(\odot g\) | **8** |

The negative-branch **8 already includes** exp recomputation. Do **not** also add a full forward recompute.

Parameter grads are a masked reduction (MAC = 2 FLOPs / element: 1 mul + 1 add) into two scalars:

\[
\frac{\partial\mathcal{L}}{\partial\tilde\alpha_p}
=\sigma(\tilde\alpha_p)\sum_{H>0} g\,H^{2},\qquad
\frac{\partial\mathcal{L}}{\partial\tilde\alpha_n}
=\sigma(\tilde\alpha_n)\sum_{H\le 0} g\,(\mathrm{expm1}(H)-H).
\]

The \(\sigma(\tilde\alpha_\bullet)\) multiply happens **once per layer after the reduction** (\(\approx 14\) FLOPs for two sigmoids + scalar scales, with \(c_{\sigma} \approx 6\)). It does not add to the per-element count. Leading term:

\[
\underbrace{8}_{\text{Jacobian, neg-branch bound}}
+\underbrace{2}_{\text{parameter MAC}}
= 10 \cdot S \cdot d_{\mathrm{ff}}.
\]

\[
\mathrm{FLOPs}_{\mathrm{xIELU,bwd}} = 10 \cdot S \cdot d_{\mathrm{ff}}.
\]

Atto’s full billed formula is \(10 B S d_{\mathrm{ff}} L + 14 L\).

**Why not \(16 \cdot S \cdot d_{\mathrm{ff}}\).** That figure was a **heuristic**, not a derived count. Two common mistakes produce it: (1) GEMM’s **backward \(\approx 2\times\) forward** (\(2 \times 8 = 16\)), which models \(dX\) and \(dW\) for a matmul, not an elementwise activation; (2) adding **forward 8 + Jacobian 8**, which **double-counts** the exp already inside the 8-FLOP VJP. SiLU in the same framework is 5 forward / **8** backward, not \(2\times\) forward.

**MLP context.** At \(S{=}8192\), \(d_{\mathrm{ff}}{=}21504\): xIELU forward \(\approx\) **1.4B FLOPs/layer** vs Up+Down GEMMs \(\approx\) **57B FLOPs/layer** — xIELU is ~**2.5%** of MLP arithmetic. Replacing 16 with 10 on backward changes that small term, not the GEMM-dominated total. xIELU can still dominate MLP **HBM traffic** when unfused (see memory below).

**Arithmetic intensity.** Elementwise at ~8 FLOPs per element with ~4 bytes moved (bf16 read + write) gives \(\approx 2\,\mathrm{FLOPs/byte}\) — **memory-bound** (HBM bandwidth, not Tensor Cores). Fusion wins by eliminating HBM round-trips for branch masks and `expm1` intermediates, not by reducing the leading FLOP count.

**Zepto:** optional `region/xielu` should bill **8 / 10**, not 8 / 16. Inference estimates may fold `softplus` into the scalars (graph inputs `effective_alpha_p` / `effective_alpha_n`). Training estimates may add two scalar `softplus` ops (negligible FLOPs: \(+18\) forward, \(+14\) backward).

### Memory (forward-lived and saved)

Let \(\lvert H \rvert = S \cdot d_{\mathrm{ff}}\). At \(S{=}8192\), \(d_{\mathrm{ff}}{=}21504\): \(\lvert H \rvert \approx 1.76 \times 10^8\) → one bf16 buffer ≈ **336 MiB**.

| Quantity | Eager / unfused (HF Python, Zepto decomposed) | Fused CUDA / `region/xielu` | ggml CUDA (inference) |
|----------|-----------------------------------------------|-----------------------------|------------------------|
| Output \(y\) | \(\lvert H \rvert \cdot e\) | \(\lvert H \rvert \cdot e\) | \(\lvert H \rvert \cdot e\) |
| Full-size temps | branch masks, `expm1` path, `where` intermediates — **2–3×** \(\lvert H \rvert \cdot e\) peak | **0** (on-chip registers) | **0** (in-register) |
| Saved for backward | full input + output | output, or **1-byte sign mask** per element (same pattern as `region/relu`) | N/A (inference) |

Fused nickjbrowning CUDA: output only; optional \(\lvert H \rvert\)-byte mask if backward needs a sign bit. Unfused eager can account for **672–1008 MiB** of elidable peak temps per layer at Apertus-8B prefill shapes if branch buffers are counted as simultaneous — `region/xielu` must **replace** the primitive chain so those temps never enter the resource-event stream.

### Implementation summary

| Implementation | Package / access | Fused? | Primary target | Memory strategy |
|----------------|------------------|--------|----------------|-----------------|
| HF Python eager | `transformers.activations.XIELUActivation` | No | Portability / training | Full branch temps in HBM |
| nickjbrowning CUDA | `pip install …/XIELU` → `torch.classes.xielu.XIELU` | Yes | CUDA inference (Apertus) | Output only; optional sign mask for backward |
| vLLM / SGLang | `CustomOp` / `BaseFusedOp` + CUDA fallback | Partial | Serving | Same as HF/CUDA |
| llama.cpp ggml | `ggml_cuda_op_xielu` | Yes | GGUF inference | In-register, single dst |
| Liger Kernel | — | — | — | **Not available** |
| HF Hub kernels | — | — | — | **Not registered** |
| Zepto (current) | `src/zepto/modules/xielu.py` | No | Cost estimation | ~10–12 FLOP leaf + branch temps |
| Zepto (planned) | `region/xielu` | Yes | Atto-parity costing | 8 fwd / 10 bwd leaf; elide branch temps |

**Key takeaways for Zepto:** xIELU is optional for Appendix-E-comparable totals (small FLOP fraction) but matters for **VRAM accounting** when unfused. No ecosystem fused **training** kernel exists outside nickjbrowning CUDA autograd; Liger and Hub have nothing. `region/xielu` is a reasonable optional leaf — **8** FLOPs/element forward (neg-branch upper bound) and **10** backward (8-FLOP Jacobian + 2-FLOP parameter MAC), elide branch temps (~336 MiB/layer at Apertus-8B prefill), fold `softplus` into scalars at inference.

---

## 14. LM head

Untied linear \(d \to V\): FLOPs \(2 S d V\). Appendix E `final_logits`. Memory: weight \(V d e\) (another ~1 GiB bf16 at 8B) plus logits \(S V e\). At \(S{=}8192\), \(V{=}131072\), bf16 logits ≈ **2.0 GiB** — often comparable to a single attention matrix at the same \(S\).

### Liger `FusedLinearCrossEntropy` (vocab softmax fusion)

Training CE needs, per token, \(\mathrm{softmax}(h W^\top)_y\) and a scalar loss. Eager PyTorch does this as GEMM \(\to\) full logits \((S,V)\) \(\to\) softmax \(\to\) NLL. At Apertus-8B, \(S{=}8192\), \(V{=}131072\), bf16:

\[
S V e = 8192 \times 131072 \times 2 = 2^{31}\ \text{bytes} = \mathbf{2.0\ GiB}.
\]

Fusion cannot drop the **arithmetic** (\(2 S d V\) GEMM + softmax over \(V\)). It can drop the **resident** \((S,V)\) allocation by streaming the GEMM in tiles and running online softmax on each tile.

**Liger** ([`LigerFusedLinearCrossEntropyLoss`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/fused_linear_cross_entropy.py), impl [`fused_linear_cross_entropy.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)) tiles **rows of hidden** (tokens), not vocabulary:

```text
C = CHUNK_MEM_CONST = 16
inc_factor = cdiv(V, C * H)
chunk_size = next_power_of_2(cdiv(BT, inc_factor))   # BT = S
logits_chunk = hidden[chunk] @ W.T                   # (chunk_size, V)
```

Only **one** `(chunk_size, V)` logits tensor is live. Historical \(C{=}1\) sized that tile so \(\mathrm{chunk\_size}\times V \approx BT\times H\) (transient logits no larger than the hidden activation). Comments in the same file call \(C{=}1\) a **memory floor**, not a performance target: at Llama-3 shapes it forced many tiny chunks and became launch-bound. Current \(C{=}16\) raises the budget; measured Blackwell fwd+bwd at these shapes dropped ~123 ms → ~25 ms (~5×); gains flatten past \(C{=}16\).

Plug in Apertus-8B (\(BT{=}8192\), \(H{=}d{=}4096\), \(V{=}131072\), \(e{=}2\), \(C{=}16\)):

\[
C\cdot H = 65536,\quad \mathrm{cdiv}(V,\,C H) = 2,\quad \mathrm{chunk\_size} = 4096.
\]

\[
\text{live logits} = 4096 \times 131072 \times 2 = 2^{30}\ \text{B} = \mathbf{1.0\ GiB}.
\]

Closed form:

\[
\text{peak logits} \approx \min(S V e,\; C\cdot S\cdot d\cdot e).
\]

| Setting | Live logits | vs eager 2.0 GiB |
|---------|-------------|------------------|
| Eager full \((S,V)\) | **2.0 GiB** | 1× |
| Liger \(C{=}16\) (current) | **1.0 GiB** | ~2× smaller, **not 0** |
| Liger \(C{=}1\) (old floor) | **64 MiB** \(= S d e\) | ~32× smaller |

“Never writes the full \((S,V)\) tensor” is true. “Peak ≈ 0 GiB” is not.

The same forward also typically allocates (training, both hidden and \(W\) need grad): `grad_input` \((S,d)\) = 64 MiB, `grad_weight` \((V,d)\) = **1.0 GiB**, plus a few KiB of per-token loss. Backward **recomputes** each logits chunk rather than saving 2 GiB — that is the autograd win, not an empty forward working set.

| | Eager `Linear + CE` | Liger fused linear CE |
|--|---------------------|------------------------|
| **Softmax axis** | last dim \(V\) (vocab) | same |
| **Relation to attention masked softmax** | **Different op** — no causal mask over keys | Boundary **D**, not B |
| **Forward FLOPs** | \(2 S d V + 3 S V\) (GEMM + softmax leaf) | same leading terms |
| **Peak logits VRAM** | **\(S V e\)** | **\(\min(SVe,\, C S d e)\)** with \(C{=}16\) |
| **Saved for backward** | logits or CE intermediates \(\Theta(SV)\) | recompute chunks; save hidden/\(W\) as needed |
| **Access** | `use_liger_kernel=True` + `LigerFusedLinearCrossEntropyLoss` | Triton CUDA/ROCm |

**Arithmetic intensity** of the vocab softmax tile (bf16, 1-pass bound): \(\mathrm{AI} = 3/(2e) = 0.75\) FLOP/byte — same derivation as attention softmax. Do not use \(3/(4e)\).

Forward-only inference costing still needs explicit logits if sampling; fused CE is a **training** optimization. Zepto should model `region/linear_ce` separately from `region/masked_softmax`.

### 14.1 HF Hub: `trl-lib/fused-linear-ce`

Same training problem as Liger FLCE ([Hub card](https://huggingface.co/trl-lib/fused-linear-ce); Wijmans et al., 2024, *Cut Your Losses in Large-Vocabulary Language Models*). Differences:

| | Liger FLCE | TRL Hub kernel |
|--|------------|----------------|
| **Delivery** | `use_liger_kernel=True` monkey-patch | `get_kernel("trl-lib/fused-linear-ce")` |
| **Tile axis** | **tokens** (`chunk_size × V`) | **vocabulary** (online LSE over \(V\) tiles, fp32 accumulator) |
| **Peak logits** | \(\min(SVe,\, C S d e)\) | \(S \times V_{\mathrm{tile}} \times 4\) (fp32 accum; tile size kernel-defined) |
| **Metrics** | optional z-loss / accuracy flags | mean token accuracy + entropy in the same vocab sweep |
| **Backend** | CUDA / ROCm Triton | CUDA-only wheels today |

Both are `region/linear_ce` variants. Neither is attention masked softmax. Prefer Liger’s closed form when costing `use_liger_kernel=True`; use the vocab-tile formula when costing the Hub kernel.

---

## 15. KV cache (decode, vLLM / HF `use_cache`)

Persistent state, not a fused compute kernel:

\[
\mathrm{bytes}(\mathrm{KV}) = 2\, L\, h_{\mathrm{kv}}\, S_{\max}\, d_h\, e.
\]

Apertus-8B, \(S_{\max}{=}65536\), bf16: **8.0 GiB**. GQA vs MHA: multiply by \(h_{\mathrm{kv}}/h = 1/4\).

Prefill attention FLOPs \(\Theta(h S^2 d_h)\); decode step \(\Theta(h \cdot 1 \cdot S_{\mathrm{cache}} d_h)\) plus GEMMs on one token. FlashAttention extra memory stays linear in the *current* query length \(\times\) cache length tiles, not a new \((S,S)\) matrix.

The kernel that **reads** this cache in vLLM-style serving is Hub [`kernels-community/paged-attention`](https://huggingface.co/kernels-community/paged-attention) / FlashInfer (§11.9), not FA2 prefill.

**Zepto:** model as persistent tensors with `semantic_type="kv_cache"` on a decode **invocation**, not as temps inside prefill lowering (gap G3). Select `region/gqa/paged` for that invocation.

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

### 16.1 XPU inference variant (`USE_HUB_KERNELS=YES`, `device="xpu"`)

```text
Embedding            gather (ATen)
RoPE materialize     PyTorch fp32 freq → cos/sin
for each layer:
  RMSNorm            kernels-community/rmsnorm (SYCL ESIMD)  or  eager fp32 variance
  QKV GEMM           oneDNN / PyTorch XPU GEMM
  Q/K RMSNorm        kernels-community/rmsnorm
  RoPE apply         rotary hub (XPU build)  or  eager
  Attention          eager / SDPA (no FA2 on XPU by default)
  O GEMM             oneDNN / PyTorch
  RMSNorm            hub rmsnorm  or  eager
  Up / Down GEMM     oneDNN / PyTorch
  xIELU              Python where
Final RMSNorm        hub rmsnorm
LM head              GEMM
```

XPU hub RMSNorm is registered for **inference only**; training uses eager `LlamaRMSNorm` unless a custom `KernelConfig` overrides the mapping.

### 16.2 MPS inference variant (`USE_HUB_KERNELS=YES`, `device="mps"`)

```text
Embedding            gather (ATen)
RoPE materialize     PyTorch fp32 freq → cos/sin
for each layer:
  RMSNorm            kernels-community/mlx-rmsnorm (Metal)  or  eager fp32 variance
  QKV GEMM           PyTorch MPS GEMM
  Q/K RMSNorm        mlx-rmsnorm (Metal)
  RoPE apply         eager rotate_half  (no default MPS hub rotary in Apertus map)
  Attention          kernels-community/metal-flash-sdpa  or  eager / SDPA math
  O GEMM             PyTorch MPS
  RMSNorm            mlx-rmsnorm  or  eager
  Up / Down GEMM     PyTorch MPS
  xIELU              Python where
Final RMSNorm        mlx-rmsnorm
LM head              GEMM
```

MPS hub RMSNorm is registered for **inference only**; training uses eager `LlamaRMSNorm` unless a custom `KernelConfig` overrides the mapping. Forward path writes **output only** — row `inv_mean` stays in Metal threadgroup memory and is **not** cached as an HBM `rstd` tensor. Attention: without `metal-flash-sdpa`, SDPA on MPS may still materialize scores (§11.2, §11.8) — cost that path as eager GQA (G4b), not as FA2.

---

## 17. Recommended Zepto leaves (summary)

Use these closed forms when a fused **implementation** is selected. Unfused identity lowering may keep primitive sums for debugging, but must not be compared to vLLM/HF-flash totals.

| Kernel | FLOP leaf (forward) | Elide in fused VRAM | Save (training) |
|--------|---------------------|---------------------|-----------------|
| Linear / GEMM | \(2 S n_{\mathrm{in}} n_{\mathrm{out}}\) | none (already one kernel) | \(X\) |
| Fused QKV | sum of three GEMMs | two extra reads of \(X\) | \(X\) |
| RMSNorm (Liger / hub leaf) | \(4 \lvert x \rvert\) | `squared`, full `normalized` | `rstd` \((S,1)\) |
| RMSNorm hub XPU (PyTorch autograd) | same | same | `rstd` + \(x\) + \(y\) if backward enabled |
| RMSNorm hub MPS (Metal forward) | same | same | none at inference; Zepto leaf still models `rstd` for training parity |
| Fused add+RMSNorm | \(4 \lvert x \rvert + \lvert x \rvert\) | residual temp | `rstd` |
| Softmax | \(5 h S^2\) (stable max+sub+exp+sum+div; §10) | `exp` buffer; not a guaranteed 1-pass at \(S{=}64k\) | \(P\) |
| Masked softmax (fused region) | \(5 h S^2\) (scale+mask+exp+sum+div) | pre-softmax logits (`scores`, `exp`, …) | \(P\) |
| Masked softmax (unfused GQA) | \(5 h S^2\) score+softmax (\(+\;2 h S^2 d_h\) for \(QK^\top\); no max) | 2–3× \((h,S,S)\) temps | \(P\) |
| PyTorch `F.softmax` / AITER softmax | \(5 h S^2\) | fused kernel elides decomposed temps | \(P\) if autograd |
| Liger fused linear CE | \(2 S d V + 3 S V\) | full \((S,V)\) **not** zero; peak \(\min(SVe,\, C S d e)\) with \(C{=}16\) | chunk recompute (training) |
| TRL Hub fused linear CE | same leading FLOPs | vocab-tile fp32 accum, not full \((S,V)\) | recompute tiles |
| Flash GQA (FA2 / FA3 / FA4 / Metal-Flash / Sage) | \(4 h S^2 d_h\) (+ softmax term; §11) | all \((h,S,S)\); see §11.6–§11.11 | row stats \(\Theta(hS)\) |
| Paged / FlashInfer decode | \(4 h S_{\mathrm{cache}} d_h\) | no \(S\times S\); paged KV is §15 | — (inference) |
| RoPE apply | \(\approx 6 h_\star S d_h\) | rotate-half temps | — |
| xIELU | \(8 S d_{\mathrm{ff}}\) (bwd \(10 S d_{\mathrm{ff}}\)) | branch/exp temps | output or sign mask |
| Repeat-KV | 0 | full repeated K/V if Flash | — |
| Causal mask (eager) | 0 | — | persistent \(S^2\) |
| Causal (Flash) | 0 | **no** \(S^2\) tensor | — |

### Still to implement in Zepto (softmax-family + related)

Registered today: `region/linear`, `region/layernorm`, `region/relu`, `region/rmsnorm`, `region/xielu`, `region/softmax`. The following fused leaves are specified above but **not** yet registered as `RegionImplementation`s:

| Priority | Region id | Spec | Boundary |
|----------|-----------|------|----------|
| 1 | `region/masked_softmax` | §10 (Megatron/TE) | B |
| 2 | `region/gqa/flash2` | §11.3–§11.7 | C |
| 3 | `region/gqa/metal-flash` | §11.8 | C (MPS) |
| 4 | `region/gqa/paged` | §11.9 + §15 | C (decode) |
| 5 | `region/linear_ce` | §14 (Liger) + §14.1 (TRL Hub) | D |
| 6 | `region/gqa/flash3`, `flash4`, `vllm-flash-attn3` | §11.5, §11.10 | C |
| optional | `region/gqa/sage` | §11.11 | C (quantized) |
| optional | AITER `softmax` routing | §10.1 | A on ROCm only |

---

## 18. Bibliography

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. (2023). GQA: Training generalized multi-query transformer models from multi-head checkpoints. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).

Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).

Chowdhery, A., et al. (2022). PaLM: Scaling language modeling with Pathways. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311).

Dai, Y., Kothapalli, V., Song, Q., Tang, S., Zhu, S., Shimizu, S., Sahni, S., Ning, H., and Chen, Y. (2024). Liger Kernel: Efficient Triton kernels for LLM training. [arXiv:2410.10989](https://arxiv.org/abs/2410.10989). Source: [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel). Hub mapping: [transformers `hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py).

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. NeurIPS. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. (2024). FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608). Implementation: [`Dao-AILab/flash-attention` `hopper/`](https://github.com/Dao-AILab/flash-attention); Hub: [`kernels-community/flash-attn3`](https://huggingface.co/kernels-community/flash-attn3), [`flash-attn4`](https://huggingface.co/kernels-community/flash-attn4), [`vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3).

Dehghani, M., et al. (2023). Scaling vision transformers to 22 billion parameters. [arXiv:2302.05449](https://arxiv.org/abs/2302.05449).

Grattafiori, A., et al. (2024). The Llama 3 herd of models. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).

Henry, A., Dachapally, P. R., Pawar, S., and Chen, Y. (2020). Query-key normalization for transformers. [arXiv:2010.04245](https://arxiv.org/abs/2010.04245).

Hernández-Cano, A., Hägele, A., Huang, A. H., et al. (2025). Apertus: Democratizing open and compliant LLMs for global language environments. [arXiv:2509.14233](https://arxiv.org/abs/2509.14233). Appendix E: FLOP script. Table 5: long-context RoPE \(\Theta\) schedule.

Huang, A. H., and Schlag, I. (2025). Deriving activation functions using integration. [arXiv:2411.13010](https://arxiv.org/abs/2411.13010). CUDA: [nickjbrowning/XIELU](https://github.com/nickjbrowning/XIELU) (also [rubber-duck-debug/xielu](https://github.com/rubber-duck-debug/xielu)). ggml: [llama.cpp `unary.cu`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/unary.cu).

Peng, B., Quesnelle, J., Fan, H., and Shippole, E. (2023). YaRN: Efficient context window extension of large language models. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071). (NTK-aware interpolation lineage used by Llama-3-style RoPE in Transformers.)

Rabe, M. N., and Staats, C. (2021). Self-attention does not need \(O(n^2)\) memory. [arXiv:2112.05682](https://arxiv.org/abs/2112.05682).

Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150).

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

Vaswani, A., et al. (2017). Attention is all you need. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

Xiong, R., et al. (2020). On layer normalization in the transformer architecture. ICML. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745).

Zhang, B., and Sennrich, R. (2019). Root mean square layer normalization. NeurIPS. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).

FlashInfer. (2024). Accelerating self-attentions for LLM serving with FlashInfer. [flashinfer.ai](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html). Related paper: [arXiv:2501.01005](https://arxiv.org/abs/2501.01005).

Wijmans, E., Huval, B., Hertzberg, A., Koltun, V., and Krähenbühl, P. (2025). Cut your losses in large-vocabulary language models. ICLR. [arXiv:2411.09009](https://arxiv.org/abs/2411.09009). Hub (clean-room Triton): [`trl-lib/fused-linear-ce`](https://huggingface.co/trl-lib/fused-linear-ce). Apple CCE: [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy).

**Library sources (implementation, not papers):**

- HuggingFace Apertus modular / generated: [`transformers/.../apertus/`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/apertus)
- HuggingFace `hub_kernels.py` RMSNorm device map: [`integrations/hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)
- HF Hub XPU RMSNorm: [`kernels-community/rmsnorm`](https://huggingface.co/kernels-community/rmsnorm)
- HF Hub MPS RMSNorm (MLX-lineage Metal): [`kernels-community/mlx-rmsnorm`](https://huggingface.co/kernels-community/mlx-rmsnorm) — source [`kernels-community/mlx-rmsnorm`](https://github.com/huggingface/kernels-community/tree/main/mlx-rmsnorm); MLX reference [`mlx/backend/metal/kernels/rms_norm.metal`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal)
- HF Hub MPS Flash SDPA: [`kernels-community/metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa) — [source README](https://github.com/huggingface/kernels-community/blob/main/metal-flash-sdpa/README.md)
- HF Hub paged attention: [`kernels-community/paged-attention`](https://huggingface.co/kernels-community/paged-attention)
- HF Hub AITER (ROCm softmax + ops): [`kernels-community/aiter-kernels`](https://huggingface.co/kernels-community/aiter-kernels) — upstream [ROCm/aiter](https://github.com/ROCm/aiter); Flash split to `aiter-flash-attn`
- HF Hub SageAttention: [`kernels-community/sage-attention`](https://huggingface.co/kernels-community/sage-attention)
- Intel XPU ESIMD norm (related): [`intel/llm-scaler` omni_xpu_kernel](https://github.com/intel/llm-scaler)
- HuggingFace `XIELUActivation`: [`activations.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py)
- Llama-3 RoPE init: [`modeling_rope_utils.py` `_compute_llama3_parameters`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)
- vLLM Apertus: [`vllm/model_executor/models/apertus.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/apertus.py)
- Megatron fused softmax: [`megatron/core/fusions/fused_softmax.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_softmax.py)
- Liger fused linear CE (`CHUNK_MEM_CONST = 16`): [`liger_kernel/ops/fused_linear_cross_entropy.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)
- PyTorch CUDA softmax: [`aten/src/ATen/native/cuda/SoftMax.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu)
- Triton fused-softmax HBM accounting: [tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- Checkpoints: [Apertus-8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509), [Apertus-70B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509)
- PyTorch SDPA: [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

*Kernel formulas cross-checked 2026-08-26 against Transformers `main`, vLLM `main`, swiss-ai Instruct-2509 configs, and the papers above. RMSNorm hub device map, `kernels-community/rmsnorm` autograd surface, and `kernels-community/mlx-rmsnorm` Metal implementation verified 2026-09-01. xIELU fused-leaf FLOPs (8 forward / 10 backward) derived 2026-09-01 from Huang & Schlag §3.5 plus Atto `55 - xielu.md`; the previous \(16 S d_{\mathrm{ff}}\) backward was a \(2\times\)-forward heuristic. Softmax fused leaves (\(5 h S^2\)), Liger linear-CE chunk peak (\(C{=}16\)), and Hub attention/softmax variants (Metal-Flash, paged-attention, AITER softmax, FA4, Sage, TRL fused-linear-ce) recorded 2026-09-04.*
