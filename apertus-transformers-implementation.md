# Apertus in HuggingFace Transformers — Implementation Reference

Investigation of the official Apertus model in [huggingface/transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/apertus), cross-checked against released checkpoints, vLLM, and the Apertus paper. Purpose: map the reference implementation (including kernel paths) to Zepto module/lowering gaps.

Kernel mathematics, FLOP leaves, and HBM/VRAM structure for these ops live in [`docs/kernel-implementation.md`](docs/kernel-implementation.md). This file stays the Apertus-specific wiring and Zepto-gap tracker.

**Sources**

| Source | URL |
|--------|-----|
| Modular source (edit here) | https://github.com/huggingface/transformers/blob/main/src/transformers/models/apertus/modular_apertus.py |
| Generated modeling | https://github.com/huggingface/transformers/blob/main/src/transformers/models/apertus/modeling_apertus.py |
| Generated config | https://github.com/huggingface/transformers/blob/main/src/transformers/models/apertus/configuration_apertus.py |
| 8B checkpoint config | https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509/blob/main/config.json |
| 70B checkpoint config | https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509/blob/main/config.json |
| Paper | https://arxiv.org/abs/2509.14233 |
| xIELU paper | https://arxiv.org/abs/2411.13010 |
| vLLM inference port | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/apertus.py |
| xIELU CUDA (optional) | https://github.com/nickjbrowning/XIELU |
| Kernel cost reference | [`docs/kernel-implementation.md`](docs/kernel-implementation.md) |

---

## 1. High-level architecture

Apertus is a **decoder-only** transformer from the Swiss AI Initiative. Relative to vanilla Llama, the distinctive choices are:

| Feature | Apertus | Llama (typical) |
|---------|---------|-----------------|
| Normalization | Pre-**RMSNorm** (γ only, no β) | Pre-RMSNorm |
| Attention | **GQA** + **QK-Norm** on head dim | GQA/MHA, no QK-Norm |
| Positional encoding | **RoPE** with **Llama 3.1 scaling** | RoPE (variant-dependent) |
| FFN | **2-linear** `up → xIELU → down` (Nemotron-style) | SwiGLU (gate + up) |
| Activation | **xIELU** | SiLU/GELU in SwiGLU |
| Biases | **None** on linear layers | Often none |
| LM head | **Untied** from embeddings | Often tied |
| Dropout | `attention_dropout=0` at inference | Configurable |

The paper ([arxiv:2509.14233](https://arxiv.org/abs/2509.14233)) lists GQA, RoPE, RMSNorm, QK-Norm, and xIELU as the architectural deltas over a Llama baseline. Training-only concerns (AdEMAMix optimizer, Goldfish loss, cross-document attention) are out of scope for forward-graph / kernel parity.

**RoPE base frequency:** §2.1 pretrains at \(\Theta = 5\times 10^5\) with context 4096. Long-context extension (§2.5, Table 5) **raises \(\Theta\)** at each stage and finishes at **\(\Theta = 12\times 10^6\)** for 64k. Released Instruct-2509 configs therefore use `rope_theta: 12000000` plus Llama-3 piecewise scaling — that is the inference/kernel truth, not the pretraining paragraph.

---

## 2. Repository layout (Transformers)

Only four Python modules under `src/transformers/models/apertus/`:

| File | Role |
|------|------|
| `modular_apertus.py` | **Source of truth** — CI regenerates the others from this |
| `modeling_apertus.py` | Generated full PyTorch implementation |
| `configuration_apertus.py` | Generated `ApertusConfig` |
| `__init__.py` | Public exports |

**Modular inheritance** (what Apertus actually reuses):

```text
ApertusConfig          — standalone PreTrainedConfig
ApertusMLP             — NemotronMLP (+ xIELU dtype wiring)
ApertusRMSNorm         — LlamaRMSNorm
ApertusRotaryEmbedding — LlamaRotaryEmbedding
ApertusAttention       — LlamaAttention + q_norm / k_norm
ApertusDecoderLayer    — LlamaDecoderLayer (renamed norms, no input/post layernorm attrs)
ApertusModel           — LlamaModel
ApertusForCausalLM     — LlamaForCausalLM
```

There is **no separate CUDA/Triton file** in the Apertus folder; kernels come from shared Transformers integrations (hub kernels, flash-attn, SDPA, optional xIELU wheel).

**Modular vs generated:** `modular_apertus.py` is the edit surface (inheritance from Llama/Nemotron). CI inlines those bases into `modeling_apertus.py`: `ApertusRMSNorm` is a full copy of `LlamaRMSNorm` **including** `@use_kernel_forward_from_hub("RMSNorm")`; `ApertusMLP` is a standalone 2-linear module (not a live `NemotronMLP` subclass at runtime); `ApertusPreTrainedModel` is generated as `PreTrainedModel` with Llama’s attention-backend flags copied onto it (`_supports_flash_attn/sdpa/flex_attn/attention_backend = True`). Modular `ApertusPreTrainedModel` is `pass` on `LlamaPreTrainedModel` — same flags via inheritance.

---

## 3. Hyperparameters (checkpoint truth)

Values below are from **released weights**, not the stale defaults in `ApertusConfig` (`intermediate_size=14336` in the class does **not** match checkpoints).

### Apertus-8B-Instruct-2509

| Field | Value |
|-------|-------|
| `hidden_size` | 4096 |
| `intermediate_size` | **21504** |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 (= 4096 / 32) |
| `vocab_size` | 131072 |
| `max_position_embeddings` | 65536 |
| `hidden_act` | `xielu` |
| `rms_norm_eps` | 1e-5 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `tie_word_embeddings` | false |
| `qk_norm` | true |
| `post_norm` | false |
| `rope_theta` | 12000000 |
| `rope_scaling` / `rope_parameters` | `rope_type: llama3`, `factor: 8.0`, `original_max_position_embeddings: 8192`, `low_freq_factor: 1.0`, `high_freq_factor: 4.0` |

`qk_norm` and `post_norm` are **checkpoint extras**, not fields of current `ApertusConfig`. QK-Norm is **hardwired** in `ApertusAttention` (`q_norm` / `k_norm` always constructed). `post_norm: false` matches the pre-norm block; there is no post-norm module in the generated layer.

### Apertus-70B-Instruct-2509

| Field | Value |
|-------|-------|
| `hidden_size` | 8192 |
| `intermediate_size` | **43008** |
| `num_hidden_layers` | 80 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| Same pattern as 8B | `vocab_size` 131072, `hidden_act` xielu, `qk_norm` true, `post_norm` false, Llama-3 RoPE (`theta` \(12\times 10^6\), `factor` 8, `original_max` 8192) |

Zepto presets in `src/zepto/modules/apertus.py` (`APERTUS_8B`, `APERTUS_70B`) **match checkpoint** intermediate sizes (21504 / 43008).

---

## 4. Forward graph (eager Python path)

### 4.1 Model stack

```text
input_ids
  → embed_tokens                    nn.Embedding(V, d)
  → rotary_emb(x, position_ids)     cos, sin caches
  → create_causal_mask(...)         additive mask (supports padding / cache)
  → for layer in layers:
        residual = x
        x = attention_layernorm(x)  ApertusRMSNorm(d)
        x = self_attn(x, cos, sin, mask)
        x = residual + x
        residual = x
        x = feedforward_layernorm(x)
        x = mlp(x)                  up → xIELU → down
        x = residual + x
  → norm(x)                         ApertusRMSNorm(d)
  → lm_head(x)                      Linear(d → V), untied
```

### 4.2 Attention (`ApertusAttention`)

Order of operations (matches Zepto `GroupedQueryAttention`):

1. `q_proj`, `k_proj`, `v_proj` — bias-free `nn.Linear`
2. Reshape to `(batch, heads, seq, head_dim)` and transpose
3. **`q_norm(query_states)`**, **`k_norm(key_states)`** — RMSNorm on **last dim = head_dim**
4. **`apply_rotary_pos_emb(q, k, cos, sin)`**
5. Optional `past_key_values.update(...)` for incremental decoding
6. Attention backend (see §5)
7. Reshape + `o_proj`

Scaling: `1 / sqrt(head_dim)`.

### 4.3 FFN (`ApertusMLP`)

```python
return self.down_proj(self.act_fn(self.up_proj(x)))
```

- **No `gate_proj`** — generated `ApertusMLP` is a 2-linear module (`up → act → down`), matching Nemotron-style MLP rather than Llama SwiGLU. Modular source subclasses `NemotronMLP` then redefines `up_proj` / `down_proj`.
- `hidden_act == "xielu"` → `ACT2CLS["xielu"](dtype=config.dtype)` (generated file also assigns `ACT2FN` first, then overwrites for xIELU).
- Paper: xIELU is **not** a gated unit; MLP width is scaled \(1.5\times\) vs SwiGLU to match compute ([arxiv:2509.14233](https://arxiv.org/abs/2509.14233) §2.4).

### 4.4 xIELU (`XIELUActivation` in `activations.py`)

Paper definition ([Huang and Schlag, 2025](https://arxiv.org/abs/2411.13010); Apertus §2.1), \(\beta = 0.5\):

\[
\mathrm{xIELU}(x) =
\begin{cases}
\alpha_p x^{2} + \beta x & x > 0, \\
\alpha_n (e^{x}-1) - \alpha_n x + \beta x & x \le 0.
\end{cases}
\]

Eager Python (always available) is algebraically the same with a stability clamp \(\varepsilon = -10^{-6}\):

```python
alpha_p = softplus(alpha_p_param)
alpha_n = beta + softplus(alpha_n_param)
out = where(x > 0,
            alpha_p * x * x + beta * x,
            (expm1(min(x, eps)) - x) * alpha_n + beta * x)
```

Defaults: `alpha_p_init=0.8`, `alpha_n_init=0.8`, `beta=0.5`, `eps=-1e-6`.

Parameters stored in **inverse-softplus space** (same as vLLM): \(\alpha_p\) as \(\log(\mathrm{expm1}(0.8))\); \(\alpha_n\) param as \(\log(\mathrm{expm1}(\alpha_n^{\mathrm{init}}-\beta))\) because the forward adds \(\beta\) after `softplus`.

**Optional CUDA kernel** (not part of Transformers repo):

- Package: `pip install git+https://github.com/nickjbrowning/XIELU`
- Entry: `torch.classes.xielu.XIELU().forward(...)`
- Used when tensor is CUDA and package is installed; otherwise Python path.
- **Not** registered in Transformers hub-kernel mapping (`USE_HUB_KERNELS`).

### 4.5 RoPE (Llama 3.1 scaling)

Init via `ROPE_INIT_FUNCTIONS["llama3"]` → `_compute_llama3_parameters` in `modeling_rope_utils.py`:

1. Base inverse frequencies: `inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))` with checkpoint **`theta = 12e6`** (not the paper’s pretraining \(5\times 10^5\); see §1)
2. Wavelength-dependent scaling with `factor=8`, `low_freq_factor=1`, `high_freq_factor=4`, `original_max_position_embeddings=8192` (low-frequency bands `/ factor`, mid-band interpolate, high-frequency unchanged)
3. Forward: `freqs = inv_freq @ position_ids`, `emb = cat(freqs, freqs)`, `cos/sin * attention_scaling` (**1.0** for llama3; unused unlike YaRN)

Runtime cache: single shared `ApertusRotaryEmbedding` at model level; cos/sin passed into every layer. Wrong `inv_freq` does not change shapes or FLOPs — only angles (gap G1).

### 4.6 RMSNorm

Standard Llama/T5-style:

```python
x_fp32 = x.to(float32)
x = x * rsqrt(mean(x^2) + eps)
return weight * x.to(input_dtype)
```

Applied at: pre-attention, pre-FFN (per block), Q/K heads, final norm.

---

## 5. Kernel and backend map

### 5.1 Hub kernels (`integrations/hub_kernels.py`, env `USE_HUB_KERNELS=YES`)

Decorators on **shared** Llama components used by Apertus (copied onto generated `ApertusRMSNorm`):

| Logical op | Decorator / hook | Hub repo (CUDA default) | Layer name |
|------------|------------------|-------------------------|------------|
| RMSNorm | `@use_kernel_forward_from_hub("RMSNorm")` on `LlamaRMSNorm` / generated `ApertusRMSNorm` | `kernels-community/liger-kernels` (v3) | `LigerRMSNorm` |
| RoPE apply | `@use_kernel_forward_from_hub("rotary_pos_emb")` on `apply_rotary_pos_emb` | `kernels-community/rotary` (v2) | `apply_rotary_transformers` |
| RoPE apply (class) | `@use_kernelized_func(apply_rotary_pos_emb)` on `LlamaAttention` / `ApertusAttention` | (wraps same rotary kernel) | — |

Device-specific fallbacks exist for RMSNorm on xpu (`kernels-community/rmsnorm`), mps (`mlx_rmsnorm`), npu/rocm (Liger). Rotary ROCm inference uses `kernels-community/aiter-rope`.

**Not hub-accelerated for Apertus:**

- xIELU (optional external wheel only)
- Linear / matmul (generic PyTorch; `Linear` hub entry exists but is not wired on Apertus linears)
- Softmax, repeat_kv, residual adds

### 5.2 Attention backends (`ALL_ATTENTION_FUNCTIONS`)

`ApertusPreTrainedModel` (generated; modular inherits the same flags from `LlamaPreTrainedModel`) declares:

```python
_supports_flash_attn = True
_supports_sdpa = True
_supports_flex_attn = True
_supports_attention_backend = True
```

Runtime selection via `config._attn_implementation`:

| Backend | Implementation | Notes |
|---------|----------------|-------|
| `eager` | `eager_attention_forward` in modeling file | `QK^T * scale + mask → softmax(fp32) → dropout → PV` |
| `sdpa` | `sdpa_attention_forward` | `F.scaled_dot_product_attention` — **dispatcher**, not one kernel: may pick math (\(\Theta(S^2)\) memory), mem-efficient, or flash |
| `flash_attention_2/3/4` | Dao-AILab flash-attention integrations | Fused tiled attention; **no** HBM \((h,S,S)\); GQA without materializing `repeat_kv` |
| `flex_attention` | PyTorch flex attention | When available |

Eager path details:

- `repeat_kv` expands KV heads to match Q heads (GQA)
- Softmax explicitly in **float32**: `softmax(..., dtype=torch.float32).to(query.dtype)`
- Causal masking via pre-built `attention_mask` from `create_causal_mask`

### 5.3 vLLM inference kernels ([`vllm/.../apertus.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/apertus.py))

Production inference stack (separate from Transformers eager):

| Component | vLLM implementation |
|-----------|---------------------|
| Projections | Fused **`QKVParallelLinear`** (q/k/v merged; FLOPs = sum of three GEMMs, one read of \(X\)) |
| Q/K norm | **`RMSNorm`** (vLLM CUDA/IR kernel) |
| RoPE | **`get_rope(..., rope_parameters=config.rope_parameters)`** — llama3-aware |
| Attention | **`Attention`** / FlashAttention-style backend (GQA, no eager `repeat_kv`) |
| MLP | **`ColumnParallelLinear` + `XIELU` + `RowParallelLinear`** |
| Norms | Fused **pre-attention / pre-FFN RMSNorm** with residual (`hidden_states, residual = norm(x, residual)` → `fused_add_rms_norm`) |
| xIELU | Same optional **`nickjbrowning/XIELU`** CUDA class; Python fallback |

---

## 6. Structural comparison: Transformers ↔ Zepto

Current Zepto modules under `src/zepto/modules/` implement the same **logical** graph at eager decomposition level.

| Component | Transformers | Zepto | Parity |
|-----------|--------------|-------|--------|
| Stack | `ApertusModel` | `Apertus` | ✅ |
| Block | `ApertusDecoderLayer` | `ApertusDecoderBlock` | ✅ pre-RMSNorm residuals |
| Norm naming | `attention_layernorm` / `feedforward_layernorm` | `pre_attn_norm` / `pre_ffn_norm` | ✅ equivalent |
| GQA | `ApertusAttention` | `GroupedQueryAttention` | ✅ |
| QK-Norm | RMSNorm on `(h, S, d_h)` after proj | `QKNormRMSNorm` | ✅ before RoPE |
| RoPE cache | `ApertusRotaryEmbedding` | `RoPEMaterialize` | ⚠️ see gap G1 |
| RoPE apply | `apply_rotary_pos_emb` | `RoPEApply` | ✅ |
| FFN | `ApertusMLP` | `FFN` + `XIELU` | ✅ no gate |
| xIELU | `XIELUActivation` | `XIELU` (Where/Exp/Pow/…) | ✅ eager formula |
| Mask | `create_causal_mask` | `MaterializedCausalMask` | ⚠️ see gap G2 |
| Embedding | `(V, d)` | `(d, V)` + `EmbeddingLookup` | ⚠️ layout convention |
| LM head | untied `Linear(d, V)` | `LMHead` | ✅ |
| Batch | `(B, S, d)` | `(S, d)` rank-2 | ⚠️ intentional for cost API |
| KV cache | `DynamicCache`, `past_key_values` | not modeled | ❌ gap G3 |
| Decode / generate | `GenerationMixin` | not modeled | ❌ |

---

## 7. Zepto gaps (actionable)

Prioritized for closing parity with Transformers **and** realistic kernel-aware cost estimation.

### How Zepto should account for costs

Zepto's costing pipeline is: **Module composition** → structural graph (primitives with provenance) → **lowering** → per-node FLOPs + `ResourceEvent`s (ALLOCATE / SAVE / RELEASE).

| Mode | FLOP authority | VRAM authority |
|------|----------------|----------------|
| **Unfused (default today)** | Sum of per-op `forward_flops` on each primitive | Each op's `resource_events`; temporaries often **over-count** peak memory |
| **Fused region** | Single closed-form leaf on `RegionImplementation` | Explicit aux saves only; **elide** internal temps that real kernels fuse away |

Fusion policy (from `module-building-plan.md`): register a `RegionImplementation` when Atto / production kernels treat an op as a **cost leaf** or when eager decomposition **mis-estimates peak VRAM**. A fused region must **replace** its primitive chain in lowering — never double-count.

**Notation for examples below** (Apertus-8B preset): \(S\) = seq len, \(d{=}4096\), \(d_{\text{ff}}{=}21504\), \(h{=}32\), \(h_{\text{kv}}{=}8\), \(d_h{=}128\), \(L{=}32\). Element sizes use bf16 (2 bytes) unless noted. **GiB = \(2^{30}\) bytes.** Closed-form kernel leaves and citations: [`docs/kernel-implementation.md`](docs/kernel-implementation.md).

Apertus Appendix E counts GEMM + softmax + RMSNorm/QK-Norm and **omits** RoPE, residual adds, embedding, and xIELU. Fused-region FLOPs for those counted ops should match Appendix E (`4\cdot\mathrm{numel}\) RMSNorm, \(3 h S^2\) softmax, \(4 S d d_{\mathrm{ff}}\) ungated MLP). Elementwise kernels still need their own leaves when matching a real backend.

---

### G1 — Llama3 RoPE frequency init (High)

**Reference:** `_compute_llama3_parameters` in `modeling_rope_utils.py` piecewise scales `inv_freq` from base RoPE frequencies using `factor=8`, `low_freq_factor=1`, `high_freq_factor=4`, `original_max_position_embeddings=8192`, `rope_theta=12e6`.

**Zepto today:** `RoPEMaterialize` (`src/zepto/modules/rope_materialize.py`) takes a persistent `inv_freq` graph input but **`Apertus.apertus_8b()` never fills it** with llama3-scaled values — callers must supply raw frequencies out-of-band.

#### Framework change

| Item | Change |
|------|--------|
| `RoPEConfig` | Add `rope_type`, `rope_theta`, `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings` (mirror checkpoint `rope_scaling`) |
| New helper | `compute_inv_freq_llama3(head_dim, config) -> Tensor` — port HF `_compute_llama3_parameters` logic (init-only, not a graph op) |
| `Apertus` factory | Bind computed `inv_freq` into the persistent input buffer at `build_graph()` time |
| Billing | Init math is **outside** the structural graph; forward `RoPEMaterialize` FLOPs unchanged |

#### Cost impact

| | FLOPs | VRAM |
|---|-------|------|
| **Wrong init (bug)** | Same as correct path | Same tensor shapes: `cos/sin` each $(S, d_h)$ persistent per forward |
| **Correct init** | Unchanged: $\approx 2 S d_h$ (trig) + $S \cdot d_h/2$ (freq matmul) per model forward | Unchanged |
| **What breaks without fix** | Nothing in FLOP totals | Golden parity vs HF/Atto at $S > 8192$ — rotary angles differ, so any correctness-based benchmark fails even if costs match |

At $S{=}65536$, $d_h{=}128$: RoPE cache is $2 \times 65536 \times 128 \times 2$ B ≈ **32 MiB** forward-lived `cos`+`sin` (same either way); only the **values** differ.

---

### G2 — Causal mask semantics (Medium)

**Reference:** HF `create_causal_mask` returns a batch-aware, cache-length-aware additive mask (often 4D) re-built when `past_key_values` grows.

**Zepto today:** `MaterializedCausalMask` → static `(1, S, S)` buffer, shared across layers (`src/zepto/modules/materialized_causal_mask.py`).

#### Framework change

| Scenario | Zepto change |
|----------|--------------|
| Uniform prefill (current) | Document as supported; mask billed once as **persistent** model-level buffer |
| Padding / variable length | New mask module or op params: `seq_len`, `attention_mask` input → still $(B,1,S,S)$ or broadcast add |
| Incremental decode | See G3 — mask slice becomes $(1,1,1,S_{\text{total}})$ per step; different graph builder mode |

#### Cost impact (prefill, current Zepto model)

| Tensor | Shape | VRAM (bf16) | FLOPs |
|--------|-------|-------------|-------|
| Mask $M$ | $(1,S,S)$ | $S^2 \times 2$ B | 0 (materialized once) |
| Mask add in GQA | broadcast to $(h,S,S)$ | No extra alloc if fused add | $h S^2$ adds per layer |

Example \(S{=}8192\): mask storage ≈ **128 MiB** (shared, not × \(L\)) if bf16. At \(S{=}65536\): **8.0 GiB** bf16 or **16.0 GiB** if the additive mask is fp32 (`-inf`). FlashAttention causal mode does **not** allocate this tensor.

**Gap vs HF:** Zepto cannot yet model padding-masked rows (softmax over padded keys) or cache-extended length without rebuilding the graph with a new \(S\).

---

### G3 — KV cache / incremental decode (Defer)

**Reference:** After RoPE, HF `past_key_values.update(key_states, value_states, layer_idx)` appends to cached K/V.

**Zepto today:** Full-sequence static graph; K/V activations live only for the current forward slice.

#### Framework change (when needed)

- New graph inputs: `past_key_values` per layer or shared cache tensor with `cache_len`
- K/V tensors tagged `semantic_type="kv_cache"` with **persistent** SAVE across decode steps
- Separate `PhaseIntent` / builder flag: `prefill` vs `decode` (Atto pattern)
- RoPE `position_ids` offset by `past_seen_tokens`

#### Cost impact (decode step, single new token)

| | Prefill (today) | Decode (target) |
|---|-----------------|-----------------|
| **Attention FLOPs per layer** | $O(h S^2 d_h)$ matmuls | $O(h \cdot S \cdot d_h)$ — one query row vs full cache |
| **KV VRAM** | Ephemeral within one forward | Persistent: $2 L h_{\text{kv}} S_{\max} d_h$ elements |
| **Peak attention activations** | $(h,S,S)$ scores (eager) | $(h,1,S)$ scores — **linear in cache len**, not quadratic in full $S$ |

Apertus-8B KV cache at \(S_{\max}{=}65536\): \(2 \times 32 \times 8 \times 65536 \times 128 \times 2\) B = **8.0 GiB** (bf16, K+V across layers) — billed as persistent, not per-forward temp. GQA vs MHA saves a factor \(h / h_{\mathrm{kv}} = 4\).

---

### G4 — Fused region lowering (High for cost accuracy)

**Registered today:** `region/linear`, `region/layernorm`, `region/relu`.  
**Missing for Apertus:** `region/rmsnorm`, `region/softmax`, `region/masked_softmax`, optional `region/xielu`, optional `region/gqa`.

Each missing region causes Zepto to **sum primitive FLOPs** and **allocate every intermediate** in the decomposed chain — diverging from Liger / FlashAttention / Atto leaf recipes.

#### G4a — `region/rmsnorm`

**Module decomposition today** (`rms_norm.py`): `square → reduce_sum → divide → add → sqrt → divide → [×γ]` — **7 ops**, 2 full-size temps (`squared`, `normalized`).

| | Unfused (today) | Fused `region/rmsnorm` (target) |
|---|-----------------|----------------------------------|
| **Forward FLOPs** | Sum of 7 op estimates ≈ **$5\text{–}6 \cdot \text{numel}$** | **$4 \cdot \text{numel}$** (γ-only, Atto leaf) |
| **Saved for backward** | $x$, multiple intermediates | $x$ (if grad), **`rstd`** $(S,1)$ only |
| **Elided temps** | `squared` $(S,d)$, full `normalized` $(S,d)$ | Internal `ms`, `sq` never ALLOCATE |

**Apertus-8B RMSNorm call sites per layer:** pre-attn + pre-ffn $(S,d)$; Q-norm $(h,S,d_h)$; K-norm $(h_{\text{kv}},S,d_h)$.  
Per layer numel sum: $2Sd + hS d_h + h_{\text{kv}} S d_h = S(8192 + 4096 + 1024) = 13312 S$.

At $S{=}8192$, one layer, unfused extra peak (order-of-magnitude, two full $(S,d)$ temps on largest norms):  
$2 \times 8192 \times 4096 \times 2$ B × 2 norms ≈ **256 MiB** elidable per layer vs fused.  
Across $L{=}32$ **if naively summed sequentially**: fused lowering avoids counting those peaks as simultaneous only if regions replace chains — policy: **per-forward peak** = max over layers, but unfused **aggregate** VRAM accounting today can inflate totals.

**Framework files:** `src/zepto/core/lowering/implementations/regions/rmsnorm.py` (mirror `layernorm.py` pattern), pattern rule on op families `{multiply, reduce_sum, divide, add, sqrt, divide, multiply}`, provenance `component_type="RMSNorm"`.

#### G4b — `region/softmax` and `region/masked_softmax`

**Module decomposition today** (`softmax.py`): optional `scale → exp → reduce_sum → divide` on last dim; GQA adds prior `matmul → add(mask)`.

| | Unfused eager GQA score path | Fused `region/masked_softmax` |
|---|------------------------------|--------------------------------|
| **Forward FLOPs (per layer)** | Scale: $hS^2$; exp+sum+div: **$3 h S^2$**; matmul $QK^\top$: **$2 h S^2 d_h$** | Leaf: **$3 h S^2$** for softmax step (Atto); matmul stays separate or folds into `region/gqa` |
| **Peak temps** | `scores`, `scaled_scores`, `exp_scores` each $(h,S,S)$ | **`softmax_out` $P$** only ($(h,S,S)$); pre-softmax logits elided |
| **Saved** | $P$ | $P$ (same) |

At \(S{=}8192\), \(h{=}32\): one \((h,S,S)\) tensor = \(32 \times 8192^2 \times 2\) B = **4.0 GiB**. Unfused path can account for **2–3×** that in peak temporaries before fusion; fused masked softmax drops **one to two** \((h,S,S)\) buffers from the estimate (~**4–8 GiB** per layer at this \(S\) in worst-case naive summation).

**Framework change:** `region/softmax` for standalone `Softmax` module; `region/masked_softmax` pattern spans `add(mask) → exp → reduce_sum → divide` inside `GroupedQueryAttention` provenance, or fuse scale+mask+softmax as one leaf matching Atto GQA recipe.

#### G4c — `region/xielu` (optional)

**Decomposed today** (`xielu.py`): `where` branch with `square`, `exp`, `minimum`, `subtract`, multiple `multiply` — **~10+ ops** over $(S, d_{\text{ff}})$.

| | Unfused | Fused `region/xielu` |
|---|---------|----------------------|
| **Forward FLOPs / layer** | Sum of primitives ≈ **$10\text{–}12 \cdot S d_{\text{ff}}$** | Atto leaf ≈ **$8 \cdot S d_{\text{ff}}$** (tune to golden) |
| **Temps** | Branch masks, `expm1` path buffer | Single output; optional $|H|$ buffer only for `python` style (Atto) |

At $S{=}8192$, $d_{\text{ff}}{=}21504$: numel ≈ 176M → unfused temp overhead ≈ **336 MiB** per elided full buffer (bf16). Optional CUDA kernel (HF / vLLM) uses same FLOP leaf but **zero** extra $|H|$ if fused in-register.

#### G4d — `region/gqa` / Flash-style (Phase 2)

| | Eager decomposed GQA (today) | FlashAttention-style region |
|---|------------------------------|----------------------------|
| **Attention FLOPs** | $4 h S^2 d_h$ (two matmuls) + $4 h S^2$ (mask+softmax ops) | Same asymptotic FLOPs; constant factors differ |
| **Peak attention memory** | **$O(h S^2)$** materialized scores + weights | **$O(h S d_h)$** — no full $(h,S,S)$ materialization |
| **When to use** | Match HF `attn_implementation="eager"` | Match `flash_attention_2`, vLLM `Attention` |

At $S{=}8192$: Flash-style removes **~8–12 GiB** of $(h,S,S)$ activation accounting per layer from peak estimates vs naive eager — the dominant VRAM gap for long-context Apertus.

**Implementation order:** (1) `region/rmsnorm`, (2) `region/softmax` + `region/masked_softmax`, (3) golden vs Atto at $S \in \{8192, 65536\}$, (4) optional `region/xielu`, (5) `region/gqa` backend flag.

---

### G5 — Softmax numerical policy (Low)

**Reference:** HF eager: `softmax(..., dtype=torch.float32).to(query.dtype)` — stable softmax in fp32.

**Zepto today:** `Softmax` ops run at graph precision (typically bf16 metadata); no upcast edge.

#### Framework change

- Tag softmax region with `numerics="stable_fp32"` in lowering metadata
- Optional: add `Cast` ops in structural graph (increases FLOP count slightly, doubles temp bytes during softmax tile)

#### Cost impact

| | bf16-only (today) | fp32-stable (HF eager) |
|---|-------------------|------------------------|
| **FLOPs** | $3 h S^2$ | $3 h S^2$ + **$2 h S^2$** cast ops (to/from fp32) |
| **Peak temp for exp** | $h S^2 \times 2$ B | $h S^2 \times 4$ B during softmax tile |

At \(S{=}8192\): the fp32 exp tile is **8.0 GiB** versus **4.0 GiB** in bf16 — relevant when comparing to HF eager, not FlashAttention.

---

### G6 — xIELU parameter / softplus graph (Low)

**Reference:** HF stores `alpha_p`, `alpha_n` in inverse-softplus space; applies `softplus` each forward. The \(\alpha_n\) parameter is \(\mathrm{invsoftplus}(\alpha_n^{\mathrm{init}}-\beta)\), not \(\mathrm{invsoftplus}(\alpha_n^{\mathrm{init}})\).

**Zepto today:** Graph inputs `effective_alpha_p` / `effective_alpha_n` fold softplus into leaf FLOPs; parameters exist but softplus is not a graph op.

#### Framework change

| Mode | Change |
|------|--------|
| **Inference estimates (default)** | Keep folded scalars — no graph change |
| **Training estimates** | Optional `softplus` op on parameter tensors before `XIELU`; +**$2$** scalar ops (negligible) but documents train vs infer |

#### Cost impact

Negligible for Apertus-8B ($2$ FLOPs vs $8 \cdot S d_{\text{ff}} \approx 1.4 \times 10^9$ per layer FFN activation).

---

### G7 — Attention backend diversity (Phase 2)

**Reference:** `config._attn_implementation` ∈ `{eager, sdpa, flash_attention_2, flex_attention, …}`.

**Zepto today:** Single eager decomposed path in `GroupedQueryAttention`.

#### Framework change

- Lowering context flag: `attention_backend: Literal["eager", "flash", "sdpa"]`
- Select `region/gqa` implementation vs per-op chain in `registry.py`
- SDPA: **not unique** — math backend ≈ eager \(\Theta(h S^2)\) memory; flash sub-backend ≈ G4d. Pin the sub-backend in the estimation context.

#### Cost impact summary (per layer, dominant terms)

| Backend | Dominant VRAM | vs Zepto today |
|---------|---------------|----------------|
| `eager` | $(h,S,S)$ scores + weights | Baseline (matches current) |
| `sdpa` | Dispatcher-dependent (math ≈ eager; flash ≈ FA-2) | Must pin sub-backend |
| `flash_attention_2` | $O(h S d_h)$ tiles | **−$O(h S^2)$** from estimate |

Use this flag when comparing Zepto totals to vLLM-served or HF flash paths — not when targeting Atto eager golden files.

---

### G8 — Hub-kernel cost modeling (Phase 2)

When benchmarking Transformers with `use_kernels=True` / Liger hub kernels, map hub substitutions to Zepto regions so totals align:

| HF hub hook | Hub kernel | Zepto region | FLOP leaf | VRAM effect |
|-------------|------------|--------------|-----------|-------------|
| `LlamaRMSNorm` | `LigerRMSNorm` | `region/rmsnorm` | $4\text{–}5 \cdot \text{numel}$ | Elide decomposed temps (G4a) |
| `apply_rotary_pos_emb` | `kernels-community/rotary` | optional `region/rope_apply` | Decomposed ≈ $6 h S d_h$ | Elide rotated-half temps |
| `XIELUActivation` | `nickjbrowning/XIELU` CUDA | `region/xielu` | Atto leaf | Elide branch temps (G4c) |
| Attention | FlashAttention | `region/gqa` | Matmul+softmax fused | Elide $(h,S,S)$ (G4d) |

**Rule:** When a region matches, lowering must **suppress** primitive nodes inside the region boundary (same as existing `region/linear` wrapping one `linear_matmul`).

---

### G9 — Config class drift (Docs only)

`ApertusConfig.intermediate_size` default **14336** in Transformers source ≠ checkpoint **21504**. Zepto presets (`APERTUS_8B`) already use **21504**. FFN FLOPs scale linearly with $d_{\text{ff}}$:

$$
\text{FFN matmul FLOPs per layer} \approx 4 S d \cdot d_{\text{ff}} \quad (\text{up+down})
$$

Using wrong $d_{\text{ff}}$ would mis-estimate FFN by factor $14336/21504 \approx 0.67$ (−33%) — Zepto presets avoid this; treat HF class defaults as non-authoritative.

---

### Gap priority matrix (framework + cost)

| Gap | Framework work | FLOP accuracy | VRAM accuracy | Priority |
|-----|----------------|---------------|---------------|----------|
| G1 RoPE init | Config + init helper | — (correctness) | — | High |
| G4a rmsnorm region | New region impl | Medium | **High** | High |
| G4b softmax regions | New region impls | Medium | **Critical** at large $S$ | High |
| G4d flash GQA | Backend flag + region | Low | **Critical** at large $S$ | Phase 2 |
| G2 mask | Docs / extend op | — | Medium for $S{=}65536$ | Medium |
| G5 fp32 softmax | Metadata / cast ops | Low | Low–medium | Low |
| G3 KV cache | New graph mode | Changes decode FLOPs | Persistent KV billing | Defer |
| G6 xIELU softplus | Optional op | Negligible | — | Low |
| G7/G8 backends | Registry + flags | Backend-dependent | Backend-dependent | Phase 2 |
| G9 config drift | Docs only | — | — | Docs |

---

## 8. End-to-end kernel trace (typical CUDA inference)

When running `ApertusForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")` with hub kernels enabled:

```text
Embedding               PyTorch embedding lookup
RotaryEmbedding.forward PyTorch (fp32 freq math) → cos/sin tensors
DecoderLayer × L:
  RMSNorm (×2)          LigerRMSNorm hub OR eager fp32 variance
  Q/K/V linear          cuBLAS
  Q/K RMSNorm           LigerRMSNorm hub
  apply_rotary_pos_emb  rotary hub (CUDA) OR eager
  Attention             FlashAttention-2 kernel (NOT eager matmul+softmax)
  O proj                cuBLAS
  MLP up/down           cuBLAS
  xIELU                 nickjbrowning/XIELU CUDA OR Python where
Final RMSNorm           LigerRMSNorm hub
LM head                 cuBLAS
```

When running **eager** attention (`attn_implementation="eager"`), attention falls back to explicit matmul + **fp32 softmax** + matmul; other hub substitutions may still apply to RMSNorm/RoPE.

---

## 9. Suggested Zepto implementation order

Derived from this investigation and `module-building-plan.md`:

1. **Llama3 RoPE init** on `RoPEMaterialize` / `Apertus` factory (G1) — checkpoint \(\Theta=12\times 10^6\), not paper pretraining \(5\times 10^5\)
2. **`region/rmsnorm`** fused lowering (G4) — Appendix E / Liger leaf \(4\cdot\mathrm{numel}\), save `rstd` only; see [`docs/kernel-implementation.md`](docs/kernel-implementation.md) §5
3. **`region/softmax`** + masked variant for GQA scores (G4) — Appendix E \(3 h S^2\); elide extra \((h,S,S)\) temps
4. Golden tests: 8B preset, `seq_len` ∈ {8192, 65536}, compare structural node counts / FLOPs to Atto Apertus recipe
5. Optional: **`region/xielu`**, flash GQA region, KV-cache graph (G3, G7)

---

## 10. Quick reference — file → responsibility

| Transformers symbol | Defined in | Kernel hook |
|--------------------|------------|-------------|
| `ApertusConfig` | `modular_apertus.py` | — |
| `ApertusMLP` | `modular_apertus.py` | xIELU optional CUDA |
| `ApertusRMSNorm` | `LlamaRMSNorm` | `@use_kernel_forward_from_hub("RMSNorm")` |
| `ApertusRotaryEmbedding` | `LlamaRotaryEmbedding` | Python cos/sin; llama3 init |
| `apply_rotary_pos_emb` | `modeling_llama.py` | `@use_kernel_forward_from_hub("rotary_pos_emb")` |
| `ApertusAttention` | `modular_apertus.py` | `@use_kernelized_func` + attention backend |
| `eager_attention_forward` | `modeling_llama.py` | PyTorch matmul + fp32 softmax |
| `XIELUActivation` | `activations.py` | Optional `torch.classes.xielu.XIELU` |

---

*Corrected 2026-08-26 from Transformers `main`, swiss-ai Instruct-2509 checkpoints, vLLM `main`, Apertus arXiv:2509.14233 (incl. Table 5 RoPE \(\Theta\) schedule and Appendix E FLOPs), and Zepto `zepto-module-api` working tree. Kernel mathematics: `docs/kernel-implementation.md`.*
