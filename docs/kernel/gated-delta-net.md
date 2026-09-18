# Zepto kernel research: Gated DeltaNet (full mixer block)

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Full Qwen3-Next / Qwen3.8 **GatedDeltaNet** token-mixer block (`GatedDeltaNet` module); prefill (\(S>1\)) and decode (\(S=1\)) with conv + scan recurrent state ports; training + inference; CUDA primary (FLA `GatedDeltaNet` layer); Tier-A leaf composition vs block envelope `region/gated_delta_net`
**Context:** Zepto reference module [`src/zepto/modules/gated_delta_net.py`](../src/zepto/modules/gated_delta_net.py) composes projections, depthwise causal conv, dual L2-normalize on Q/K, gated delta scan, gated RMSNorm output, and output projection. Child kernel research lives in `docs/kernel/{l2-normalize,depthwise-causal-conv1d,gated-delta-scan,gated-rms-norm}.md`. Default **block** costing = **sum of kernel-accurate Tier-A fused leaves** (no double-count) until `region/gated_delta_net` implements a single envelope recipe; FLA **mega-fusion** (`use_gate_in_kernel`, in-kernel Q/K L2) changes **which region ids** bill, not asymptotic MAC totals when configured consistently.

---

## Section 0: Mathematical definition

Let batch \(B=1\) in the structural graph (scale by \(B\) for batched layouts). Residual input \(\mathbf X \in \mathbb{R}^{S \times d}\). Write \(d_Q = h_k d_k\), \(d_V = h_v d_v\), repeat factor \(r = h_v / h_k\), conv channel count \(C = 2 d_Q + d_V\).

**Projections** (two linear maps from \(\mathbf X\)):

\[
[\mathbf Q_0, \mathbf K_0, \mathbf V_0, \mathbf Z]
  = \mathbf X \mathbf W_{qkvz}, \quad
\mathbf W_{qkvz} \in \mathbb{R}^{d \times (2 d_Q + 2 d_V)},
\]
\[
[\mathbf b, \mathbf a] = \mathbf X \mathbf W_{ba}, \quad
\mathbf W_{ba} \in \mathbb{R}^{d \times 2 h_v}.
\]

**Local causal depthwise conv** (Q, K, V only; SiLU activation; no bias on Qwen checkpoint):

\[
[\widetilde{\mathbf Q}, \widetilde{\mathbf K}, \widetilde{\mathbf V}]
  = \operatorname{SiLU}\bigl(\operatorname{DepthwiseCausalConv1d}_{K}([\mathbf Q_0; \mathbf K_0; \mathbf V_0])\bigr).
\]

Reshape to heads \(\mathbf Q \in \mathbb{R}^{S \times h_k \times d_k}\), \(\mathbf K\) likewise, \(\mathbf V \in \mathbb{R}^{S \times h_v \times d_v}\). **L2-normalize** last dim (fp32 reduction, FLA/Qwen semantics):

\[
\widehat{\mathbf Q} = \operatorname{L2Norm}(\mathbf Q), \quad
\widehat{\mathbf K} = \operatorname{L2Norm}(\mathbf K).
\]

**GVA repeat** (no MACs — structural `RepeatKV`): broadcast each Q/K head to \(r\) value heads → shapes \((S, h_v, d_k)\).

**Gate preparation** (owned by mixer, not `GatedDeltaScan` leaf):

\[
\widetilde{\mathbf Q} = d_k^{-1/2} \odot \widehat{\mathbf Q}, \quad
\beta = \sigma(\mathbf b), \quad
g = -\exp(A_{\log}) \odot \operatorname{softplus}(\mathbf a + \mathrm{dt\_bias}),
\]

with learned \(\exp(A_{\log})\) and `dt_bias` per value head ([`GatedDeltaNet`](../src/zepto/modules/gated_delta_net.py), [HF `Qwen3NextGatedDeltaNet`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)).

**Gated delta scan** (per head matrix state \(\mathbf H_t \in \mathbb{R}^{d_k \times d_v}\)):

\[
\alpha_t = \exp(g_t), \quad
\widetilde{\mathbf H}_t = \alpha_t \odot \mathbf H_{t-1}, \quad
\widehat{\mathbf v}_t = \mathbf k_t^\top \widetilde{\mathbf H}_t,
\]
\[
\boldsymbol\delta_t = \beta_t \odot (\mathbf v_t - \widehat{\mathbf v}_t), \quad
\mathbf H_t = \widetilde{\mathbf H}_t + \mathbf k_t \boldsymbol\delta_t^\top, \quad
\mathbf o_t = \mathbf q_t^\top \mathbf H_t.
\]

**Output norm + gate** (per value head, Zepto loops heads; bill one leaf over total numel):

\[
\mathbf y = \operatorname{GatedRMSNorm}(\mathbf o, \mathbf z) = \gamma \odot \operatorname{RMSNorm}(\mathbf o) \odot \operatorname{SiLU}(\mathbf z).
\]

**Output projection:** \(\mathbf Y = \mathrm{flatten}(\mathbf y)\, \mathbf W_o\), \(\mathbf W_o \in \mathbb{R}^{d_V \times d}\).

**I/O and state shapes** (rank-2 graph):

| Tensor / state | Shape | Notes |
|----------------|-------|-------|
| `hidden_states` | \((S, d)\) | Residual stream |
| `output` | \((S, d)\) | Mixer output |
| `conv_state` | \((C, K)\) | Rolling buffer ([`Conv1DState`](../src/zepto/analysis/horizon/state.py)) |
| `scan_state` | \((h_v, d_k, d_v)\) | fp32 accumulation in reference |
| Q/K/V/Z splits | \((S, d_Q)\), \((S, d_Q)\), \((S, d_V)\), \((S, d_V)\) | From `qkvz_proj` |
| \(b, a\) | \((S, h_v)\) each | From `ba_proj` |

**Numerics (bytes, not FLOPs):** L2 and scan state use fp32 compute dtype (`RMSNORM_COMPUTE_DTYPE`); activations default bf16. Decay \(g \le 0\) ensures \(\alpha \le 1\). No causal attention mask — recurrence replaces \(S \times S\) attention.

**Structural vs production:** Zepto identity graph materializes `Split`/`Concat`/`Transpose`/`Reshape`/`RepeatKV` between leaves; fused FLA [`GatedDeltaNet`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py) may fuse conv+scan+norm flags (`use_gate_in_kernel`, `use_qk_l2norm_in_kernel`) — composition rules in §3 prevent double billing.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | FLA **`GatedDeltaNet`** layer (`mode=chunk` prefill, `fused_recurrent` decode) calling `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`, optional hub Triton conv ([`fla/layers/gated_deltanet.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py)); HF modular Qwen3-Next delegates to FLA when installed |
| **Identity lowering** | Unfused semantic chain | Zepto **`GatedDeltaNet`**: full primitive graph through child modules (`Linear`, `DepthwiseCausalConv1d`, `L2Normalize`×2, gate ops, `GatedDeltaScan` S-unroll, per-head `GatedRMSNorm`, `Linear`) — used when `requested_capabilities` lacks `fused` or no region match |
| **Zepto fused region** | Kernel-accurate cost leaf | **`region/gated_delta_net`** (stub in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py)) — **composition envelope** equal to Tier-A leaf sum until implementer adds block recipe; mutually exclusive with billing the same module instance’s child regions |

**Default estimates:** Use **kernel-accurate Tier-A fused leaves** summed with §3 mutual-exclusion rules (same as future block envelope). Do **not** sum unfused primitive FLOPs for production leaves.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **Zepto identity module** | [`GatedDeltaNet`](../src/zepto/modules/gated_delta_net.py) | No (composed) | Block | Any | Both; emits conv + scan state ports |
| **HF eager Qwen3-Next** | [`modeling_qwen3_next.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) | Partial | Block | CUDA / CPU | Both |
| **HF modular + FLA** | [`modular_qwen3_next.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py) | Yes when FLA installed | Block | CUDA | Both |
| **FLA GatedDeltaNet layer** | [`fla/layers/gated_deltanet.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py) | Yes (configurable sub-fusions) | Block / mega | CUDA / ROCm | `chunk` vs `fused_recurrent` |
| **FLA chunk GDR** | [`chunk_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py) | Yes (scan sub-leaf) | Recurrence | CUDA / ROCm | Prefill + training |
| **Dao causal-conv1d** | [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | Yes (conv sub-leaf) | A | CUDA | Both |
| **torchtitan Qwen3.5 reference** | [`model.py`](https://github.com/pytorch/torchtitan/blob/cd8950ba/torchtitan/models/qwen3_5/model.py) | No | Block | Any | Reference numerics |
| **Atto cost framework** | [34 - gated-deltanet.md](https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md) | N/A (analytic) | Block | N/A | Golden cross-check |
| **Zepto block region (stub)** | `region/gated_delta_net` | Envelope (TBI) | Block | `fused` gate | `status: to_implement` |
| **Zepto Tier-A leaves** | `region/linear`, `region/l2_normalize/*`, `region/depthwise_causal_conv1d/*`, `region/gated_delta_scan/*`, `region/gated_rms_norm/*` | Yes (per leaf) | A / recurrence | Mixed | Primary costing path today |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **Mixer block envelope** — one `GatedDeltaNet` forward invocation maps to either (a) **one** `region/gated_delta_net` leaf, or (b) a **set of Tier-A leaves** listed below. This is **not** attention boundary C/D; the scan recurrence uses **`region/gated_delta_scan`** (recurrence leaf, see [`gated-delta-scan.md`](../docs/kernel/gated-delta-scan.md)).

**Mutually exclusive (same `GatedDeltaNet` module instance, same forward):**

- `region/gated_delta_net` **subsumes** all child regions on matched provenance ([`step5` §8.1](../docs/plans/step5-stateful-mixers.md)) — do **not** stack block + leaves.
- `region/gated_rms_norm/*` vs separate `region/rmsnorm` + `region/silu` on `GatedRMSNorm` provenance.
- `region/gated_delta_scan` with **`use_qk_l2norm_in_kernel=True`** vs two standalone `region/l2_normalize` on the same Q/K tensors ([`l2-normalize.md`](../docs/kernel/l2-normalize.md) §3).
- `chunk_gated_delta_rule(..., use_gate_in_kernel=True)` vs separate `region/gated_rms_norm` on scan output ([`gated-rms-norm.md`](../docs/kernel/gated-rms-norm.md) §3).

**Composable (default Zepto stack — bill each leaf once):**

1. `region/linear` — `qkvz_proj`, `ba_proj`, `o_proj` (three matmul regions).
2. `region/depthwise_causal_conv1d/*` — \(C = 2 d_Q + d_V\), SiLU, \(K=4\) default.
3. `region/l2_normalize/*` — Q and K (\(n_Q = S h_k d_k\) each).
4. **Glue ops** in parent — Q \(d_k^{-1/2}\) scale, \(\beta=\sigma(b)\), softplus decay chain (§4.1.5); no dedicated region today.
5. `region/gated_delta_scan/*` — prefill vs `.../decode` when \(S=1\) + recurrent port.
6. `region/gated_rms_norm/*` — \(n = S h_v d_v\).
7. Horizon **`PERSIST`** — `conv_state_out`, `scan_state_out` (gap **G3** cross-step bytes).

**Execution constraints:**

- `num_v_heads % num_qk_heads == 0` ([`GatedDeltaNetConfig`](../src/zepto/modules/mixer_config.py)).
- Decode: projections scale with \(S=1\); conv/scan use decode variants + state **PERSIST** / **LOAD**.
- `requires_grad=False` → leaf backward FLOPs = 0; scan saves only final state at inference.

---

## Section 4: Forward FLOPs — step-by-step derivation

Notation: one multiply-add = **2 FLOPs**. Subscripts: \(h_k\) Q/K heads, \(h_v\) value heads, \(d_k = d_v =\) `head_dim` unless noted.

### 4.1 Identity lowering (Zepto `GatedDeltaNet` — decomposed bill)

#### 4.1.1 Projections (`region/linear`)

| Step | Primitive | Tile | FLOPs rule | Subtotal |
|------|-----------|------|------------|----------|
| 1 | `LinearMatMul` qkvz | \((S,d) \times (d, 2d_Q+2d_V)\) | \(2 S d (2d_Q+2d_V)\) | \(4 S d (d_Q + d_V)\) |
| 2 | `LinearMatMul` ba | \((S,d) \times (d, 2h_v)\) | \(2 S d \cdot 2 h_v\) | \(4 S d h_v\) |
| 3 | `LinearMatMul` o_proj | \((S,d_V) \times (d_V,d)\) | \(2 S d_V d\) | \(2 S d_V d\) |

#### 4.1.2 Depthwise conv + SiLU (`region/depthwise_causal_conv1d`)

\(n = S C\), \(f_{\mathrm{conv}} = 2K-1 + 5\mathbb{1}_{\mathrm{SiLU}} = 12\) for \(K=4\), no bias ([`depthwise-causal-conv1d.md`](../docs/kernel/depthwise-causal-conv1d.md)):

| Step | Primitive | FLOPs | Subtotal |
|------|-----------|-------|----------|
| 1 | Depthwise MAC window | \((2K-1)n\) | \(7 S C\) |
| 2 | SiLU | \(5n\) | \(5 S C\) |
| **Subtotal** | | | \(12 S C\) |

#### 4.1.3 L2 normalize Q and K (`region/l2_normalize` ×2)

Per tensor \(n_Q = S h_k d_k\), fused leaf **\(3 n_Q\)** ([`l2-normalize.md`](../docs/kernel/l2-normalize.md)):

\[
\mathrm{FLOPs}_{\mathrm{L2,pair}} = 6 S h_k d_k.
\]

#### 4.1.4 Gate glue (parent module — no region)

| Step | Primitive | Per \((t, h)\) | Subtotal |
|------|-----------|----------------|----------|
| 1 | `Sigmoid` on \(b\) | 4 | \(4 S h_v\) |
| 2 | Bias + `Softplus` chain on \(a\) | \(\approx 5\) | \(5 S h_v\) |
| 3 | Decay scale + neg | 2 | \(2 S h_v\) |
| 4 | Q scale by \(d_k^{-1/2}\) | \(d_k\) | \(S h_v d_k\) |
| **Glue total** | | | \(S h_v (d_k + 11)\) |

(`RepeatKV`, `Split`, `Concat`, `Reshape`, `Transpose`: **0 FLOPs**.)

#### 4.1.5 Scan (`region/gated_delta_scan`)

Per [`gated-delta-scan.md`](../docs/kernel/gated-delta-scan.md) §4.1:

\[
\mathrm{FLOPs}_{\mathrm{scan}} = S h_v \bigl(8 d_k d_v + 2 d_v + 1\bigr).
\]

**Decode:** \(h_v (8 d_k d_v + 2 d_v + 1)\) per token (plus projections at \(S=1\)).

#### 4.1.6 Gated RMSNorm (`region/gated_rms_norm`)

\(n = S h_v d_v\), fused leaf **\(10 n\)** ([`gated-rms-norm.md`](../docs/kernel/gated-rms-norm.md)):

\[
\mathrm{FLOPs}_{\mathrm{o\_norm}} = 10 S h_v d_v.
\]

### 4.2 Fused region leaf (kernel-accurate block envelope)

The block envelope matches **§4.1 sum** when each Tier-A leaf uses its fused recipe (not unfused primitive sums):

\[
\begin{aligned}
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{block}}
= &\ 4 S d (d_Q + d_V) + 4 S d h_v + 12 S C \\
&+ 6 S h_k d_k + S h_v (d_k + 11) \\
&+ S h_v \bigl(8 d_k d_v + 2 d_v + 1\bigr) + 10 S h_v d_v + 2 S d_V d.
\end{aligned}
\]

Substitute \(C = 2 d_Q + d_V\), \(d_Q = h_k d_k\), \(d_V = h_v d_v\).

**Dominant terms** (typical Qwen3.8: \(d \gg d_k\)): **qkvz** and **o_proj** GEMMs (\(\mathcal{O}(S d d_V)\)), then **scan** (\(\mathcal{O}(S h_v d_k d_v)\)).

### 4.3 Paper-comparable (Atto block vs Zepto leaves)

Atto groups scan recurrence as **\(9 S h_v d_k d_v\)** when \(d_k = d_v\) ([Atto §Computation](https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md)) vs Zepto scan leaf **\(8 S h_v d_k d_v\)** — label **paper-comparable** when citing Atto totals. Atto full-block tables include projections + conv + norm; **cross-check block sum** against Atto rather than replacing GEMM counts.

Gated RMSNorm paper stacks sometimes count **RMS only (\(4n\))** and omit SiLU gate **\(6n\)** — kernel-accurate Zepto includes **\(10n\)** ([`gated-rms-norm.md`](../docs/kernel/gated-rms-norm.md) §4.3).

**Closed form (kernel-accurate default):**

\[
\boxed{
\mathrm{FLOPs}_{\mathrm{fwd}} = \mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{block}}
}
\]

**Arithmetic intensity (numeric example — Qwen3.8-27B language DeltaNet layer):**

| Symbol | Value |
|--------|-------|
| \(S\) | 8192 |
| \(d\) | 5120 |
| \(h_k, h_v\) | 16, 48 |
| \(d_k, d_v\) | 128 |
| \(d_Q, d_V\) | 2048, 6144 |
| \(C\) | 10240 |
| bf16 \(e\) | 2 |

| Leaf | Approx. FLOPs |
|------|----------------|
| qkvz + ba + o_proj | \(\approx 1.37 \times 10^{12} + 8.1 \times 10^{9} + 5.15 \times 10^{11} \approx 1.89 \times 10^{12}\) |
| conv | \(12 \cdot S \cdot C \approx 1.01 \times 10^{9}\) |
| L2 pair | \(\approx 1.0 \times 10^{8}\) |
| glue + scan + o_norm | \(\approx 5.7 \times 10^{10}\) |
| **Total** | \(\approx 1.95 \times 10^{12}\) FLOPs/layer/prefill |

Minimum HBM touched (single read/write per activation tensor, no fusion): dominated by **qkvz input** \(S d e\), **qkvz output** \((2d_Q+2d_V) S e\), **conv** \(2 S C e\), **scan I/O** \(\approx (3 d_k + d_v + 2) S h_v e\), **o_proj** — order **\(10^9\)–\(10^{10}\)** bytes vs **\(10^{12}\)** FLOPs → block is **compute-heavy on GEMMs**, **memory-bound** on norm/conv slices (same pattern as [`gated-rms-norm.md`](../docs/kernel/gated-rms-norm.md) AI \(\approx 2.5\) FLOP/byte for norm alone).

**Apertus-8B note:** Default Apertus uses GQA attention, not DeltaNet. Use the Qwen3.8 trace above for DeltaNet layers; Apertus \((S,d,h,d_h)\) applies only when substituting a DeltaNet layer at comparable \(S\).

---

## Section 5: Backward FLOPs — step-by-step derivation

Training context (`requires_grad=True`). Inference-only → **backward FLOPs = 0** for all leaves.

Block backward = sum of leaf backward formulas (when block region enabled, same sum):

| Component | Backward (kernel-accurate leaf) | Source |
|-----------|----------------------------------|--------|
| Each `region/linear` | \(\approx 2 \times\) forward matmul | Standard GEMM bwd |
| Conv + SiLU | \((4K+6) n = 22 n\) for \(K=4\) | [`depthwise-causal-conv1d.md`](../docs/kernel/depthwise-causal-conv1d.md) §5 |
| L2 ×2 | \(4 n_Q\) each → \(8 S h_k d_k\) | [`l2-normalize.md`](../docs/kernel/l2-normalize.md) |
| Glue | \(\approx 2 \times\) forward glue | Elementwise autograd |
| Scan | \(S h_v (16 d_k d_v + 4 d_v + 4)\) | [`gated-delta-scan.md`](../docs/kernel/gated-delta-scan.md) |
| Gated RMSNorm | \(14 n\), \(n = S h_v d_v\) | [`gated-rms-norm.md`](../docs/kernel/gated-rms-norm.md) |

**Closed form:**

\[
\mathrm{FLOPs}_{\mathrm{bwd}}^{\mathrm{block}}
= \mathrm{FLOPs}_{\mathrm{bwd}}^{\mathrm{linear}}
+ 22 S C + 8 S h_k d_k + 2 \cdot \mathrm{FLOPs}_{\mathrm{glue,fwd}}
+ S h_v (16 d_k d_v + 4 d_v + 4) + 14 S h_v d_v.
\]

Atto scan backward **\(10 S h_v d_k d_v\)** remains **paper-comparable** vs Zepto **\(16 S h_v d_k d_v\)** dominant term.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

### 6.1 Transient allocations (forward)

| Path | Live buffers (representative) | Peak bytes (order) | Elided by block / FLA fusion |
|------|------------------------------|--------------------|------------------------------|
| Identity block | qkvz, conv intermediates, per-step scan temps, per-head norm temps | Sum of child peaks; scan worst case **per-step** \(\mathbf H_t\) materialization | `region/gated_delta_net` or FLA layer |
| Tier-A fused | Leaf outputs only + workspace | Scan chunk workspace; conv fused path drops per-lag partials | FLA `chunk_gated_delta_rule` |
| Decode | Same at \(S=1\) | Conv state **PERSIST** \((C,K)e\); scan state \((h_v d_k d_v) e_{\mathrm{fp32}}\) | Recurrent kernels |

**Qwen3.8 example (prefill, bf16 \(e=2\), fp32 state \(e=4\)):**

- Conv state: \(C K e = 10240 \cdot 4 \cdot 2 \approx 80\) KiB/layer.
- Scan state out: \(h_v d_k d_v \cdot 4 = 48 \cdot 128^2 \cdot 4 \approx 3.1\) MiB/layer (fp32).
- Training scan checkpoint: up to **\(S h_v d_k d_v\)** fp32 saved — **§6.2**.

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes (order) | Zepto SAVE policy |
|------|---------------|-------|---------------|-------------------|
| L2 Q/K | `rstd` | \((S, h_k, 1)\) fp32 ×2 | \(2 \cdot 4 S h_k\) | `SAVE(rstd)` per [`l2-normalize.md`](../docs/kernel/l2-normalize.md) |
| Conv + SiLU | pre-activation | \((S, C)\) | \(S C e\) | `SAVE(pre_activation)` when SiLU |
| Scan | `state_t` or checkpoint | \([S, h_v, d_k, d_v]\) fp32 | \(4 S h_v d_k d_v\) | `SAVE(state_checkpoint)` [`gated-delta-scan.md`](../docs/kernel/gated-delta-scan.md) |
| Gated RMSNorm | `rstd` | \((S, h_v, 1)\) fp32 | \(4 S h_v\) | `SAVE(rstd)`; value/gate from upstream |
| Linears | standard autograd | weights + inputs | GEMM policy | `region/linear` |

Inference decode: **only final** `scan_state_out` and `conv_state_out` persist across horizon steps (**G3**).

### 6.3 Resource event chains

**Identity lowering (unfused block — conceptual concatenation):**

```
ALLOCATE(qkvz) → ALLOCATE(ba) → ALLOCATE(conv_out) → PERSIST(conv_state_out)
→ ALLOCATE(q_heads,k_heads) → ALLOCATE(rstd_q,rstd_k) → SAVE(rstd_q) → SAVE(rstd_k)
→ ALLOCATE(beta,g) → WORKSPACE(scan_temps) → SAVE(state_t…) → ALLOCATE(scan_out)
→ ALLOCATE(rstd_o) → SAVE(rstd_o) → ALLOCATE(output) → RELEASE(temps)
```

**Fused Tier-A leaf stack (default costing path):**

```
# linear ×3: ALLOCATE(outputs) per region/linear recipe
# conv: ALLOCATE(y) → SAVE(pre_activation?) → PERSIST(conv_state_out)
# l2×2: ALLOCATE(y) → SAVE(rstd)
# scan: ALLOCATE(output) → WORKSPACE(chunk) → PERSIST(scan_state_out) → SAVE(state_checkpoint)
# gated_rms: ALLOCATE(y) → SAVE(rstd)
# o_proj: ALLOCATE(output)
```

**Fused block envelope (`region/gated_delta_net` — future):**

```
ALLOCATE(mixer_output) → PERSIST(conv_state_out) → PERSIST(scan_state_out)
→ SAVE(training_auxiliaries per merged policy)
```

Child **identity** chains remain mandatory in architecture even when recommending block fusion ([`kernel-search` gotchas](../.pi/skills/kernel-search/references/gotchas.md)).

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Zepto Tier-A sum | Zepto recipes | Per leaf | Block compose | §4.2 closed form | §5 sum | Leaf SAVE policies | Multiple |
| Zepto block stub | `mixer_stub.py` | Envelope TBI | Block | Same as sum | Same as sum | Merged TBI | `region/gated_delta_net` |
| FLA GatedDeltaNet | FLA layer | Yes | Block / mega | Match §4.2 when flags align | Fused autograd | Optional in-kernel L2/gate | (external) |
| HF + FLA modular | Transformers | Partial | Block | FLA or eager | FLA or eager | Model cache | — |
| Atto analytic | Atto doc | N/A | Block | Cross-check | Cross-check | All states | golden |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: gated_delta_net
recommended_region_ids:
  - id: region/gated_delta_net
    variant: composed_tier_a
    hardware_gate: any
    fusion_boundary: block
    status: to_implement
  - id: region/gated_delta_net
    variant: fla_layer_parity
    hardware_gate: cuda
    fusion_boundary: block
    status: to_implement
  - id: region/gated_delta_net/decode
    variant: composed_tier_a_decode
    hardware_gate: any
    fusion_boundary: block
    status: to_implement
pattern_rule:
  op_families:
    - LinearMatMul
    - DepthwiseCausalConv1d
    - L2Normalize
    - Sigmoid
    - Softplus
    - GatedDeltaScan
    - GatedRMSNorm
    - RepeatKV
    - Reshape
    - Split
    - Concat
    - Transpose
recipe:
  forward_flops: >-
    4*S*d*(d_Q + d_V) + 4*S*d*h_v + 12*S*C + 6*S*h_k*d_k + S*h_v*(d_k + 11)
    + S*h_v*(8*d_k*d_v + 2*d_v + 1) + 10*S*h_v*d_v + 2*S*d_V*d
  backward_flops: >-
    2*(4*S*d*(d_Q + d_V) + 4*S*d*h_v + 2*S*d_V*d) + 22*S*C + 8*S*h_k*d_k
    + 2*S*h_v*(d_k + 11) + S*h_v*(16*d_k*d_v + 4*d_v + 4) + 14*S*h_v*d_v
  elided_temps:
    - per_step_scan_decoded_prediction
    - per_step_scan_outer
    - conv_per_lag_partials
    - l2_squared_normalized_full_rank
    - gated_rms_silu_intermediate
  saved_backward:
    - "l2 rstd_q, rstd_k: [S, h_k, 1] fp32"
    - "conv pre_activation: [S, C] when SiLU training"
    - "scan state_checkpoint: [S, h_v, d_k, d_v] fp32"
    - "gated_rms rstd: [S, h_v, 1] fp32"
  resource_events_forward:
    - "PERSIST(conv_state_out: [C, K])"
    - "PERSIST(scan_state_out: [h_v, d_k, d_v])"
    - "COMPOSE from child region chains without double ALLOCATE on same tensor"
  resource_events_backward:
    - "LOAD(child SAVE tensors per §6.2)"
  numerics_tags:
    - fp32_l2_reduction
    - fp32_scan_state
    - log_decay_gates
    - structural_repeat_kv_zero_flop
capabilities:
  - fused
  - conv_state_port
  - recurrent_state_port
priority: 7
variants:
  prefill:
    trigger: "seq_len > 1 or phase == prefill"
    forward_flops: "recipe.forward_flops with full S"
  decode:
    trigger: "seq_len == 1 and (conv_state_in or scan_state_in)"
    forward_flops: >-
      4*d*(d_Q + d_V) + 4*d*h_v + 12*C + 6*h_k*d_k + h_v*(d_k + 11)
      + h_v*(8*d_k*d_v + 2*d_v + 1) + 10*h_v*d_v + 2*d_V*d
composition:
  subsumes_when_matched:
    - region/linear
    - region/depthwise_causal_conv1d
    - region/l2_normalize
    - region/gated_delta_scan
    - region/gated_rms_norm
  do_not_double_bill:
    - "region/l2_normalize when fused scan use_qk_l2norm_in_kernel"
    - "region/gated_rms_norm when chunk_gated_delta_rule use_gate_in_kernel"
  child_decode_variants:
    - region/depthwise_causal_conv1d/decode
    - region/gated_delta_scan/decode
symbols:
  d_Q: "num_qk_heads * head_dim"
  d_V: "num_v_heads * head_dim"
  C: "2 * d_Q + d_V"
  h_k: num_qk_heads
  h_v: num_v_heads
  K: conv_kernel_size
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Yang et al., Gated Delta Networks (ICLR 2025): https://arxiv.org/abs/2412.06464
- Yang et al., Parallelizing Linear Transformers with the Delta Rule: https://arxiv.org/abs/2406.06484
- Schlag et al., Linear Transformers Are Secretly Fast Weight Programmers: https://arxiv.org/abs/2006.16216
- Atto cost framework — Gated DeltaNet (block golden): https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md
- HF Qwen3-Next GatedDeltaNet: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py
- HF modular Qwen3-Next (FLA delegation): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modular_qwen3_next.py
- FLA GatedDeltaNet layer: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
- FLA chunk gated delta rule: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
- torchtitan Qwen3.5 reference: https://github.com/pytorch/torchtitan/blob/cd8950ba/torchtitan/models/qwen3_5/model.py
- Dao-AILab causal-conv1d: https://github.com/Dao-AILab/causal-conv1d
- Zepto plan Step 5 stateful mixers: [`docs/plans/step5-stateful-mixers.md`](../docs/plans/step5-stateful-mixers.md)
- Zepto child kernel archives: [`docs/kernel/l2-normalize.md`](../docs/kernel/l2-normalize.md), [`docs/kernel/depthwise-causal-conv1d.md`](../docs/kernel/depthwise-causal-conv1d.md), [`docs/kernel/gated-delta-scan.md`](../docs/kernel/gated-delta-scan.md), [`docs/kernel/gated-rms-norm.md`](../docs/kernel/gated-rms-norm.md)

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 7 | `region/gated_delta_net` | §8 | Block | Qwen3.8 DeltaNet horizon costing; stub in `mixer_stub.py` — needs compose recipe + mutual-exclusion with Tier-A leaves |
| 7 | `region/gated_delta_net/decode` | §8 variants.decode | Block | Single-token decode envelope matching child decode regions |
| — | (composition TBD) | §3 | Block | Architect must finalize `use_qk_l2norm_in_kernel` vs standalone L2 double-count rule when FLA parity variant lands |

**Open gaps (not blocking research):**

- Block mega-fusion parity variant (`fla_layer_parity`) SAVE policy vs Tier-A sum — **TBD** in kernel-architect pass.
- Golden numeric test vs Atto block table — **TBD** in implementer tests (`tests/lowering/regions/test_gated_delta_net.py`).
