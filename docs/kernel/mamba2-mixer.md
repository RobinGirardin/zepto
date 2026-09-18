# Zepto kernel research: Mamba-2 mixer (full block)

**Date:** 2026-09-17
**Proposer:** Robin Girardin
**Scope:** Full Nemotron-style **Mamba2Mixer** token-mixer block (`Mamba2Mixer` module); prefill (\(S>1\)) and decode (\(S=1\)) with depthwise conv + selective SSM recurrent state ports; training + inference; CUDA primary (`mamba-ssm` Triton SSD + optional Hub mega-kernel); Tier-A leaf composition vs block envelope `region/mamba2_mixer`
**Context:** Zepto reference module [`src/zepto/modules/mamba2_mixer.py`](../src/zepto/modules/mamba2_mixer.py) composes `in_proj`, depthwise causal conv on the SSM payload, selective SSM scan, SiLU-gate-before-grouped-RMSNorm, and `o_proj`. Child kernel research: [`docs/kernel/depthwise-causal-conv1d.md`](../docs/kernel/depthwise-causal-conv1d.md), [`docs/kernel/mamba2-scan.md`](../docs/kernel/mamba2-scan.md), [`docs/kernel/gated-grouped-rms-norm.md`](../docs/kernel/gated-grouped-rms-norm.md). Default **block** costing = **sum of kernel-accurate Tier-A fused leaves** (no double-count) until `region/mamba2_mixer` implements a single envelope recipe; Hub **`mamba2_split_conv1d_scan_combined`** (boundary **B**) changes **which region ids** bill and which HBM temps elide, not leading MAC totals when configured consistently.

---

## Section 0: Mathematical definition

Let batch \(B=1\) in the structural graph (scale FLOPs/bytes by \(B\) for \((B,S,d)\)). Residual input \(\mathbf X \in \mathbb{R}^{S \times d}\). Define intermediate width \(m = h p\) (heads × head dim), conv channel count \(C_c = m + 2 G N\) (SSM input + grouped \(B\) + grouped \(C\)), and projection width \(P_{\mathrm{in}} = m + C_c + h\) ([`Mamba2MixerConfig`](../src/zepto/modules/mixer_config.py), [step5 §7.12](../docs/plans/step5-stateful-mixers.md)).

**Input projection** (single linear from \(\mathbf X\)):

\[
[\mathbf Z,\, \mathbf U_{xBC},\, \boldsymbol\Delta]
= \mathbf X \mathbf W_{\mathrm{in}},
\qquad
\mathbf W_{\mathrm{in}} \in \mathbb{R}^{d \times P_{\mathrm{in}}},
\]

with \(\mathbf Z \in \mathbb{R}^{S \times m}\) (gate, bypasses conv/scan), \(\mathbf U_{xBC} \in \mathbb{R}^{S \times C_c}\) (conv payload), \(\boldsymbol\Delta \in \mathbb{R}^{S \times h}\) (raw time steps, bypasses conv).

**Depthwise causal conv + SiLU** (payload only):

\[
\widetilde{\mathbf U} = \operatorname{SiLU}\bigl(\operatorname{DepthwiseCausalConv1d}_{K}(\mathbf U_{xBC})\bigr).
\]

Split \(\widetilde{\mathbf U}\) into \(\mathbf x_{\mathrm{flat}} \in \mathbb{R}^{S \times m}\), \(\mathbf B \in \mathbb{R}^{S \times G N}\), \(\mathbf C \in \mathbb{R}^{S \times G N}\); reshape \(\mathbf x\) to \((S,h,p)\), \(\mathbf B,\mathbf C\) to \((S,G,N)\).

**Selective SSM scan** ([Mamba-2, Dao & Gu 2024](https://arxiv.org/abs/2405.21060); [`SelectiveSSMScan`](../src/zepto/modules/selective_ssm_scan.py)) — per head matrix state \(\mathbf H_t \in \mathbb{R}^{p \times N}\):

\[
\begin{aligned}
\overline{\Delta}_{t,i} &= \operatorname{softplus}(\Delta_{t,i} + \mathrm{dt\_bias}_i),\\
H_{t,i} &= \exp(\overline{\Delta}_{t,i} A_i) \odot H_{t-1,i} + (\overline{\Delta}_{t,i} \odot B_{t,\gamma(i)}) \odot x_{t,i},\\
y_{t,i} &= C_{t,\gamma(i)}^\top H_{t,i} + D_i \odot x_{t,i},
\end{aligned}
\]

with group map \(\gamma(i) = \lfloor i / (h/G) \rfloor\). Learned \(A_i\) (via `decay_rate`), \(D_i\) (`d_skip`), and `dt_bias` live on the scan module, not in parent projections.

**Gated grouped RMSNorm** (gate **before** group RMS — Nemotron / Zamba order):

\[
\mathbf u = \mathbf y \odot \operatorname{SiLU}(\mathbf Z), \quad
\widehat{\mathbf y} = \operatorname{GroupedRMSNorm}(\mathbf u; \boldsymbol\gamma, G),
\]

with \(\mathbf y = \mathrm{flatten}(\text{scan output}) \in \mathbb{R}^{S \times m}\).

**Output projection:** \(\mathbf Y = \widehat{\mathbf y}\, \mathbf W_o\), \(\mathbf W_o \in \mathbb{R}^{m \times d}\).

**I/O and state shapes** (rank-2 Zepto graph):

| Tensor / state | Shape | Notes |
|----------------|-------|-------|
| `hidden_states` | \((S, d)\) | Residual stream |
| `output` | \((S, d)\) | Mixer output |
| `conv_state_out` | \((C_c, K)\) | Rolling buffer ([`Conv1DState`](../src/zepto/analysis/horizon/state.py)) |
| `scan_state_out` | \((h, p, N)\) | fp32 accumulation in reference |
| Gate \(\mathbf Z\) | \((S, m)\) | From `in_proj` split; **not** scan input |
| \(\boldsymbol\Delta\) | \((S, h)\) | Raw \(\Delta\); softplus inside scan |

**Numerics (bytes, not FLOPs):** Scan state \(\mathbf H\) accumulates fp32; activations default bf16. Gate \(\mathbf Z\) never enters conv or scan in Nemotron wiring ([`mamba2_mixer.py`](../src/zepto/modules/mamba2_mixer.py)). B/C group broadcast is layout in fused kernels — no extra recurrence MACs.

**Structural vs production:** Identity graph materializes `Split`/`Reshape`/`Transpose` between leaves. HF Hub [`mamba2_split_conv1d_scan_combined`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) may fuse conv + SSD scan + grouped gated norm + `out_proj` when `rmsnorm_weight`, `outproj_weight`, and `norm_before_gate=False` (Nemotron) — §3 prevents double billing.

---

## Section 1: Three-layer costing model

| Layer | Role | This operation's instance |
|-------|------|----------------------------|
| **Reference fused kernel** | Production GPU baseline | `mamba_ssm` Triton SSD (`mamba2_chunk_scan` prefill, `mamba2_selective_state_update` decode) + Dao [`causal_conv1d_fn`](https://github.com/Dao-AILab/causal-conv1d); optional Hub **`mamba2_split_conv1d_scan_combined`** (conv+scan+norm+out_proj mega-path, boundary **B**) on [`NemotronHMamba2Mixer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) |
| **Identity lowering** | Unfused semantic chain | Zepto **`Mamba2Mixer`**: `LinearMatMul` → splits → `DepthwiseCausalConv1d` → splits/reshapes → `SelectiveSSMScan` (S-unrolled at compose) → `GatedGroupedRMSNorm` → `LinearMatMul` — used when `requested_capabilities` lacks `fused` or no region match |
| **Zepto fused region** | Kernel-accurate cost leaf | **`region/mamba2_mixer`** (stub in [`mixer_stub.py`](../src/zepto/analysis/lowering/implementations/regions/mixer_stub.py)) — **composition envelope** equal to Tier-A leaf sum until implementer adds block recipe; mutually exclusive with billing the same module instance’s child regions |

**Default estimates:** Use **kernel-accurate Tier-A fused leaves** summed with §3 mutual-exclusion rules (same as future block envelope). Do **not** sum unfused primitive graph node counts for production leaves.

---

## Section 2: Implementation catalog & taxonomy

| Implementation | Package / access | Fused? | Boundary | Device routing | Training vs inference |
|----------------|------------------|--------|----------|----------------|----------------------|
| **Zepto identity module** | [`Mamba2Mixer`](../src/zepto/modules/mamba2_mixer.py) | No (composed) | Block | Any | Both; emits conv + scan state ports |
| **HF Nemotron H** | [`NemotronHMamba2Mixer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | Partial / mega when Hub returns | Block / **B** | CUDA + Hub | Prefill vs decode branches |
| **HF Mamba2 generic** | [`Mamba2Mixer`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) | Hub fallbacks | Block / **B** | CUDA + Hub | Both |
| **Mamba SSD chunk scan** | [`mamba_ssm` Triton SSD](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py) | Yes (scan sub-leaf) | C | CUDA | Prefill + training |
| **Selective state update** | `mamba2_selective_state_update` | Yes (decode step) | C | CUDA | \(S=1\) decode |
| **Dao causal-conv1d** | [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | Yes (conv sub-leaf) | A | CUDA | Both |
| **Mega-kernel** | `mamba2_split_conv1d_scan_combined` | Yes (conv+scan+norm+out_proj) | **B** | CUDA (Hub) | Training prefill when enabled |
| **Zepto block region (stub)** | `region/mamba2_mixer` | Envelope (TBI) | Block / **B** | `fused` gate | `status: to_implement` |
| **Zepto Tier-A leaves** | `region/linear`, `region/depthwise_causal_conv1d/*`, `region/mamba2_scan/*`, `region/gated_grouped_rms_norm/*` | Yes (per leaf) | A / C | Mixed | Primary costing path today |

---

## Section 3: Fusion boundary & composition rules

**Boundary:** **Mixer block envelope** — one `Mamba2Mixer` forward maps to either (a) **one** `region/mamba2_mixer` leaf, or (b) a **set of Tier-A leaves** below. Optional **boundary B** when Hub mega-kernel subsumes conv + scan + grouped gated norm + `o_proj` on the same layer (do not also bill standalone conv, scan, norm, and output linear on that path).

**Mutually exclusive (same `Mamba2Mixer` instance, same forward):**

- `region/mamba2_mixer` **subsumes** child regions on matched provenance ([`step5` §8.1](../docs/plans/step5-stateful-mixers.md)) — do **not** stack block + leaves.
- `region/mamba2_mixer/hub_mega` (boundary **B**) vs separate `region/depthwise_causal_conv1d/*`, `region/mamba2_scan/*`, `region/gated_grouped_rms_norm/*`, and `o_proj` `region/linear` on the same mixer slot.
- `region/gated_grouped_rms_norm/*` vs `region/gated_rms_norm` or `region/silu` + `region/rmsnorm` on the same norm provenance.

**Composable (default Zepto stack — bill each leaf once):**

1. `region/linear` — `in_proj` (\(2 S d P_{\mathrm{in}}\) FLOPs).
2. `region/depthwise_causal_conv1d/*` — channels \(C_c\), SiLU, \(K\) from config (Nemotron: bias + \(K=4\)).
3. `region/mamba2_scan/*` — prefill SSD vs `region/mamba2_scan/decode` when \(S=1\) + `scan_state_in` / horizon port.
4. `region/gated_grouped_rms_norm/*` — \(n = S m\).
5. `region/linear` — `o_proj` (\(2 S m d\)).
6. **`Split` / `Reshape` / `Transpose`:** 0 FLOPs (structural only).
7. Horizon **`PERSIST`** — `conv_state_out`, `scan_state_out` once at block boundary (gap **G3**).

**Execution constraints:**

- `num_heads % num_groups == 0`; \(m = h p\), \(C_c = m + 2 G N\), \(P_{\mathrm{in}} = m + C_c + h\) from config — never hand-code Nemotron constants without config.
- Decode: all GEMMs scale with \(S=1\); conv/scan use decode variants + state **PERSIST** / **LOAD**.
- Mega-kernel: CUDA Hub + training prefill typical; Nemotron uses `norm_before_gate=False` and grouped `group_size = m/G`.
- `requires_grad=False` → leaf backward FLOPs = 0; inference persists final conv + scan states only.

---

## Section 4: Forward FLOPs — step-by-step derivation

Notation: one multiply-add in GEMM = **2 FLOPs**; elementwise mul/add = **1 FLOP** (Zepto elementwise convention). Symbols: \(d\) hidden, \(h\) heads, \(p\) head dim, \(N\) state size, \(G\) groups, \(m = hp\), \(C_c = m + 2GN\), \(P_{\mathrm{in}} = m + C_c + h\).

Per-output conv element ([`depthwise-causal-conv1d.md`](../docs/kernel/depthwise-causal-conv1d.md)):

\[
f_{\mathrm{conv}} = 2K - 1 + \mathbb{1}_{\mathrm{bias}} + 5\,\mathbb{1}_{\mathrm{SiLU}}.
\]

### 4.1 Identity lowering (Zepto `Mamba2Mixer` — decomposed bill)

#### 4.1.1 Projections (`region/linear`)

| Step | Primitive | Tile | FLOPs rule | Subtotal |
|------|-----------|------|------------|----------|
| 1 | `LinearMatMul` in_proj | \((S,d) \times (d, P_{\mathrm{in}})\) | \(2 S d P_{\mathrm{in}}\) | \(2 S d P_{\mathrm{in}}\) |
| 2 | `LinearMatMul` o_proj | \((S,m) \times (m,d)\) | \(2 S m d\) | \(2 S m d\) |

#### 4.1.2 Depthwise conv + SiLU (`region/depthwise_causal_conv1d`)

| Step | Primitive | FLOPs | Subtotal |
|------|-----------|-------|----------|
| 1 | Depthwise window MAC | \((2K-1) S C_c\) | \((2K-1) S C_c\) |
| 2 | Bias (optional) | \(\mathbb{1}_{\mathrm{bias}} S C_c\) | \(\mathbb{1}_{\mathrm{bias}} S C_c\) |
| 3 | SiLU | \(5 S C_c\) | \(5 S C_c\) |
| **Subtotal** | | | \(f_{\mathrm{conv}} S C_c\) |

Nemotron default (\(K=4\), SiLU, bias): \(f_{\mathrm{conv}} = 13\).

#### 4.1.3 Scan (`region/mamba2_scan`)

Per [`mamba2-scan.md`](../docs/kernel/mamba2-scan.md) §4.1 (dominant term \(5 S h p N\)):

\[
\mathrm{FLOPs}_{\mathrm{scan}} = S h \bigl(5 p N + N + 2 p + 7\bigr).
\]

**Decode:** replace \(S\) with 1.

#### 4.1.4 Gated grouped RMSNorm (`region/gated_grouped_rms_norm`)

\(n = S m\), kernel-accurate fused leaf **\(10n\)** ([`gated-grouped-rms-norm.md`](../docs/kernel/gated-grouped-rms-norm.md)):

\[
\mathrm{FLOPs}_{\mathrm{norm}} = 10 S m.
\]

(`Split`, `Reshape`, `Transpose`: **0 FLOPs**.)

**Identity block total (Tier-A sum):**

\[
\begin{aligned}
\mathrm{FLOPs}_{\mathrm{fwd}}^{\mathrm{identity}}
= {} & 2 S d P_{\mathrm{in}} + 2 S m d + f_{\mathrm{conv}} S C_c \\
& + S h \bigl(5 p N + N + 2 p + 7\bigr) + 10 S m.
\end{aligned}
\]

### 4.2 Fused region leaf (kernel-accurate block envelope)

`region/mamba2_mixer/composed_tier_a` bills the **same closed form** as §4.1 — block fusion does not reduce leading GEMM or scan MACs; it elides intermediate HBM temps (§6). Hub **`region/mamba2_mixer/hub_mega`** (boundary **B**) bundles conv+scan+norm+out_proj execution; **FLOP leaf remains the §4.1 sum** unless a future implementer documents intentional recomputation discounts (none in reference kernels today).

### 4.3 Paper-comparable (if different)

Appendix-style SSM papers emphasize \(\Theta(S h p N)\) scan work and often omit local conv (\(\Theta(S C_c K)\), small \(K\)) and gate+norm (\(\Theta(S m)\)) in headline totals. When citing paper tables, label explicitly; **default Zepto leaf = §4.1 kernel-accurate sum**.

**Closed form (kernel-accurate default):**

\[
\boxed{
\mathrm{FLOPs}_{\mathrm{fwd}} =
2 S d P_{\mathrm{in}} + 2 S m d + f_{\mathrm{conv}} S C_c +
S h \bigl(5 p N + N + 2 p + 7\bigr) + 10 S m
}
\]

**Decode (\(S=1\)):** replace leading factors \(S \to 1\) in each term (projections, conv, scan, norm).

**Arithmetic intensity (numeric example — Nemotron 3.5 Lightning Mamba slice):**

| Symbol | Value |
|--------|-------|
| \(d\) | 2688 |
| \(h,p,N,G\) | 64, 64, 128, 8 |
| \(m\) | 4096 |
| \(C_c\) | 6144 |
| \(P_{\mathrm{in}}\) | 10304 |
| \(S\) | 8192 |
| \(K\), bias, SiLU | 4, yes, yes → \(f_{\mathrm{conv}}=13\) |

| Term | Approx. FLOPs |
|------|----------------|
| in_proj | \(2 \cdot 8192 \cdot 2688 \cdot 10304 \approx 4.55 \times 10^{11}\) |
| o_proj | \(2 \cdot 8192 \cdot 4096 \cdot 2688 \approx 1.81 \times 10^{11}\) |
| conv | \(13 \cdot 8192 \cdot 6144 \approx 6.55 \times 10^{8}\) |
| scan | \(8192 \cdot 64 \cdot (5\cdot64\cdot128 + \cdots) \approx 2.16 \times 10^{10}\) |
| norm | \(10 \cdot 8192 \cdot 4096 \approx 3.36 \times 10^{8}\) |
| **Total** | \(\approx 6.4 \times 10^{11}\) |

Minimum HBM (unfused, one touch each major activation, bf16 \(e=2\)): reads/writes for \(\mathbf X\), `in_proj` output slice, conv in/out, scan I/O, norm I/O, output — order \(\mathcal{O}(S(d + P_{\mathrm{in}} + C_c + m + h + GN))\) bytes; **projections dominate traffic** (\(\approx 2 S P_{\mathrm{in}} e + 2 S m e\) for activations alone \(\approx 340\) MiB at this shape). Intensity \(\approx 6.4 \times 10^{11} / (4 \times 10^{8}) \approx 1.6 \times 10^{3}\) FLOP/byte — **compute-heavy** at Nemotron width (unlike norm-only leaves).

**Apertus-8B cross-check:** Default Apertus stack uses GQA attention, not Mamba-2 ([`mamba2-scan.md`](../docs/kernel/mamba2-scan.md) §4). No Apertus mixer replay; use Nemotron preset ([`mixer_presets.py`](../src/zepto/modules/mixer_presets.py)) for golden block traces.

---

## Section 5: Backward FLOPs — step-by-step derivation

Training context (`requires_grad=True`). Inference-only → **backward FLOPs = 0** at block level.

### 5.1 Saved vs recomputed (block policy)

| Sub-leaf | Typical SAVE | Block note |
|----------|--------------|------------|
| `in_proj` / `o_proj` | activations + weights (GEMM autograd) | standard `region/linear` |
| conv | **`pre_activation`** when SiLU + grad | [`depthwise-causal-conv1d.md`](../docs/kernel/depthwise-causal-conv1d.md) |
| scan | **`state_checkpoint`** \([S,h,p,N]\) fp32 or chunk recompute | [`mamba2-scan.md`](../docs/kernel/mamba2-scan.md) |
| norm | **`group_rstd`** \((S,G,1)\) fp32 | [`gated-grouped-rms-norm.md`](../docs/kernel/gated-grouped-rms-norm.md) |
| Mega-kernel B | fused checkpoint policy | match Hub bwd or conservative SAVE union |

### 5.2 Backward step table (Tier-A sum)

| Stage | Reference | Closed form (dominant) |
|-------|-----------|------------------------|
| in_proj GEMM | 2× forward MAC rule | \(\approx 2 \cdot 2 S d P_{\mathrm{in}} = 4 S d P_{\mathrm{in}}\) |
| o_proj GEMM | 2× forward | \(4 S m d\) |
| conv + SiLU | [`depthwise-causal-conv1d`](../docs/kernel/depthwise-causal-conv1d.md) | \((4K + 6) S C_c\) for SiLU path (\(K=4 \Rightarrow 22 S C_c\) default no-bias table; add bias adjoint if enabled) |
| scan BPTT | [`mamba2-scan`](../docs/kernel/mamba2-scan.md) §5 | \(S h (10 p N + 2 p + N + 4)\) |
| gated grouped RMS | [`gated-grouped-rms-norm`](../docs/kernel/gated-grouped-rms-norm.md) §5 | \(14 S m\) |

**Closed form (Zepto default block envelope, kernel-accurate):**

\[
\boxed{
\begin{aligned}
\mathrm{FLOPs}_{\mathrm{bwd}} = {} &
4 S d P_{\mathrm{in}} + 4 S m d + (4K + 6) S C_c \\
& + S h \bigl(10 p N + 2 p + N + 4\bigr) + 14 S m
\end{aligned}
}
\]

(Adjust conv backward constant when `conv_bias=True` — bias adds \(O(S C_c)\) adjoint; leading term unchanged.)

**Decode (\(S=1\)):** replace \(S\) with 1 in each term.

---

## Section 6: Memory — forward-lived tensors & peak VRAM

Assume bf16 activations (\(e=2\)) unless noted; scan state fp32 (\(e_s=4\)). Block-level view — weights **PERSIST** on module.

### 6.1 Transient allocations (forward)

| Path | Live buffers (identity) | Peak bytes (order) | Elided by block / mega fusion |
|------|-------------------------|--------------------|--------------------------------|
| in_proj | `projected` \((S, P_{\mathrm{in}})\) | \(S P_{\mathrm{in}} e\) | partial: keep splits in registers (mega) |
| conv | extended history, `pre_activation` | \(\mathcal{O}(S C_c e)\) | **`region/depthwise_causal_conv1d`** elides window temps |
| scan | per-step discretization temps | \(\mathcal{O}(S h p N e_s)\) if materialized | **`region/mamba2_scan`** elides per-step HBM |
| norm | `silu_gate`, `gated`, RMS temps | \(\mathcal{O}(S m e)\) | **`region/gated_grouped_rms_norm`** elides |
| o_proj | `gated` / output | \(\mathcal{O}(S m e + S d e)\) | mega may fuse norm→proj |

**Nemotron example (\(S=8192\)):** `in_proj` activation alone \(\approx 8192 \times 10304 \times 2 \approx 159\) MiB — often **dominant forward transient** vs scan checkpoint (\(\approx 1\) GiB **only when training** saves full state — see §6.2).

### 6.2 Saved activations (training backward)

| Path | Saved tensors | Shape | Bytes (bf16 unless noted) | Zepto SAVE policy |
|------|---------------|-------|---------------------------|-------------------|
| conv leaf | `pre_activation` | \((S, C_c)\) | \(S C_c e\) | SAVE when SiLU + grad |
| scan leaf | `state_checkpoint` | \([S,h,p,N]\) fp32 | \(S h p N e_s\) | SAVE (dominant training) |
| norm leaf | `group_rstd` | \((S, G, 1)\) fp32 | \(4 S G\) | SAVE |
| linears | inputs to GEMMs | various | standard autograd | per `region/linear` |

**Inference decode:** **PERSIST** `conv_state` \((C_c, K)\) and `scan_state` \((h,p,N)\) — not full \(S\) checkpoints.

### 6.3 Resource event chains

**Identity lowering (unfused block — conceptual):**

```
PERSIST(W_in, W_o, conv_weight, scan_params, norm_gamma)
ALLOCATE(projected) → SPLIT(z, u_xbc, delta)
ALLOCATE(conv_out) → SAVE(pre_activation) [training]
ALLOCATE(scan_out) → SAVE(state_checkpoint) [training]
ALLOCATE(norm_out) → SAVE(group_rstd) [training]
ALLOCATE(output)
PERSIST(conv_state_out) | PERSIST(scan_state_out)  [horizon]
```

**Fused region leaf (`region/mamba2_mixer/composed_tier_a`):**

```
PERSIST(module_weights...)
ALLOCATE(output)                    # block may stream in_proj → conv → scan → norm → o_proj
ELIDE(projected_splits, conv_window_temps, scan_step_temps, norm_intermediates)
SAVE(pre_activation)                # conv policy, training
SAVE(state_checkpoint)              # scan policy, training
SAVE(group_rstd)                    # norm policy, training
PERSIST(conv_state_out, scan_state_out)
```

**Boundary B (`region/mamba2_mixer/hub_mega`):**

```
PERSIST(...)
ALLOCATE(output)
WORKSPACE(ssd_chunk)                # on-chip / ephemeral
ELIDE(conv_out, scan_out, norm_out as separate HBM peaks when kernel fuses)
SAVE(policy = max(conv, scan, norm) or Hub-documented checkpoints)
PERSIST(conv_state_out, scan_state_out)
```

---

## Section 7: Ecosystem comparison summary

| Implementation | Package | Fused? | Boundary | Fwd FLOP leaf | Bwd FLOP leaf | Memory strategy | Zepto region |
|----------------|---------|--------|----------|---------------|---------------|-----------------|--------------|
| Zepto Tier-A sum | composed leaves | Per leaf | Block | §4.1 boxed | §5.2 boxed | Sum SAVE policies | `region/mamba2_mixer/composed_tier_a` (TBI) |
| HF eager Nemotron | Transformers | Partial | Block | §4.1 | §5.2 | Materialized temps | identity lowering |
| SSD + causal-conv | mamba_ssm + Dao | Sub-leaf | A + C | §4.1 | §5.2 | Elide scan/conv temps | child regions |
| Hub mega | `mamba2_split_conv1d_scan_combined` | Yes | **B** | §4.1 (same MACs) | §5.2 | Fewer HBM peaks | `region/mamba2_mixer/hub_mega` (TBI) |
| Zepto stub | `mixer_stub.py` | — | Block | — | — | — | `region/mamba2_mixer` |

---

## Section 8: Zepto implementation spec

```yaml
region_kind: mamba2_mixer
recommended_region_ids:
  - id: region/mamba2_mixer/composed_tier_a
    variant: default
    hardware_gate: any
    fusion_boundary: block
    status: to_implement
  - id: region/mamba2_mixer/hub_mega
    variant: mamba_ssm
    hardware_gate: cuda
    fusion_boundary: B
    status: to_implement
  - id: region/mamba2_mixer/decode
    variant: decode
    hardware_gate: any
    fusion_boundary: block
    status: to_implement
pattern_rule:
  op_families:
    - LinearMatMul
    - Split
    - DepthwiseCausalConv1d
    - SelectiveSSMScan
    - GatedGroupedRMSNorm
    - Reshape
    - Transpose
recipe:
  forward_flops: >-
    2*S*d*P_in + 2*S*m*d + f_conv(K,bias,silu)*S*C_c +
    S*h*(5*p*N + N + 2*p + 7) + 10*S*m;
    P_in = m + C_c + h; m = h*p; C_c = m + 2*G*N
  backward_flops: >-
    4*S*d*P_in + 4*S*m*d + (4*K + 6)*S*C_c +
    S*h*(10*p*N + 2*p + N + 4) + 14*S*m;
    requires_grad=False -> 0
  elided_temps:
    - projected_splits
    - conv_extended_history
    - conv_per_timestep_window
    - scan_per_step_discretization
    - scan_step_state_hbm
    - norm_silu_gate
    - norm_gated
    - norm_squared
    - norm_normalized
  saved_backward:
    - name: pre_activation
      shape: "(S, C_c)"
    - name: state_checkpoint
      shape: "[S, h, p, N] fp32"
    - name: group_rstd
      shape: "(S, G, 1) fp32"
  resource_events_forward:
    - PERSIST(module_weights)
    - ALLOCATE(output)
    - PERSIST(conv_state_out)
    - PERSIST(scan_state_out)
  resource_events_backward:
    - SAVE(pre_activation)
    - SAVE(state_checkpoint)
    - SAVE(group_rstd)
  numerics_tags:
    - fp32_scan_state
    - silu_before_grouped_rms
    - structural_causal_conv
    - horizon_state_ports
capabilities:
  - fused
priority: 8
variants:
  tier_a_leaves:
    - region/linear
    - region/depthwise_causal_conv1d
    - region/mamba2_scan
    - region/gated_grouped_rms_norm
    - region/linear
  mutual_exclusion:
    - block_subsumes_tier_a_on_Mamba2Mixer_provenance
    - hub_mega_subsumes_conv_scan_norm_o_proj_on_same_layer
```

---

## Section 9: Sources & implementation backlog

### 9.1 Bibliography

- Mamba-2 (structured state space duality): https://arxiv.org/abs/2405.21060
- Nemotron 3.5 Lightning architecture (hybrid Mamba): https://arxiv.org/html/2512.20848v1
- HF Nemotron H Mamba-2 mixer (`Zamba2RMSNormGated`, mega-kernel hook): https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py
- HF Mamba2 Hub fallbacks (`mamba2_split_conv1d_scan_combined`, chunk scan, selective update): https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py
- mamba-ssm Triton SSD implementation: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py
- Dao causal-conv1d CUDA kernels: https://github.com/Dao-AILab/causal-conv1d
- Zepto Step 5 stateful mixers plan (Mamba2Mixer contract): https://github.com/RobinGirardin/zepto/blob/main/docs/plans/step5-stateful-mixers.md
- Child Zepto kernel docs: `docs/kernel/mamba2-scan.md`, `docs/kernel/depthwise-causal-conv1d.md`, `docs/kernel/gated-grouped-rms-norm.md`
- Gated DeltaNet block envelope pattern (analogous composition): `docs/kernel/gated-delta-net.md`

**Verification date:** 2026-09-17

### 9.2 Zepto backlog row

| Priority | Region id | Spec §ref | Boundary | Why |
|----------|-----------|-----------|----------|-----|
| 8 | `region/mamba2_mixer/composed_tier_a` | §8 | Block | Nemotron Mamba layers need one envelope recipe matching Tier-A sum; stub in `mixer_stub.py` blocks accurate horizon costing |
| 8 | `region/mamba2_mixer/hub_mega` | §8 variants | B | Optional `mamba2_split_conv1d_scan_combined` mutual exclusion vs conv+scan+norm+o_proj leaves |
| 8 | `region/mamba2_mixer/decode` | §4 decode | Block | \(S=1\) + conv/scan state ports for prefill/decode horizon tests |

**Open gaps:** Horizon recurrent byte accounting (**G3**) for `conv_state:{layer}` and `scan_state:{layer}` across long decode; `region/mamba2_mixer` recipe dataclass + lowering not yet wired (identity primitives only); conv backward constant with `conv_bias=True` may need explicit bias adjoint term in recipe; batch \(B>1\) rank-3 layouts not in reference module.
