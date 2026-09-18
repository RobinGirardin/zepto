# Implementation plan: Step 5 — Stateful sequence mixers

**Status:** design / ready for implementation  
**Date:** 2026-09-17  
**Scope:** Zepto structural graph (modules + horizon state ports). Fused costing leaves via `/kernel-full` harness (separate PRs per region). No full model builders (Qwen3.8 VLM, Nemotron H full stack), no vision (step 6), no MTP heads (step 7).  
**Authority:** [`docs/model-architecture-gaps-2026-09-14.md`](../model-architecture-gaps-2026-09-14.md) § “5. Stateful sequence mixers”  
**Prerequisites:** Step 1 (`SiLU`, `Softplus`, `Sigmoid`, `SwiGLU`), Step 2 (`FlexibleAttention`, sliding-window masks, output gates), Step 3 (partial/mRoPE — required only for Qwen3.8 **full-attention** sublayers, not for DeltaNet/Mamba mixers themselves), Step 4 (Nemotron MoE blocks — required only for Nemotron **MoE mixer** sublayers).

---

## 1. Goal

Add **stateful sequence mixers** to Zepto so a fresh agent can compose checkpoint-faithful recurrent blocks for:

| Checkpoint | Mixer types | Block shape |
|------------|-------------|-------------|
| **Qwen3.8-27B (language stack)** | Gated DeltaNet (3/4 layers) + gated full GQA (1/4) | `RMSNorm → mixer → residual → RMSNorm → SwiGLU → residual` |
| **Nemotron 3.5 Lightning** | Mamba-2 (23/52), GQA (6/52), MoE (23/52) | `RMSNorm → single mixer → residual` (no FFN in MoE/Mamba blocks) |

After this step, Zepto can:

1. Build **reference (decomposed) structural graphs** for conv + scan mixers using **existing semantic operations only** — no new `Operation` subclasses.
2. Carry **conv rolling buffers** and **recurrent scan state** across prefill/decode horizon steps (parallel to KV cache).
3. Schedule **hybrid decoder blocks** where each layer has exactly one mixer type (Nemotron) or mixer+FFN (Qwen3.8 language).
4. Register **region discovery rules** (`ProvenanceMatchRule`) so fused kernels from `/kernel-full` attach to composed modules.

**Cost modeling requirement:** Until fused regions land, estimates use **identity lowering** on the decomposed primitive graph (same policy as Step 4 MoE before `region/moe/*`). Production costing targets **fused region leaves** billed from Atto-aligned recipes (see §8).

**Out of scope (later steps):**

- Qwen3.8 vision encoder, multimodal RoPE, soft-token scatter — Step 6.
- Full `Qwen3_8Model`, `NemotronHModel` builders — future composition work.
- Chunked DeltaNet / fused CUDA kernels beyond Zepto region recipes — optional kernel follow-ups.
- MTP / drafter heads — Step 7.

**Explicitly out of scope for this step (do not implement):**

- New semantic operations named `L2Normalize`, `DepthwiseConv1D`, `GatedDeltaScan`, `SelectiveSSMScan`, etc.
- Opaque helpers that perform mixer math without recording graph nodes.
- Assuming every layer is `attention + FFN` (`ApertusDecoderBlock` pattern).

---

## 2. Design principles (normative)

These decisions come from review of Steps 1–4 and the module-vs-operation split:

1. **Operations = atomic graph nodes** with shape rules and per-op FLOP/VRAM (`MatMul`, `Gather`, `ReduceSum`, …). **Do not add mixer-specific operations.**
2. **Modules = composition boundaries** that **inline-call** operations with shared `module_kind` provenance (pattern from `SwiGLU`, `FlexibleAttention`).
3. **Decomposed module graphs are reference identity paths.** Callers compose modules; estimators match **fused regions** on provenance when `requested_capabilities={'fused'}` (pattern from `region/swiglu`).
4. **Cross-invocation state** (conv buffer, scan state) is a **horizon concern**, not a new operation family — mirror `KVCacheState` + `kv_state_port_events`.
5. **Per-checkpoint presets** return wired module graphs (`LayerSpec` tuples), not mega-config enums (pattern from `AttentionConfig` presets, `moe_presets.py`).
6. **Fused kernels** are researched and implemented via **`/kernel-full <slug>`** (AGENTS.md harness). This plan defines slugs, region kinds, and registration hooks; kernel PRs can land in parallel after Phase 0–2.

---

## 3. Design summary

```
HorizonSpec (ConvConfig + RecurrentConfig + optional KVConfig)
        │
        ▼
StatePortRegistry ──► InvocationContext.state
        │                    │
        │                    ▼
LayerSpec (per layer)   Module.forward inlines primitives
        │                    │
        ▼                    ▼
HybridDecoderBlock / Qwen35LanguageDecoderBlock
        │
        ├── DepthwiseCausalConv1D (module)
        ├── L2Normalize (module)
        ├── GatedDeltaScan / SelectiveSSMScan (module, S-unrolled)
        ├── GatedRMSNorm / GatedGroupedRMSNorm (module)
        └── LinearMatMul projections
        │
        ▼ (lowering, when fused)
region/* provenance match ──► fused FLOP/VRAM leaf
```

**Naming rule:** mixer families are **module classes + preset factories**, not operation families. Fused costing uses **`region/<kind>`** kinds listed in §8.

---

## 4. Target checkpoints (authoritative constants)

### 4.1 Qwen3.8-27B language (DeltaNet sublayer)

Source: [`docs/model-architecture-gaps-2026-09-14.md`](../model-architecture-gaps-2026-09-14.md), HF [`modeling_qwen3_next.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py).

| Field | Value |
|-------|-------|
| Hidden size `d` | 5120 |
| Language layers `L` | 64 |
| Layer cycle | ×16: `[DeltaNet, DeltaNet, DeltaNet, FullAttn+FFN]` |
| DeltaNet Q/K heads | 16, head dim 128 |
| DeltaNet V heads | 48, head dim 128 |
| Conv kernel | 4, depthwise, SiLU |
| FFN | SwiGLU `5120 → 17408 → 5120` |

Atto FLOP/VRAM reference: [`atto/docs/cost-estimation-framework/30 - attention/34 - gated-deltanet.md`](../../atto/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md) (sibling repo; use for golden derivations).

### 4.2 Nemotron 3.5 Lightning (Mamba-2 sublayer)

Source: gap doc § Nemotron, HF [`modeling_nemotron_h.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py).

| Field | Value |
|-------|-------|
| Hidden size `d` | 2688 |
| Blocks `L` | 52 |
| Mamba heads | 64, head dim 64 |
| SSM state size | 128 |
| Norm groups | 8 |
| Conv kernel | 4, depthwise, SiLU |
| Block types | 23 Mamba, 6 attention, 23 MoE (exact indices in gap doc) |

Mamba-2 scan golden derivations: **author from HF forward + Atto notation doc** (`atto/docs/cost-estimation-framework/10 - notation-and-definitions.md` phase conventions). No Atto page exists yet for Mamba-2 — add `tests/lowering/regions/derivations/mamba2_scan.md` during kernel work.

### 4.3 Mathematical authority, notation, and reading convention

The equations in this plan are structural contracts for the modules, not
alternative algorithms. They are checked against the Hugging Face reference
implementations at commit
[`8e608337`](https://github.com/huggingface/transformers/commit/8e6083374735b8c95f13b1cf2f80cd9c7486c389):

- [`Qwen3NextGatedDeltaNet`](https://github.com/huggingface/transformers/blob/8e6083374735b8c95f13b1cf2f80cd9c7486c389/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
  is authoritative for projection layout, Q/K normalization, decay, recurrence,
  and Qwen gated RMSNorm.
- [`NemotronHMamba2Mixer`](https://github.com/huggingface/transformers/blob/8e6083374735b8c95f13b1cf2f80cd9c7486c389/src/transformers/models/nemotron_h/modeling_nemotron_h.py)
  is authoritative for Mamba projection layout, discretization, grouped B/C
  sharing, recurrent-state shape, and scan/output order.
- [`Zamba2RMSNormGated`](https://github.com/huggingface/transformers/blob/8e6083374735b8c95f13b1cf2f80cd9c7486c389/src/transformers/models/zamba2/modeling_zamba2.py)
  is authoritative for Nemotron's **gate-before-group-normalization** order.
- [`LinearAttentionLayer.update_conv_state`](https://github.com/huggingface/transformers/blob/8e6083374735b8c95f13b1cf2f80cd9c7486c389/src/transformers/cache_utils.py)
  is authoritative for the fixed \(K\)-sample production convolution cache.

Common symbols:

| Symbol | Meaning |
|--------|---------|
| \(B\) | Batch size. Module examples often omit it and show rank-2 \((S,d)\) tensors. |
| \(S\) | Sequence length composed into the static graph. \(S=1\) on ordinary decode steps. |
| \(d\) | Residual-stream hidden width. |
| \(h_k,h_v\) | DeltaNet key/query-head count and value-head count. |
| \(d_k,d_v\) | DeltaNet key/query-head width and value-head width. |
| \(h,p,N,G\) | Mamba head count, channels per head, SSM state size, and B/C sharing-group count. |
| \(K\) | Causal depthwise-convolution kernel width. |
| \(\epsilon\) | Positive numerical stabilizer. |
| \(\odot\) | Elementwise multiplication with ordinary broadcasting. |

Vectors are column vectors in equations. DeltaNet state
\(\mathbf H_t\in\mathbb R^{d_k\times d_v}\), so its read is
\(\mathbf q_t^\top\mathbf H_t\in\mathbb R^{d_v}\); writing
\(\mathbf H_t\mathbf q_t\) would have the wrong orientation. Mamba state is a
matrix per head,
\(\mathbf H_{t,i}\in\mathbb R^{p\times N}\), not one length-\(N\) vector per
head.

Zepto records shapes and operation dependencies, not numerical values.
Accordingly, a `forward()` method must translate each equation into visible
semantic nodes. `Reshape`/`Transpose` express index reinterpretation and cost
zero FLOPs; `Multiply`/`Add` express elementwise terms; `MatMul` expresses
contractions and outer products; `ReduceSum` expresses reductions. Parameters
such as \(\epsilon\), inverse dimensions, indices, and zero initial states are
shape-bearing graph tensors where no dedicated scalar-literal mechanism exists.

Registered `Parameter` objects cannot be passed through generic data-input ops
such as `Exp` or `Multiply`; current parameter-aware primitives are
`LinearMatMul`, `ParameterScale`, and `ParameterBias`. The reference
parameterization \(A_{\log}\mapsto\exp(A_{\log})\) must therefore be represented
as a positive transformed parameter
\(\lambda=\exp(A_{\log})\), loaded/initialized from the checkpoint once, then
used through `ParameterScale`. This preserves the exact forward equation and
parameter count, but does not model the tiny \(\exp(A_{\log})\) training VJP in
the identity graph. Fused training recipes must either include that VJP or
document it as an omitted \(O(h)\) term.

`Module.__call__` currently accepts only positional `Tensor` inputs. Optional
state, mask, and RoPE tensors in the APIs below are therefore positional
optional arguments even when their semantic role reads like a keyword.
Configuration flags remain constructor arguments. Do not add keyword-only
`forward()` inputs unless `Module.__call__` is generalized in a separate,
explicit prerequisite.

Unless a config explicitly adds batch-aware state accounting, horizon byte
formulas follow the existing KV convention and represent \(B=1\). Module
tensors themselves should support both rank-2 and rank-3 inputs where the
existing primitives permit it.

---

## 5. Implementation phases (strict order)

Implement phases sequentially. Do not start phase N+1 until phase N tests pass.

| Phase | Deliverable | Blocks |
|-------|-------------|--------|
| **0** | Read this plan + existing horizon/KV tests | everything |
| **1** | Horizon state contracts (conv + recurrent ports) | modules, lowering |
| **2** | Leaf modules: `L2Normalize`, `GatedRMSNorm`, `GatedGroupedRMSNorm` | block modules |
| **3** | `DepthwiseCausalConv1D` module + conv state I/O | DeltaNet, Mamba |
| **4** | `GatedDeltaScan` module | `GatedDeltaNet` |
| **5** | `SelectiveSSMScan` module | `Mamba2Mixer` |
| **6** | Block modules: `GatedDeltaNet`, `Mamba2Mixer` | presets |
| **7** | `LayerSpec` + hybrid decoder blocks + presets | integration |
| **8** | Region provenance rules + lowering registration stubs | fused costing |
| **9** | Integration + horizon tests | merge |
| **10** | Doc touch-up + gap doc link | merge |

**Parallel track (kernel harness, not gated on Phase 9):** run `/kernel-full` per §8.3 after Phase 2 (leaf modules exist for reference graphs).

---

## 6. File inventory

### 6.1 New files — horizon & helpers

| Path | Purpose |
|------|---------|
| `src/zepto/analysis/horizon/conv.py` | `ConvStateConfig`, `Conv1DState` dataclass (or colocate in `state.py`) |
| `src/zepto/analysis/horizon/recurrent.py` | `RecurrentStateConfig`, `RecurrentScanState` dataclass (or colocate in `state.py`) |

### 6.2 New files — modules

| Path | `module_kind` | Purpose |
|------|---------------|---------|
| `src/zepto/modules/l2_normalize.py` | `L2Normalize` | Last-dim ℓ₂ normalize |
| `src/zepto/modules/gated_rms_norm.py` | `GatedRMSNorm` | Per-head RMSNorm, then SiLU gate (Qwen DeltaNet output) |
| `src/zepto/modules/gated_grouped_rms_norm.py` | `GatedGroupedRMSNorm` | SiLU gate, then grouped RMSNorm (Mamba) |
| `src/zepto/modules/depthwise_causal_conv1d.py` | `DepthwiseCausalConv1d` | Depthwise causal conv K=4 (+ optional SiLU) |
| `src/zepto/modules/gated_delta_scan.py` | `GatedDeltaScan` | Gated DeltaNet recurrence (S-unrolled) |
| `src/zepto/modules/selective_ssm_scan.py` | `SelectiveSSMScan` | Mamba-2 selective scan (S-unrolled) |
| `src/zepto/modules/gated_delta_net.py` | `GatedDeltaNet` | Full Qwen3.5 DeltaNet mixer |
| `src/zepto/modules/mamba2_mixer.py` | `Mamba2Mixer` | Full Nemotron Mamba-2 mixer |
| `src/zepto/modules/mixer_config.py` | — | `GatedDeltaNetConfig`, `Mamba2MixerConfig` dataclasses |
| `src/zepto/modules/layer_spec.py` | — | `LayerSpec`, `MixerKind` literals, validation |
| `src/zepto/modules/hybrid_decoder_block.py` | `HybridDecoderBlock` | Single-mixer Nemotron block |
| `src/zepto/modules/qwen35_language_decoder_block.py` | `Qwen35LanguageDecoderBlock` | Mixer + SwiGLU (Qwen language) |
| `src/zepto/modules/mixer_presets.py` | — | `qwen35_language_layer_specs()`, `nemotron_h_layer_specs()` |

### 6.3 New files — lowering (registration only in Step 5; variants from kernel-full)

| Path | Purpose |
|------|---------|
| `src/zepto/analysis/lowering/implementations/regions/l2_normalize/__init__.py` | Export region impl tuple |
| `src/zepto/analysis/lowering/implementations/regions/l2_normalize/rules.py` | `ProvenanceMatchRule` |
| `src/zepto/analysis/lowering/implementations/regions/l2_normalize/variants.py` | Fused variant(s) — stub raises until kernel-full |
| `src/zepto/analysis/lowering/recipes/l2_normalize.py` | `L2NormalizeRecipe` dataclass |
| *(repeat same layout for each region in §8.1)* | |
| `src/zepto/analysis/lowering/implementations/regions/gated_delta_net/rules.py` | Block-level provenance |
| `src/zepto/analysis/lowering/implementations/regions/mamba2_mixer/rules.py` | Block-level provenance |

### 6.4 New files — tests

| Path | Purpose |
|------|---------|
| `tests/horizon/test_conv_recurrent_state_ports.py` | State registry advance + byte accounting |
| `tests/modules/test_l2_normalize.py` | Compose + shape checks |
| `tests/modules/test_gated_rms_norm.py` | Compose |
| `tests/modules/test_gated_grouped_rms_norm.py` | Compose |
| `tests/modules/test_depthwise_causal_conv1d.py` | Prefill vs decode state tensors |
| `tests/modules/test_gated_delta_scan.py` | Small-S unroll graph node count |
| `tests/modules/test_selective_ssm_scan.py` | Small-S unroll |
| `tests/modules/test_gated_delta_net.py` | End-to-end mixer compose |
| `tests/modules/test_mamba2_mixer.py` | End-to-end mixer compose |
| `tests/modules/test_layer_spec.py` | Preset validation (52 Nemotron entries, 64 Qwen entries) |
| `tests/modules/test_hybrid_decoder_block.py` | Nemotron block variants compose |
| `tests/integration/test_gated_delta_prefill_decode_horizon.py` | Horizon prefill→decode |
| `tests/integration/test_mamba2_prefill_decode_horizon.py` | Horizon prefill→decode |
| `tests/lowering/regions/derivations/gated_delta_scan.md` | Golden FLOP/VRAM (Atto-aligned) |
| `tests/lowering/regions/derivations/mamba2_scan.md` | Golden FLOP/VRAM |
| `tests/lowering/regions/test_l2_normalize.py` | Region discovery (skip if no variant) |
| *(one test file per §8.1 region after kernel-full)* | |

### 6.5 Modified files

| Path | Change |
|------|--------|
| `src/zepto/analysis/horizon/state.py` | `Conv1DState`, `RecurrentScanState`, registry `configure_conv`, `configure_recurrent`, `advance()` |
| `src/zepto/analysis/horizon/spec.py` | `ConvStateConfig`, `RecurrentStateConfig`; extend `HorizonSpec.build_state_registry()` |
| `src/zepto/analysis/horizon/__init__.py` | Export new config types |
| `src/zepto/analysis/__init__.py` | Re-export if KV types are exported today |
| `src/zepto/analysis/lowering/helpers.py` | `conv_state_port_events`, `recurrent_state_port_events`, scenario resolvers |
| `src/zepto/analysis/lowering/implementations/regions/__init__.py` | Register new region tuples |
| `src/zepto/analysis/lowering/recipes/__init__.py` | Export new recipe dataclasses |
| `src/zepto/modules/__init__.py` | Public exports |
| `src/zepto/__init__.py` | Re-export if applicable |
| `docs/model-architecture-gaps-2026-09-14.md` | Link to this plan under §5 |

### 6.6 Unchanged (do not edit unless a test forces it)

- `src/zepto/modules/apertus.py`, `apertus_decoder_block.py` — Apertus regression must pass.
- `src/zepto/semantic/operations/*` — **no new operation files** in Step 5.
- Existing `region/gqa/*` — attention blocks reuse Step 2 paths.
- Existing MoE modules — Nemotron MoE blocks reuse Step 4 paths.

---

## 7. Phase specifications

### Phase 1 — Horizon state contracts

**Goal:** Extend cross-step state beyond KV cache so conv and scan mixers have accounted persistent bytes on decode timelines.

#### 7.1 Add dataclasses to `src/zepto/analysis/horizon/state.py`

```python
@dataclass(frozen=True, slots=True)
class Conv1DState:
    """Rolling depthwise conv buffer for one layer."""

    layer_index: int
    channels: int
    kernel_size: int
    dtype: DType

    @property
    def bytes(self) -> int:
        # Reference fused update keeps a fixed K-sample in-place buffer.
        itemsize = self.dtype.itemsize or 0
        return self.channels * self.kernel_size * itemsize


@dataclass(frozen=True, slots=True)
class RecurrentScanState:
    """Recurrent scan hidden state for one layer (batch size one)."""

    layer_index: int
    kind: Literal["gated_delta", "mamba2"]
    num_heads: int
    head_dim: int
    state_dim: int  # d_v for DeltaNet; d_state for Mamba
    dtype: DType

    @property
    def bytes(self) -> int:
        itemsize = self.dtype.itemsize or 0
        if self.kind == "gated_delta":
            # one matrix per head: d_k × d_v; head_dim is d_k, state_dim is d_v
            return self.num_heads * self.head_dim * self.state_dim * itemsize
        # mamba2: one p × N matrix per head; head_dim is p, state_dim is N
        return self.num_heads * self.head_dim * self.state_dim * itemsize
```

The byte formulas follow directly from the recurrent tensors carried between
calls:

\[
\begin{aligned}
\text{DeltaNet state shape} &= (h_v,d_k,d_v), &
\text{bytes} &= h_vd_kd_v\,e,\\
\text{Mamba-2 state shape} &= (h,p,N), &
\text{bytes} &= hpN\,e,
\end{aligned}
\]

where \(e\) is dtype item size. For the Nemotron checkpoint this is
\(64\times64\times128\) elements per Mamba layer. Omitting \(p\) would
undercount recurrent state by \(64\times\). A causal convolution mathematically
needs only \(K-1\) past samples, but the reference fused update keeps a
fixed-address \(K\)-sample buffer, shifts it, inserts the new sample, and
evaluates the updated window. For runtime-faithful accounting, use shape
\((C,K)\) and bytes \(CKe\), not the smaller theoretical minimum.

Extend `StateSnapshot` usage via existing `custom: tuple[tuple[str, object], ...]` — keys:

- `conv_state:{layer_index}` → `Conv1DState`
- `scan_state:{layer_index}` → `RecurrentScanState`

#### 7.2 Extend `StatePortRegistry`

Add templates + configure methods (mirror `configure_kv`):

```python
def configure_conv_layers(
    self,
    *,
    layer_indices: tuple[int, ...],
    channels: int,
    kernel_size: int,
    dtype: DType,
) -> StatePortRegistry: ...

def configure_recurrent_layers(
    self,
    *,
    specs: tuple[tuple[int, str, int, int, int], ...],
    # (layer_index, kind, num_heads, head_dim, state_dim)
    dtype: DType,
) -> StatePortRegistry: ...
```

Extend `advance()`:

| Step kind | Conv behavior | Recurrent behavior |
|-----------|---------------|-------------------|
| `PREFILL` | Allocate `Conv1DState` per configured layer at `seq_len` boundary (reference buffer holds last K samples, left-padded if needed) | If training context: no single-state persist; scan states sized for full S handled inside module/region. If inference: allocate final `RecurrentScanState` after prefill. |
| `DECODE` | Persist rolling buffer; bytes constant K×C | Persist one scan state per layer; bytes constant |

**Training vs inference contract:** Module composition always builds the full S recurrence internally. Horizon recurrent ports for **inference decode** persist **one** state per layer. Training peak VRAM for all S scan snapshots is modeled by **fused region recipes** (§8.2), not by horizon ports.

#### 7.3 Extend `src/zepto/analysis/horizon/spec.py`

```python
@dataclass(frozen=True, slots=True)
class ConvStateConfig:
    layer_indices: tuple[int, ...]
    channels: int
    kernel_size: int
    dtype: DType

@dataclass(frozen=True, slots=True)
class RecurrentStateConfig:
    layers: tuple[tuple[int, str, int, int, int], ...]
    dtype: DType
```

Add fields to `HorizonSpec`:

```python
conv: ConvStateConfig | None = None
recurrent: RecurrentStateConfig | None = None
```

Wire in `build_state_registry()`.

#### 7.4 Lowering helpers (`src/zepto/analysis/lowering/helpers.py`)

Add (mirror `kv_state_port_events` / `KVScenario`):

```python
@dataclass(frozen=True, slots=True)
class ConvScenario:
    layer_index: int | None
    channels: int
    kernel_size: int

@dataclass(frozen=True, slots=True)
class RecurrentScenario:
    layer_index: int | None
    kind: Literal["gated_delta", "mamba2"]
    num_heads: int
    head_dim: int
    state_dim: int

def conv_state_port_events(...) -> tuple[StatePortEvent, ResourceEvent, ResourceEvent, str]: ...
def recurrent_state_port_events(...) -> tuple[StatePortEvent, ResourceEvent, ResourceEvent, str]: ...
def resolve_conv_scenario(context, module_path) -> ConvScenario | None: ...
def resolve_recurrent_scenario(context, module_path) -> RecurrentScenario | None: ...
```

#### 7.5 Tests — `tests/horizon/test_conv_recurrent_state_ports.py`

- Build `HorizonSpec.inference(prefill=64, decode_steps=4, conv=..., recurrent=...)`.
- Assert `state_final.custom` contains expected entries.
- Assert conv bytes == `C * K * itemsize`.
- Assert DeltaNet scan bytes == `h * d_k * d_v * itemsize` per layer.
- Assert Mamba scan bytes == `h * head_dim * state_size * itemsize` per layer.

---

### Phase 2 — Normalization leaf modules (pure primitive decomposition)

**No new operations.** Reuse `Multiply`, `ReduceSum`, `Divide`, `SquareRoot`, `Add`, `Cast`, `ParameterScale`, `Sigmoid`, `SiLU` module or inline SiLU ops.

#### 7.6 `src/zepto/modules/l2_normalize.py`

**What it is.** L2 normalization projects each feature vector onto the unit
sphere (up to \(\epsilon\)). It removes vector magnitude while preserving
direction. DeltaNet applies it to each Q/K head because the delta-rule update
uses \(\mathbf k_t\mathbf k_t^\top\); bounding
\(\lVert\mathbf k_t\rVert_2\) prevents that rank-one correction from growing
with arbitrary projection scale.

For each last-axis vector \(\mathbf x\in\mathbb R^n\):

\[
\operatorname{L2Norm}_\epsilon(\mathbf x)
=
\frac{\mathbf x}{\sqrt{\sum_{i=1}^{n}x_i^2+\epsilon}}.
\]

This is intentionally **not RMSNorm**. RMSNorm divides the sum of squares by
\(n\), whereas the FLA-compatible Qwen reference uses the un-averaged sum.
After L2 normalization, Qwen additionally scales queries by \(d_k^{-1/2}\)
inside the DeltaNet path; that query scaling is not part of this module.

```python
class L2Normalize(Module):
    module_kind = "L2Normalize"

    def forward(self, value: Tensor) -> Tensor:
        # axis = last dim
        # squared = Multiply(x, x)
        # norm_sq = ReduceSum(squared, keepdim=True)  # SUM, not mean
        # norm = SquareRoot(norm_sq + eps)   # Tensor semantic_type="epsilon"
        # return Divide(x, norm)
```

`__init__(eps=1e-6, compute_dtype=FP32)` stores the numerical policy and the
epsilon tensor. `forward()` validates a ranked input, casts to fp32, implements
the equation in the same order as the reference, and restores the activation
dtype. The cast matters because the reference explicitly normalizes Q/K in
fp32. Do not use `_inv_norm_size`; that helper belongs to RMSNorm's mean-square
calculation.

#### 7.7 `src/zepto/modules/gated_rms_norm.py`

**What it is.** Qwen's scan produces a value vector per value head and a
separate projected gate \(\mathbf z\). It normalizes each \(d_v\)-wide head
before modulating it with a SiLU gate:

\[
\begin{aligned}
\operatorname{rms}_\epsilon(\mathbf y)
&=\sqrt{\frac{1}{d_v}\sum_{j=1}^{d_v}y_j^2+\epsilon},\\
\operatorname{GatedRMS}(\mathbf y,\mathbf z)
&=\boldsymbol\gamma\odot
  \frac{\mathbf y}{\operatorname{rms}_\epsilon(\mathbf y)}
  \odot\operatorname{SiLU}(\mathbf z),\\
\operatorname{SiLU}(z)&=z\,\sigma(z).
\end{aligned}
\]

The order is **normalize, apply learned scale, then gate**. A raw
`Multiply(normalized, gate)` is not checkpoint-faithful because the reference
applies SiLU to the gate.

```python
class GatedRMSNorm(Module):
    module_kind = "GatedRMSNorm"

    def __init__(self, normalized_shape: int, *, eps: float = 1e-6) -> None: ...

    def forward(self, value: Tensor, gate: Tensor) -> Tensor:
        # Inline RMSNorm arithmetic (do NOT call self.rms_norm submodule — splits provenance)
        # silu_gate = Multiply(gate_fp32, Sigmoid(gate_fp32))
        # return Cast(Multiply(ParameterScale(normalized), silu_gate), input_dtype)
```

`__init__()` owns \(\boldsymbol\gamma\in\mathbb R^{d_v}\), epsilon, and the
inverse feature-size scalar. `forward()` checks equal `value`/`gate` shapes,
accumulates the RMS path in fp32, restores the value dtype at the same boundary
as the reference, and inlines `Sigmoid` + `Multiply` so every arithmetic node
retains `GatedRMSNorm` provenance.

#### 7.8 `src/zepto/modules/gated_grouped_rms_norm.py`

**What it is.** Nemotron first gates the Mamba scan output, then partitions the
\(m=h\,p\) channels into \(G\) contiguous groups and independently normalizes
each group. Let \(r=m/G\), \(\mathbf y,\mathbf z\in\mathbb R^m\), and
\(\mathbf u=\mathbf y\odot\operatorname{SiLU}(\mathbf z)\). For group
\(\mathbf u^{(g)}\in\mathbb R^r\):

\[
\begin{aligned}
\mathbf u &= \mathbf y\odot\operatorname{SiLU}(\mathbf z),\\
\widehat{\mathbf u}^{(g)}
&=\frac{\mathbf u^{(g)}}{
  \sqrt{\frac{1}{r}\sum_{j=1}^{r}(u_j^{(g)})^2+\epsilon}},\\
\operatorname{GroupedGatedRMS}(\mathbf y,\mathbf z)
&=\boldsymbol\gamma\odot
  \operatorname{concat}_{g=1}^{G}\widehat{\mathbf u}^{(g)}.
\end{aligned}
\]

Gate placement is material: Nemotron/Zamba gates **before** grouped
normalization. Moving the gate after normalization changes both each group's
denominator and the function represented.

```python
class GatedGroupedRMSNorm(Module):
    module_kind = "GatedGroupedRMSNorm"

    def __init__(self, hidden_size: int, num_groups: int, *, eps: float = 1e-6) -> None: ...

    def forward(self, value: Tensor, gate: Tensor) -> Tensor:
        # SiLU on gate inline (Sigmoid + Multiply)
        # gated = Multiply(value_fp32, silu_gate)
        # Reshape (S, d) -> (S, num_groups, d // num_groups)
        # Group RMSNorm over the gated values (inline)
        # Reshape back
        # return ParameterScale(Cast(normalized, input_dtype))
```

`__init__()` requires `hidden_size % num_groups == 0` and owns one scale
parameter per channel. Here `hidden_size` means the Mamba intermediate width
\(m=h\,p\), **not** the residual width \(d\). Nemotron therefore uses
`hidden_size=64*64=4096`, `num_groups=8`, and group width \(512\); its residual
width remains 2688. `forward()` uses `Reshape` views to expose the group axis,
reduces only the final group-width axis, then reshapes back.

#### 7.9 Tests

- Compose each module with small shapes `(S=4, d=32)`.
- Assert graph contains only existing op families (grep node provenance in compose test).
- Assert `L2Normalize` has no divide-by-feature-size node/tensor.
- Assert Qwen gated RMS provenance orders norm before SiLU gate multiplication.
- Assert grouped gated RMS provenance orders SiLU multiplication before
  `Reshape → ReduceSum`, and rejects non-divisible group widths.

---

### Phase 3 — `DepthwiseCausalConv1d` module

**What it is.** A depthwise convolution is one independent finite-impulse
response filter per channel; unlike an ordinary convolution, it never mixes
channels. Causality means output \(t\) may depend only on samples at or before
\(t\). With zero-extension \(x_{u,c}=0\) for \(u<0\), PyTorch's
cross-correlation convention is:

\[
\begin{aligned}
a_{t,c}
&= b_c + \sum_{j=0}^{K-1} w_{c,j}\,
   x_{t-K+1+j,c},\\
y_{t,c}&=\phi(a_{t,c}),
\qquad \phi\in\{\operatorname{id},\operatorname{SiLU}\}.
\end{aligned}
\]

There is no sum over channels. Conceptually
\(W\in\mathbb R^{C\times K}\) therefore encodes exactly \(C\) independent
filters. Qwen uses no convolution bias; Nemotron uses the checkpoint's
`use_conv_bias` policy, so the common module must support both.

**Why the persisted state is \(K\), despite a \(K-1\) minimum.** At the next
invocation, \(K-1\) past samples plus the new sample are mathematically
sufficient. The reference `causal_conv1d_update`, however, maintains a
fixed-address buffer
\(\mathbf R_t=[\mathbf x_{t-K+1},\ldots,\mathbf x_t]\) of shape \((C,K)\).
Decode shifts this buffer, inserts \(\mathbf x_{t+1}\), and evaluates the
updated \(K\)-sample window in place. Zepto models this production layout;
the oldest pre-update sample is the one-storage-step overhead.

**Decomposition strategy (static \(K\), static \(S\)):** Avoid `nn.Conv1d`.
For each output timestep `t`, gather the causal window, multiply by depthwise
weight, and reduce over \(K\).

Because Zepto composes static graphs with fixed `S`:

1. Normalize to a sequence-leading view. Prefill concatenates an explicit zero
   boundary of \(K-1\) samples; decode transposes the \((C,K)\) state to
   \((K,C)\) and concatenates the new input.
2. Store \(W\) as \(K\) registered lag vectors
   \(\mathbf w_j=W[:,j]\in\mathbb R^C\). Split the sequence-leading history
   into one-sample tensors with `Split(sizes=(1,) * history_len)`; for each
   output `t`, select the needed \(K\) Python tuple entries and apply
   `ParameterScale(weight_j)`.
3. Add the \(K\) scaled channel vectors, apply optional `ParameterBias`, inline
   SiLU when configured, reshape to \((1,C)\), and concatenate outputs along
   the sequence axis.
4. Form `conv_state_out` from the last \(K\) **raw input** samples, left-padding
   when the available prefix is shorter. Activation outputs are not the state.

**Module API:**

```python
class DepthwiseCausalConv1d(Module):
    module_kind = "DepthwiseCausalConv1d"

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        *,
        activation: Literal["none", "silu"] = "silu",
        bias: bool = False,
    ) -> None: ...

    def forward(
        self,
        value: Tensor,
        conv_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (output, conv_state_out).

        Prefill: conv_state_in is None; conv_state_out holds last K raw samples per channel.
        Decode (S=1): conv_state_in required; output length 1.
        """
```

`__init__()` validates positive \(C,K\), registers `weight_0` through
`weight_{K-1}` as \(K\) `Parameter(shape=(channels,))` vectors, and optionally
owns a channel bias. This has the same \(CK\) learned scalars as checkpoint
weight \((C,1,K)\); a future weight loader transposes/slices that tensor into
lag vectors. The representation is required because generic `Multiply` cannot
bind a registered parameter and `ParameterScale` intentionally accepts only a
1-D parameter matching the activation's last dimension.

`forward()` validates `value.shape[-1] == C` and, when state is supplied,
`conv_state_in.shape == (C,K)` (plus matching batch prefix in rank-3 form).
The loop is a graph-construction loop, not runtime Python execution in the
modeled kernel: each iteration emits visible
`Split → ParameterScale → Add` dependencies. Python tuple indexing selects
already-recorded graph tensors and is not hidden runtime tensor indexing.

Current `Split` only splits the leading dimension. Any implementation that
needs a last-axis split must transpose that axis to the front, call `Split`,
then transpose back. Do not call `Split` as though it accepted `dim=-1`.

**Important:** Inline all ops under `DepthwiseCausalConv1d` provenance (SwiGLU pattern). Do not delegate to a `SiLU` submodule.

#### Tests — `tests/modules/test_depthwise_causal_conv1d.py`

- Prefill `S=8, C=16`: output shape `(8, 16)`, state shape `(16, K)`.
- Decode `S=1` with carried state: output shape `(1, 16)`.
- Assert Qwen construction has no bias parameter; Nemotron construction does.
- Assert `conv_state_out` is gathered from the pre-activation input history.

---

### Phase 4 — `GatedDeltaScan` module

**What it is.** DeltaNet is recurrent linear attention. Each head stores an
associative matrix \(\mathbf H_t\) that maps a key-like vector to its current
value prediction. A new token computes the state's prediction at
\(\mathbf k_t\), writes only the prediction error
\(\mathbf v_t-\mathbf k_t^\top\widetilde{\mathbf H}_t\), and reads the updated
state at \(\mathbf q_t\). This is the neural analogue of an online delta-rule
update.

For one value head, after Q/K L2 normalization and query scaling:

\[
\begin{aligned}
\widetilde{\mathbf H}_t
  &= \alpha_t\mathbf H_{t-1},
  &&\alpha_t=\exp(g_t)\in(0,1],\\
\widehat{\mathbf v}_t
  &= \mathbf k_t^\top\widetilde{\mathbf H}_t,\\
\boldsymbol\delta_t
  &= \beta_t(\mathbf v_t-\widehat{\mathbf v}_t),
  &&\beta_t=\sigma(b_t)\in(0,1),\\
\mathbf H_t
  &= \widetilde{\mathbf H}_t+
     \mathbf k_t\boldsymbol\delta_t^\top,\\
\mathbf o_t
  &= \mathbf q_t^\top\mathbf H_t.
\end{aligned}
\]

Equivalently,
\(\mathbf H_t=\alpha_t(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)
\,\mathbf H_{t-1}+\beta_t\mathbf k_t\mathbf v_t^\top\). The staged form above
matches the reference code and is preferable for the decomposed graph because
it exposes one state read, one error calculation, one rank-one write, and one
query read.

The scan's Q/K/V inputs are already convolved, Q/K are already L2-normalized,
and Q is already multiplied by \(d_k^{-1/2}\). It receives
\(\beta\) and log-decay \(g\), not a free \(\alpha\):

\[
g_t=-\exp(A_{\log})\operatorname{softplus}(a_t+\mathrm{dt\_bias}),
\qquad \alpha_t=\exp(g_t).
\]

Representing \(g\le 0\) in log space makes the product of many decays a sum and
guarantees a non-expanding \(\alpha\). `GatedDeltaNet` computes \(g\) and
\(\beta\); `GatedDeltaScan` applies `Exp` at each recurrent step.

```python
class GatedDeltaScan(Module):
    module_kind = "GatedDeltaScan"

    def forward(
        self,
        query: Tensor,       # (S, h_v, d_k), after Q/K head expansion
        key: Tensor,
        value: Tensor,       # (S, h_v, d_v)
        beta: Tensor,        # (S, h_v)
        log_decay: Tensor,   # (S, h_v), values g <= 0
        scan_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (output, scan_state_out)."""
```

After head expansion, `query` and `key` also have shape
\((S,h_v,d_k)\); the scan itself requires equal head counts. Its initial/final
state is \((h_v,d_k,d_v)\) for \(B=1\), and output is
\((S,h_v,d_v)\). Qwen's \(h_k=16,h_v=48\) mismatch is resolved in the parent
module by repeating every Q/K head \(h_v/h_k=3\) times, exactly where the
reference does it.

**`forward()` mapping:** Python `for t in range(S)` unrolls the following at
compose time:

1. Initialize `state_prev` from an explicit zero tensor or `scan_state_in`.
2. `decayed = Multiply(state_prev, Exp(log_decay_t))` implements
   \(\widetilde{\mathbf H}_t\).
3. `MatMul(k_t^T, decayed)` implements
   \(\widehat{\mathbf v}_t\).
4. `Multiply(beta_t, Subtract(v_t, prediction))` implements
   \(\boldsymbol\delta_t\).
5. Reshape \(k_t\) and \(\delta_t\) to column/row matrices; their `MatMul`
   is the outer product \(\mathbf k_t\boldsymbol\delta_t^\top\).
6. Add the outer product to `decayed`, then
   `MatMul(q_t^T, state)` for \(\mathbf o_t\).
7. Concatenate outputs along \(S\); return the final state directly.

`__init__(num_heads, key_head_dim, value_head_dim)` stores shape invariants,
not learned weights. `forward()` validates all sequence/head dimensions and
the optional state before emitting nodes. Keep the recurrent state in fp32,
matching the reference scan's accumulation, and cast only the emitted sequence
output back to the activation dtype.

#### Tests

- Small `S=3, h=2, d_k=d_v=4`: assert graph builds; count `MatMul` nodes scales with S.
- Assert output `(S,h,d_v)` and final state `(h,d_k,d_v)`.
- Assert each timestep contains `Exp(g)`, key-state prediction, rank-one
  update, and query-state read in provenance order.
- Reject unequal Q/K/V head counts at the scan boundary; test expansion in
  `GatedDeltaNet` instead.

---

### Phase 5 — `SelectiveSSMScan` module

**What it is.** A state-space model treats each channel as a small dynamical
system. The continuous-time template
\(\dot{\mathbf h}=A\mathbf h+B\mathbf x\),
\(\mathbf y=C\mathbf h+D\mathbf x\) is discretized at a learned,
input-dependent step size \(\overline\Delta_t\). Mamba is *selective* because
\(\Delta_t,B_t,C_t\) depend on the current token; \(A,D\) remain learned but
input-independent.

Nemotron uses one state matrix per head,
\(\mathbf H_{t,i}\in\mathbb R^{p\times N}\). The first axis corresponds to
the \(p\) channels in that head and the second to \(N\) SSM coordinates. With
head \(i\), channel \(r\), SSM coordinate \(n\), and sharing-group
\(\gamma(i)=\lfloor i/(h/G)\rfloor\):

\[
\begin{aligned}
A_i &= -\exp(A_{\log,i}) < 0,\\
\overline\Delta_{t,i}
  &= \operatorname{softplus}(\Delta_{t,i}+\mathrm{dt\_bias}_i)>0,\\
\overline A_{t,i,r,n}
  &= \exp(\overline\Delta_{t,i}A_i),\\
\overline B_{t,i,n}
  &= \overline\Delta_{t,i}B_{t,\gamma(i),n},\\
H_{t,i,r,n}
  &= \overline A_{t,i,r,n}H_{t-1,i,r,n}
     + x_{t,i,r}\overline B_{t,i,n},\\
y_{t,i,r}
  &= \sum_{n=1}^{N}H_{t,i,r,n}C_{t,\gamma(i),n}
     +D_i x_{t,i,r}.
\end{aligned}
\]

Negative \(A_i\) and positive \(\overline\Delta\) imply
\(0<\overline A\le1\), so old state decays rather than explodes. B/C are
produced once per group and repeated conceptually across the \(h/G\) heads in
that group. They are not repeated across head channels: broadcasting
\(x_{t,i,r}\overline B_{t,i,n}\) creates the \(p\times N\) state update.

The optional Mamba kernel argument `z` is `None` in Nemotron. Gating therefore
does **not** belong in this scan; `GatedGroupedRMSNorm` applies the projected
gate after scan output and before normalization.

```python
class SelectiveSSMScan(Module):
    module_kind = "SelectiveSSMScan"

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        state_size: int,
        num_groups: int,
    ) -> None: ...
        # owns decay_rate=exp(A_log): (h,), D: (h,), dt_bias: (h,)

    def forward(
        self,
        x: Tensor,           # (S, h, p)
        delta: Tensor,       # raw Δ, (S, h)
        B: Tensor,           # (S, G, N)
        C: Tensor,           # (S, G, N)
        scan_state_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...
```

`__init__()` validates `num_heads % num_groups == 0` and owns the
input-independent parameters
`decay_rate` \(=\exp(A_{\log})\), \(D\), and `dt_bias`. `decay_rate` is the
positive transformed checkpoint parameter required by the parameter-port
constraint in §4.3. Keeping these parameters in the scan module makes its graph
boundary equal to the mathematical SSM boundary; the parent mixer owns only
projections and convolution weights.

**`forward()` mapping:**

1. Validate the shapes above and optional state \((h,p,N)\).
2. Apply `dt_bias` with `ParameterBias`, inline Softplus as
   `Exp → Add(one) → Log`, then apply `decay_rate` with `ParameterScale` and
   `Multiply(minus_one)` to form \(\overline\Delta A\).
3. Reshape/repeat B and C from \(G\) groups to \(h\) heads. Since Zepto's
   `RepeatKV` semantics may not match this rank/layout, use
   `Reshape → explicit Gather/Concat → Reshape` if necessary; do not silently
   relabel a tensor's shape.
4. For each \(t\), build \(\overline A\), \(\overline B\), the broadcast outer
   product \(x_t[...,None]\odot\overline B_t[:,None,:]\), and the state update
   with `Exp`, `Multiply`, and `Add`.
5. Contract the state axis with C using `MatMul` or elementwise
   `Multiply → ReduceSum(axis=-1)`. For \(D\odot x_t\), transpose the head
   axis to the last dimension, apply `ParameterScale(D)`, then transpose back;
   this shares each \(D_i\) over all \(p\) channels without creating \(hp\)
   parameters. Add the skip and concatenate sequence outputs.
6. Return output \((S,h,p)\) and final state \((h,p,N)\). Keep the state path
   in fp32 and restore activation dtype on the sequence output.

The unrolled recurrence and the chunked reference are mathematically equivalent:
chunking factorizes the same causal linear recurrence to expose parallel
matrix products. The Zepto identity graph intentionally uses the simple
token recurrence; the fused region recipe represents the production chunked
kernel.

#### Tests

- Compose with Nemotron-mini shapes and `S=4`.
- Assert output `(S,h,p)` and final state `(h,p,N)`.
- Assert B/C group expansion maps each group to exactly `h // G` contiguous
  heads.
- Assert `Softplus(delta + dt_bias)`, negative-exponential A, D skip, and C
  contraction are present; assert no scan-local gate.

---

### Phase 6 — Block modules

#### 7.10 `src/zepto/modules/mixer_config.py`

The config classes are dimensional proofs for the equations above. Validate
the identities at construction time so `forward()` never has to guess a head
layout:

\[
\begin{aligned}
d_{Q}=d_{K}&=h_kd_k,& d_V=d_Z&=h_vd_v,& h_v\bmod h_k&=0,\\
m&=hp,& C_{\text{mamba-conv}}&=m+2GN,& h\bmod G&=0.
\end{aligned}
\]

```python
@dataclass(frozen=True, slots=True)
class GatedDeltaNetConfig:
    hidden_size: int
    num_qk_heads: int
    num_v_heads: int
    head_dim: int
    conv_kernel_size: int = 4
    conv_bias: bool = False

@dataclass(frozen=True, slots=True)
class Mamba2MixerConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    state_size: int
    num_groups: int
    conv_kernel_size: int = 4
    conv_bias: bool = True
    projection_bias: bool = False
```

If distinct key/value head dimensions are expected later, replace the single
`head_dim` field with `key_head_dim` and `value_head_dim`; do not infer one
from residual width. The present target checkpoints use 128/128 for Qwen and
64 channels per Mamba head.

#### 7.11 `src/zepto/modules/gated_delta_net.py`

**What it is.** `GatedDeltaNet` maps the residual stream to the sufficient
statistics consumed by the delta-rule memory, applies local causal filtering,
runs the recurrent memory, gates/normalizes the readout, and projects it back
to residual width. It is a linear-attention token mixer: the persistent
\(d_k\times d_v\) state replaces an \(S\times S\) attention matrix.

Let \(\mathbf X\in\mathbb R^{B\times S\times d}\),
\(r=h_v/h_k\), \(d_Q=d_K=h_kd_k\), and \(d_V=h_vd_v\).
The checkpoint has two distinct input projections:

\[
\begin{aligned}
[\mathbf Q_0,\mathbf K_0,\mathbf V_0,\mathbf Z]
  &= \mathbf XW_{qkvz},&
W_{qkvz}&\in\mathbb R^{d\times(2d_Q+2d_V)},\\
[\mathbf b,\mathbf a]
  &= \mathbf XW_{ba},&
W_{ba}&\in\mathbb R^{d\times2h_v}.
\end{aligned}
\]

`fix_query_key_value_ordering()` is not extra arithmetic. It reshapes each Q/K
head together with its \(r\) associated V/Z heads before splitting, then
flattens the value-head group. This preserves the checkpoint's parameter
layout. In Zepto, implement last-axis split as
`Transpose → Split(leading axis) → Transpose`; current `Split` has no axis
argument.

The mathematical forward pipeline is:

1. Apply the two `LinearMatMul` projections and recover Q, K, V, Z, \(b\), and
   \(a\) with the exact sizes above.
2. Concatenate **Q, K, and V only**, then apply
   `DepthwiseCausalConv1d(2d_Q+d_V, K, activation="silu", bias=False)`.
   The gate Z and scalars \(b,a\) bypass convolution.
3. Reshape Q/K by \(h_k\), apply exact L2 normalization, then repeat each Q/K
   head \(r\) times so all scan inputs have \(h_v\) heads.
4. Scale Q by \(d_k^{-1/2}\), compute
   \(\beta=\sigma(b)\), and compute
   \(g=-\exp(A_{\log})\operatorname{softplus}(a+\mathrm{dt\_bias})\).
   The mixer owns learned `decay_rate=exp(A_log)` and `dt_bias`, both
   \((h_v,)\). Apply them with `ParameterScale` and `ParameterBias`;
   inline Softplus as `Exp → Add(one) → Log` when parent provenance must remain
   `GatedDeltaNet`.
5. Call `GatedDeltaScan(Q,K,V,beta,g)` with the optional recurrent state.
6. Reshape the scan output to expose each \(d_v\)-wide value head and apply
   `GatedRMSNorm(scan_output, Z)` — norm first, SiLU gate second.
7. Flatten \(h_vd_v\) and apply
   \(W_o\in\mathbb R^{d_V\times d}\).

These steps also explain parameter ownership: projections, transformed
`decay_rate`, `dt_bias`, and output projection belong to the mixer; the
depthwise conv and gated norm own their local weights; `GatedDeltaScan` owns no
weights.

Constructor stores `GatedDeltaNetConfig`. Forward signature:

```python
def forward(
    self,
    hidden_states: Tensor,
    conv_state_in: Tensor | None = None,
    scan_state_in: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (output, conv_state_out, scan_state_out)."""
```

`forward()` accepts \((S,d)\) or \((B,S,d)\), validates the residual width, and
returns output with the same shape. `conv_state_out` has
\((2d_Q+d_V,K)\) and `scan_state_out` has
\((h_v,d_k,d_v)\) for the rank-2/B=1 form. State inputs affect the first
causal window/recurrent update but do not change the output shape.

#### 7.12 `src/zepto/modules/mamba2_mixer.py`

**What it is.** `Mamba2Mixer` converts each token into a gate, SSM input,
selective B/C vectors, and a time step; locally filters the SSM-related
channels; evolves bounded recurrent state; then gates, group-normalizes, and
returns to residual width.

Let \(m=hp\) and \(C_c=m+2GN\). One projection produces:

\[
[\mathbf Z,\mathbf U_{xBC},\Delta]
=\mathbf XW_{\mathrm{in}},
\qquad
W_{\mathrm{in}}\in\mathbb R^{d\times(m+C_c+h)}.
\]

The exact method order is:

1. `LinearMatMul` then split into gate
   \(\mathbf Z\in\mathbb R^m\), convolution payload
   \(\mathbf U_{xBC}\in\mathbb R^{C_c}\), and raw
   \(\Delta\in\mathbb R^h\).
2. Apply
   `DepthwiseCausalConv1d(C_c, K, activation="silu", bias=config.conv_bias)`
   to \(\mathbf U_{xBC}\) only.
3. Split the convolved payload into
   \(\mathbf x\in\mathbb R^m\),
   \(B\in\mathbb R^{G\times N}\), and
   \(C\in\mathbb R^{G\times N}\); reshape
   \(\mathbf x\) to \((h,p)\).
4. Call `SelectiveSSMScan(x, delta, B, C)`. The scan owns
   transformed `decay_rate`, \(D\), and `dt_bias`, and returns \((S,h,p)\).
5. Flatten to \(m\), then call
   `GatedGroupedRMSNorm(scan_output, Z)`. This multiplies by
   \(\operatorname{SiLU}(Z)\) **before** group RMS normalization.
6. Apply \(W_o\in\mathbb R^{m\times d}\).

The gate bypasses convolution and scan; \(\Delta\) bypasses convolution; B/C
are convolved because the reference includes them in
\(\mathbf U_{xBC}\). This wiring is the reason `in_proj` has the seemingly
non-obvious output width \(m+(m+2GN)+h\).

Constructor stores `Mamba2MixerConfig`; `forward()` mirrors the DeltaNet state
signature and returns `(output, conv_state_out, scan_state_out)`. For Nemotron,
the dimensions are \(d=2688,h=64,p=64,N=128,G=8\), hence
\(m=4096\), \(C_c=6144\), and projection width
\(m+C_c+h=10{,}304\). Compute this value from config rather than hand-coding
it. The conv state is
\((6144,K)\), and recurrent state is \((64,64,128)\) for \(B=1\).

#### Phase 6 method-level tests

- Check Qwen projection widths `2*d_Q + 2*d_V` and `2*h_v`, and assert only
  Q/K/V enter convolution.
- Check Q/K repeat factor `h_v // h_k`, query scale \(d_k^{-1/2}\), beta
  sigmoid, and log-decay chain.
- Check Mamba projection width `m + (m + 2*G*N) + h`, split widths, and that
  only `x/B/C` enter convolution.
- Assert both mixers preserve residual-stream shape and return exact conv/scan
  state shapes.

---

### Phase 7 — Hybrid scheduling

#### 7.13 `src/zepto/modules/layer_spec.py`

`LayerSpec` is a typed description of a piecewise function, not a numerical
module. For layer \(\ell\), it selects exactly one token mixer \(M_\ell\) and
whether a second feed-forward map \(F_\ell\) exists. Its validation ensures the
selected branch has all dimensions needed to define its equations and that
unselected branches cannot accidentally contribute weights or graph nodes.

```python
MixerKind = Literal["gated_delta_net", "mamba2", "attention", "moe"]

@dataclass(frozen=True, slots=True)
class LayerSpec:
    mixer: MixerKind
    # Exactly one of the following populated according to mixer:
    gated_delta: GatedDeltaNetConfig | None = None
    mamba2: Mamba2MixerConfig | None = None
    attention: AttentionConfig | None = None
    moe: None = None  # use NemotronMoEBlock factory args separately — see below
    ffn: Literal["swiglu", "none"] = "none"
    swiglu_intermediate: int | None = None
```

Validation: `mixer="gated_delta_net"` ⇒ `gated_delta` not None; `ffn="swiglu"` ⇒ `swiglu_intermediate` not None; Nemotron Mamba/attention/MoE use `ffn="none"`.
Also require exactly one matching branch source (a config payload, or
`moe_factory` for MoE), reject irrelevant non-`None` payloads, and validate
that every mixer maps residual width \(d\) back to \(d\) so the residual
addition is defined.

#### 7.14 `src/zepto/modules/hybrid_decoder_block.py`

Nemotron-style **single mixer** block:

\[
\mathbf x_{\ell+1}
=\mathbf x_\ell+
M_\ell\!\left(\operatorname{RMSNorm}_\ell(\mathbf x_\ell)\right).
\]

Pre-normalization stabilizes the input to whichever heterogeneous mixer is
selected. The residual path remains an identity map, so all Mamba, attention,
and MoE branches must return the same \((B,S,d)\) shape even though their
internal widths differ.

```python
class HybridDecoderBlock(Module):
    module_kind = "HybridDecoderBlock"

    def __init__(self, spec: LayerSpec, *, moe_factory: Callable[..., Module] | None = None) -> None: ...

    def forward(
        self,
        hidden_states: Tensor,
        *mixer_inputs: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        # y = hidden_states + mixer(RMSNorm(hidden_states))
        # recurrent branch: return (y, conv_state_out, scan_state_out)
        # stateless branch: return y
```

Wire mixers:

| `spec.mixer` | Submodule |
|--------------|-----------|
| `mamba2` | `Mamba2Mixer` |
| `attention` | `FlexibleAttention` |
| `moe` | `NemotronMoEBlock` via preset factory |
| `gated_delta_net` | `GatedDeltaNet` |

`__init__()` resolves one `self.mixer` from the spec and owns one pre-mixer
`RMSNorm(d)`. `forward()` saves the residual, normalizes once, dispatches only
the selected branch, validates the branch-specific `mixer_inputs` arity, and
emits one final `Add`. Interpret them as `(conv_state_in, scan_state_in)` for
Mamba, `(attention_mask, rope_cos, rope_sin)` for attention, and `()` for MoE.
Mamba returns `(hidden_states, conv_state_out, scan_state_out)`; attention and
MoE return only `hidden_states` because attention state is accounted by the KV
horizon port. Do not put `None` in a module output tuple:
`compose_graph()` marks every tuple member as a graph `Tensor`.

#### 7.15 `src/zepto/modules/qwen35_language_decoder_block.py`

Qwen has a token mixer **and** a SwiGLU in every language layer:

\[
\begin{aligned}
\mathbf u_\ell
&=\mathbf x_\ell+
  M_\ell(\operatorname{RMSNorm}_{1,\ell}(\mathbf x_\ell)),\\
\mathbf x_{\ell+1}
&=\mathbf u_\ell+
  \operatorname{SwiGLU}_\ell(
    \operatorname{RMSNorm}_{2,\ell}(\mathbf u_\ell)).
\end{aligned}
\]

The first residual must be completed before the second RMSNorm; normalizing
the original \(\mathbf x_\ell\) twice would represent a different block.

```python
class Qwen35LanguageDecoderBlock(Module):
    module_kind = "Qwen35LanguageDecoderBlock"

    def __init__(self, spec: LayerSpec) -> None:
        # spec.mixer in {gated_delta_net, attention}
        # spec.ffn == "swiglu" always for Qwen language

    def forward(
        self,
        hidden_states: Tensor,
        *mixer_inputs: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        # pre-norm attention/delta + residual + pre-norm SwiGLU + residual
```

`__init__()` chooses either `GatedDeltaNet` or `FlexibleAttention`, then owns
two independent RMSNorm parameter vectors and one SwiGLU. `forward()` threads
conv/recurrent state only for DeltaNet and attention mask/position/cache inputs
only for full attention, while preserving the two equations above. DeltaNet
layers return `(hidden_states, conv_state_out, scan_state_out)`; full-attention
layers return only `hidden_states`. The fixed `LayerSpec` makes this return
arity knowable at composition time.

#### 7.16 `src/zepto/modules/mixer_presets.py`

```python
def qwen35_language_layer_specs() -> tuple[LayerSpec, ...]:
    """64 entries: repeat 16× [delta, delta, delta, attn+ffn]."""

def nemotron_h_layer_specs() -> tuple[LayerSpec, ...]:
    """52 entries from gap doc block-type indices."""
```

These factories contain no tensor math. They instantiate the discrete layer
function \(\ell\mapsto(M_\ell,F_\ell)\):

\[
\begin{aligned}
\text{Qwen: }&
[D,D,D,A]^{\times16},
&&F_\ell=\text{SwiGLU for all }\ell,\\
\text{Nemotron: }&
M_\ell\in\{\text{Mamba},\text{Attention},\text{MoE}\},
&&F_\ell=\varnothing.
\end{aligned}
\]

Nemotron's exact index sets are copied from the released config (§4.2).
Hard-code mixer kind per layer index (document source URL in docstring), but
derive all widths and state shapes through config constructors rather than
duplicating arithmetic constants.

#### Tests — `tests/modules/test_layer_spec.py`

- `len(qwen35_language_layer_specs()) == 64`
- `len(nemotron_h_layer_specs()) == 52`
- Count mixer kinds matches gap doc (23 mamba, 6 attn, 23 moe).
- Assert every Qwen layer has SwiGLU and every Nemotron layer has no second FFN.
- Compose one block of each mixer kind and assert the residual `Add` operands
  have identical shapes.

---

### Phase 8 — Region registration (discovery now; variants via kernel-full)

**Policy:**

| Component | Step 5 change |
|-----------|---------------|
| All primitive ops used by modules | Existing identity lowering |
| Fused `region/*` leaves | **`/kernel-full`** delivers recipes + variants; Step 5 registers **rules + recipe dataclass stubs** |

For each region kind in §8.1, add:

1. `recipes/<slug>.py` — frozen dataclass with `forward_flops(...)`, `saved_tensors`, decode/prefill flags.
2. `implementations/regions/<slug>/rules.py` — `ProvenanceMatchRule` on `module_kind`.
3. `implementations/regions/<slug>/variants.py` — at minimum a **`reference`** variant; fused variant added by kernel implementer.
4. Register in `implementations/regions/__init__.py` and wire into `LoweringRegistry` in `implementations/__init__.py`.

**Provenance rules (required in Step 5):**

| Rule id | `kind` | `component_type` |
|---------|--------|------------------|
| `prov-l2-normalize` | `region/l2_normalize` | `L2Normalize` |
| `prov-depthwise-causal-conv1d` | `region/depthwise_causal_conv1d` | `DepthwiseCausalConv1d` |
| `prov-gated-delta-scan` | `region/gated_delta_scan` | `GatedDeltaScan` |
| `prov-mamba2-scan` | `region/mamba2_scan` | `SelectiveSSMScan` |
| `prov-gated-rms-norm` | `region/gated_rms_norm` | `GatedRMSNorm` |
| `prov-gated-grouped-rms-norm` | `region/gated_grouped_rms_norm` | `GatedGroupedRMSNorm` |
| `prov-gated-delta-net` | `region/gated_delta_net` | `GatedDeltaNet` |
| `prov-mamba2-mixer` | `region/mamba2_mixer` | `Mamba2Mixer` |

**Mutually exclusive composition (document in each rules file):**

- `region/gated_delta_net` subsumes child regions on the same module instance.
- `region/mamba2_mixer` subsumes child regions on the same module instance.
- Do not stack leaf and block regions on the same forward path when block provenance matches.

Until fused variants exist, lowering falls back to identity on primitives (no double-counting).

---

## 8. Fused kernel catalog (`/kernel-full`)

### 8.1 Region kinds and harness slugs

Use these **exact** names when invoking the kernel pipeline:

| Priority | `/kernel-full` slug | Region kind | Reference module | Atto / golden source |
|----------|---------------------|-------------|------------------|----------------------|
| 1 | `l2-normalize` | `region/l2_normalize` | `L2Normalize` | un-averaged sum-of-squares + no γ |
| 2 | `depthwise-causal-conv1d` | `region/depthwise_causal_conv1d` | `DepthwiseCausalConv1d` | new derivation |
| 3 | `gated-delta-scan` | `region/gated_delta_scan` | `GatedDeltaScan` | Atto `34 - gated-deltanet.md` |
| 4 | `gated-rms-norm` | `region/gated_rms_norm` | `GatedRMSNorm` | RMSNorm + SiLU gate |
| 5 | `mamba2-scan` | `region/mamba2_scan` | `SelectiveSSMScan` | new derivation + HF |
| 6 | `gated-grouped-rms-norm` | `region/gated_grouped_rms_norm` | `GatedGroupedRMSNorm` | new derivation |
| 7 | `gated-delta-net` | `region/gated_delta_net` | `GatedDeltaNet` | compose Tier A leaves |
| 8 | `mamba2-mixer` | `region/mamba2_mixer` | `Mamba2Mixer` | compose Tier A leaves |

**Decode sub-kinds** (same slug research doc; variant id suffix):

| Region sub-kind | Trigger |
|-----------------|---------|
| `region/depthwise_causal_conv1d/decode` | `InvocationContext.phase == "decode"` or `seq_len == 1` with conv state port |
| `region/gated_delta_scan/decode` | decode + recurrent state port |
| `region/mamba2_scan/decode` | decode + recurrent state port |

### 8.2 Recipe obligations (for kernel implementer)

Each kernel doc in `docs/kernel/<slug>.md` must specify:

1. Forward/backward FLOP closed form (theoretical FLOP = 2× MAC).
2. Saved-for-backward tensors (training): all \(S\) DeltaNet states are
   `S × h_v × d_k × d_v`; all \(S\) Mamba states are
   `S × h × head_dim × state_size` unless the kernel recomputes/checkpoints
   them.
3. Inference decode: only the final recurrent state is persisted
   (`h_v × d_k × d_v` for DeltaNet, `h × head_dim × state_size` for Mamba);
   reference conv buffer is `K × C` (while `K-1` is the mathematical minimum).
4. Prefill vs decode FLOP differences (S=1 projections vs full S).
5. VRAM elision vs identity decomposed path (which temps disappear).
6. State port events to emit (call helpers from §7.4).

**Gated DeltaNet scan (Atto-aligned forward FLOP sketch):**

- Per value head per timestep: rank-1 update + output ≈ `O(d_k × d_v)` MACs.
- Full sequence: multiply by `S × h_v`.
- Projections billed separately as `region/linear` unless block fusion subsumes them.

### 8.3 Kernel harness workflow

For each slug in priority order:

```
/architect-kernel <slug>   # or /kernel-full <slug> for combined pipeline
/implement-kernel
```

Archive to `docs/kernel/<slug>.md`. Then add variant to `implementations/regions/<slug>/variants.py` and tests in `tests/lowering/regions/test_<slug>.py`.

---

## 9. Integration tests

### 9.1 `tests/integration/test_gated_delta_prefill_decode_horizon.py`

Mini config: `d=128`, `h=2`, `S_prefill=32`, `decode_steps=4`.

```python
spec = HorizonSpec.inference(
    prefill=32,
    decode_steps=4,
    conv=ConvStateConfig(layer_indices=(0,), channels=..., kernel_size=4, dtype=DType.BF16),
    recurrent=RecurrentStateConfig(
        layers=((0, "gated_delta", 2, 16, 16),),
        dtype=DType.BF16,
    ),
)
```

Assert:

- `len(report.per_step) == 5`
- Recurrent + conv state bytes > 0 after prefill
- Decode step FLOPs < prefill FLOPs (S=1 vs S=32)

### 9.2 `tests/integration/test_mamba2_prefill_decode_horizon.py`

Same structure with `kind="mamba2"`.

### 9.3 Regression command

```bash
cd /Users/veoon/hslu/zepto-kernel-implementation
python -m pytest tests/ -q
```

Must pass unchanged:

- `tests/integration/test_apertus_prefill_estimate.py`
- `tests/integration/test_apertus_prefill_decode_horizon.py`
- `tests/lowering/regions/test_gqa.py`
- Step 4 MoE tests (if merged)

---

## 10. Public API (final exports)

Add to `src/zepto/modules/__init__.py`:

```python
from .l2_normalize import L2Normalize
from .gated_rms_norm import GatedRMSNorm
from .gated_grouped_rms_norm import GatedGroupedRMSNorm
from .depthwise_causal_conv1d import DepthwiseCausalConv1d
from .gated_delta_scan import GatedDeltaScan
from .selective_ssm_scan import SelectiveSSMScan
from .gated_delta_net import GatedDeltaNet
from .mamba2_mixer import Mamba2Mixer
from .mixer_config import GatedDeltaNetConfig, Mamba2MixerConfig
from .layer_spec import LayerSpec, MixerKind
from .hybrid_decoder_block import HybridDecoderBlock
from .qwen35_language_decoder_block import Qwen35LanguageDecoderBlock
from .mixer_presets import qwen35_language_layer_specs, nemotron_h_layer_specs
```

Horizon exports (if public today for `KVConfig`):

```python
from .horizon.spec import ConvStateConfig, RecurrentStateConfig
```

---

## 11. Acceptance criteria (merge checklist)

### Structural graph (Step 5 PR)

- [ ] **No new files** under `src/zepto/semantic/operations/` for mixer-specific ops.
- [ ] Leaf modules compose graphs using only existing operation families.
- [ ] Learned values bind only through parameter-aware operations
      (`LinearMatMul`, `ParameterScale`, `ParameterBias`); no registered
      `Parameter` is passed as a generic data input.
- [ ] L2 normalization uses sum-of-squares (not mean); Qwen gate is
      norm-before-SiLU and Nemotron gate is SiLU-before-group-norm.
- [ ] DeltaNet state is `(h_v, d_k, d_v)` and Mamba state is
      `(h, head_dim, state_size)` in both module outputs and byte accounting.
- [ ] Qwen convolution contains Q/K/V only; Mamba convolution contains x/B/C
      only.
- [ ] All leaf/block modules inline operations with correct `module_kind` provenance.
- [ ] `Conv1DState` and `RecurrentScanState` integrated into `StatePortRegistry` with prefill/decode `advance()` logic.
- [ ] `ConvStateConfig` / `RecurrentStateConfig` wired through `HorizonSpec`.
- [ ] `conv_state_port_events` and `recurrent_state_port_events` implemented in lowering helpers.
- [ ] `GatedDeltaNet` and `Mamba2Mixer` block modules compose end-to-end.
- [ ] `LayerSpec` + `HybridDecoderBlock` + `Qwen35LanguageDecoderBlock` implemented.
- [ ] `qwen35_language_layer_specs()` returns 64 layers; `nemotron_h_layer_specs()` returns 52 layers with correct mixer counts.
- [ ] Module compose tests pass for all new modules.
- [ ] Horizon integration tests pass for DeltaNet and Mamba2 mini configs.
- [ ] Provenance rules registered for all §8.1 region kinds.
- [ ] No regression on Apertus integration tests and GQA lowering tests.
- [ ] Public exports updated.
- [ ] Gap doc §5 links to this plan.
- [ ] All module calls use positional tensor inputs compatible with the current
      `Module.__call__`; stateful block return tuples contain tensors only.

### Fused kernels (follow-up PRs via `/kernel-full`)

- [ ] Each slug in §8.1 has `docs/kernel/<slug>.md` archived.
- [ ] Region variants bill Atto-aligned FLOPs/VRAM (golden tests in `tests/lowering/regions/`).
- [ ] Decode sub-variants emit correct state port events.
- [ ] Block regions (`gated-delta-net`, `mamba2-mixer`) subsume child regions on match.

---

## 12. Future work (explicitly not this step)

| Item | Step / follow-up |
|------|------------------|
| Qwen3.8 vision + multimodal fusion | Step 6 |
| Full model modules (`Qwen3_8`, `NemotronH`) | after Steps 5–6 |
| Chunked DeltaNet kernel (chunk_size > 1) | kernel follow-up |
| CUDA `mamba-ssm` / `causal-conv1d` backend variants | kernel follow-up |
| MTP heads | Step 7 |

---

## 13. Agent implementation notes

1. **Read first:** `src/zepto/modules/swiglu.py` (provenance), `src/zepto/modules/rms_norm.py` (primitive decomposition), `src/zepto/analysis/horizon/state.py` (KV pattern), `tests/integration/test_apertus_prefill_decode_horizon.py`.
2. **Read HF before Mamba/DeltaNet block modules:** Qwen3-Next and Nemotron-H modeling files — match projection splits and head grouping exactly.
3. **Inline ops in modules** — do not call submodules in ways that split provenance (exception: `Linear` weight holders are OK if ops are inlined at call site like SwiGLU).
4. **Run tests after every phase** — do not implement all phases before testing.
5. **Do not** add mixer-specific `Operation` subclasses to avoid catalog bloat.
6. **Do not** change Apertus constants or break `ApertusDecoderBlock`.
7. **Unroll scans at compose time** with Python `for t in range(S)` — S must be known from input tensor shape during composition.
8. **Fused costing** comes from regions, not from summing unrolled scan MatMuls — register provenance early, implement variants via `/kernel-full`.
9. When Atto and HF disagree on a minor detail, **checkpoint config + HF forward** win for graph structure; **Atto** wins for FLOP/VRAM goldens unless documented otherwise.
10. For Nemotron MoE layers in `HybridDecoderBlock`, reuse Step 4 `nemotron_moe_block()` — do not reimplement routing.
