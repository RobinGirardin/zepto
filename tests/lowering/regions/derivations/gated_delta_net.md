# Golden FLOP / VRAM notes: GatedDeltaNet block envelope

Cross-check against `_workspace/research.md` §4.2 / §5 and Atto [34 - gated-deltanet.md](https://github.com/RobinGirardin/atto/blob/main/docs/cost-estimation-framework/30%20-%20attention/34%20-%20gated-deltanet.md).

## Tier-A composed default (Zepto)

Kernel-accurate block forward (prefill):

\[
4 S d (d_Q + d_V) + 4 S d h_v + 12 S C + 6 S h_k d_k + S h_v (d_k + 11)
+ S h_v (8 d_k d_v + 2 d_v + 1) + 10 S h_v d_v + 2 S d_V d,
\]

with \(C = 2 d_Q + d_V\), \(d_Q = h_k d_k\), \(d_V = h_v d_v\).

## Qwen3.8-scale spot check (research §4.2)

| Symbol | Value |
|--------|-------|
| \(S\) | 8192 |
| \(d\) | 5120 |
| \(h_k, h_v\) | 16, 48 |
| \(d_k, d_v\) | 128 |

Dominant terms: qkvz + ba + o_proj GEMMs, then scan \(\mathcal{O}(S h_v d_k d_v)\). Total forward \(\approx 1.95 \times 10^{12}\) FLOPs/layer/prefill (see research table).

## Atto paper grouping (documentation only)

Scan recurrence Atto \(\approx 9 S h_v d_k d_v\) vs Zepto scan leaf \(8 S h_v d_k d_v\) — cross-check full block against Atto tables, not leaf-for-leaf replacement.
