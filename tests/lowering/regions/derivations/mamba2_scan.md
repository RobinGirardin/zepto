# Golden FLOP / VRAM notes: Mamba-2 selective SSM scan

Cross-check against `_workspace/research.md` §4–§6 and HF Nemotron Mamba-2 forward slice.

## Zepto identity grouping (default fused leaf)

Per head per timestep:

\[
5 p N + N + 2 p + 7 \quad\text{(forward)}, \qquad
10 p N + 2 p + N + 4 \quad\text{(backward, training)}.
\]

Over \(S\) steps and \(h\) heads:

\[
\mathrm{FLOPs}_{\mathrm{fwd}} = S h (5 p N + N + 2 p + 7).
\]

## Nemotron-style spot check

\(S=8192\), \(h=64\), \(p=64\), \(N=128\):

\[
8192 \times 64 \times (5 \times 64 \times 128 + 128 + 128 + 7)
\approx 2.16 \times 10^{10}\ \text{forward FLOPs}.
\]

Saved fp32 states (training, all timesteps): \(4 \cdot S h p N \approx 1.0\ \mathrm{GiB}\) per layer.

## Paper SSD grouping (documentation only)

Mamba-2 SSD block factorization can be quoted as \(\approx 4 S h p N\) forward on the dominant recurrence — Zepto fused leaf bills §4.1 (\(5 S h p N\) leading term), not the coarser paper grouping.
