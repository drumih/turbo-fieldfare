# Qwen3.8 MTP on the 24 GB M3: what the second row actually costs (2026-11-09)

Status: investigation closed without a speedup. The two kernel-level fixes tried
here were measured and had no effect; the pass is bounded by expert I/O and the
two-row compute, not by the prefill tile shapes or the router kernel.

## The measured problem

Recorded Qwen3.8-Flash-Next 4-bit trace
(`benchmark/tinytitan_mtp_phases.py`, `--ram-budget` 12 G, 256 greedy tokens,
`presence_penalty=0`, verify=pair):

| arm | tok/s | pass | acceptance | emitted/pass |
| --- | ---: | ---: | ---: | ---: |
| scalar | 4.620 | 216 ms/token | — | — |
| MTP | 3.891 | 480 ms/pass | 86.9% | 1.869 |

Break-even for 1.869 emitted is a pass below **404 ms**; it costs **480 ms
(2.22x a token)**, so MTP loses 16% on this trace. The earlier `-43%` figure was
the same shape under GPU contention (`scalar 3.827`, `MTP 2.175`).

Per-pass attribution:

| phase | ms/pass | note |
| --- | ---: | --- |
| proposal + checkpoint | 23.6 | one draft layer |
| verify backbone | 417.5 | two target rows |
| verify head | 22.2 | two-row head |
| commit / rollback | 15.7 | |
| **total** | **480** | |

Inside the verify backbone, GPU roles total **233 ms/pass** and device/host gaps
**~184 ms/pass**. The largest gaps are:

- `prefill_shared_expert → verify_routed_pair`: **164 ms/pass** — the union miss
  fetch, 227 MiB/pass at the measured ~3.9 GB/s;
- `verify_head → prefill_gdn_router`: **46 ms/pass** — pass turnaround.

The per-token arithmetic is the whole story:

| per emitted token | scalar | MTP pass ÷ 1.869 |
| --- | ---: | ---: |
| expert I/O | 102 ms | **88 ms** |
| GPU / verify | 147 ms | **125 ms** |
| pass overhead (turnaround, proposal, commit) | — | **44 ms** |
| total (with the scalar arm's 33 ms of I/O/compute overlap) | 216 ms | **~257 ms** |

MTP already does *less* I/O and *less* GPU work per token than scalar. It loses
because the 2-row pass still has to process two rows (2.0x compute before any
sharing) and because the two-row union still needs 1.6x the bytes of one token.
At 86.9% acceptance the pass has to be 1.87x a token to break even; a perfectly
fused pass is at best ~1.6x, and the current one is 2.22x.

## What was tried and measured

### 1. Shrink the verify prefill chunk from 32 to 2

The verify path called `PrefillRuntimeConfig.production(chunkTokens: 32)` while
running exactly two rows. A chunk sized to the rows dispatches the same kernels
for two queries and skips the QMM projection path (`useTwoRowProjection` already
supplies the two-row GEMV).

Result on Ornith 1.5 35B-A3B 4-bit at `--ram-budget 12G`:

| arm | pass | verify backbone |
| --- | ---: | ---: |
| chunk 32 | 413 ms | 379 ms |
| chunk 2 | 415 ms | 377 ms |

No change. The pass is not tile-bound; the wall is expert I/O and the serial
two-row stages.

### 2. Use the decode router for the two verify rows

The prefill router is one 32-thread threadgroup per token, while the decode
router is fully parallel over experts. Replacing the two-row prefill dispatch
with two single-row decode-router dispatches is numerically identical on the
tested prompt.

Result: **no measurable change** (415 ms vs 413 ms pass, within the run noise).
The router is not the bottleneck; the role labelled `prefill_gdn_router` is the
whole GDN block, not the router alone.

Both experiments were reverted. They added configuration without changing the
outcome.

## Why Unsloth's 1.3–1.7x does not transfer

Unsloth's MTP measurements are on a GPU-resident model: both the draft and the
target weights are in HBM/VRAM, and a second row costs mostly extra FLOPs, which
a wide GPU absorbs. TinyTitan's 125B target streams ~400 MB of experts per token
from a 3.9 GB/s SSD. A two-row pass needs the **union** of both rows' top-10
experts, 1.6x the bytes of one row, and the SSD is already at its measured wall
(no queueing left to fill). MTP can at best amortise compute; it cannot amortise
the expert bytes.

The project's own earlier gate reached the same place from the other side:
`docs/qwen38-mtp-and-overlap-verdict.md` measured a pass at 2.238x a token,
byte-identical output, and closed the decode-native-verify proposal. This
investigation confirms that verdict on a second model (Ornith) and shows why the
two plausible kernel fixes do not move it.

## What a fix would actually require

1. **A decode-native two-row target pass** — one command buffer, two-row GEMV,
   attention/GDN sharing weights, no 32-token tile scratch. The earlier "85 ms"
   estimate assumed the GPU term was 2.0x scalar; the measured term is 233 ms,
   or **1.59x** scalar (147 ms), so a perfect 1.3x fusion saves only ~42 ms.
   Combining that with a 1.27x-only union fetch and less host turnaround gives
   the optimistic case below; the pass still has to get under 404 ms.

   | term | now | optimistic fused | realistic fused |
   | --- | ---: | ---: | ---: |
   | GPU (233 ms now) | 233 | 191 (1.3x scalar) | 210 |
   | expert I/O (164 ms now) | 164 | 130 (1.27x scalar) | 150 |
   | turnaround/proposal/commit (83 ms) | 83 | 60 | 75 |
   | **pass** | **480** | **381** | **435** |
   | tok/s at 86.9% acceptance | 3.89 | **4.90** | **4.30** |

   Against scalar's 4.62 tok/s, the ceiling is roughly **+6%**, and the
   realistic result is **−7%**. It is a large kernel project for a bet that
   lands somewhere between "slightly slower" and "slightly faster", not a
   decode-speed feature.
2. **Overlap the union fetch with the next layer's attention**, impossible while
   layer L+1 depends on layer L's routed residual; the only real lever is a
   better expert predictor, which the project has already measured to its
   ceiling.
3. **Make the target resident** (96 GB+ machine). Then the I/O term collapses
   and the Unsloth-style MTP gain becomes available. On a 24 GB M3 it does not.

## Recommendation

Do not enable MTP for the SSD-streaming Qwen3.8 install on this machine. It
cannot beat scalar decode without a decode-native two-row pass, and even that
pass only reaches break-even at the measured acceptance — its realistic
outcome is 4.3 tok/s against scalar's 4.62. The 43% loss is not a
mystery bug in the draft or the verifier; it is the arithmetic of streaming
experts twice for a 1.87-token payoff.

The MTP code stays correct and opt-in. If it is kept, the useful guard is a
startup projection using the measured pass multiple and acceptance, refusing to
engage when `emitted/pass < pass_multiple`, rather than paying the loss to
discover it.
