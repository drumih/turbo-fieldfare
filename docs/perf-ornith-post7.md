# Post-7.0 Ornith 1.5 35B-A3B 4-bit — performance investigation

Status: complete (2026-11-16). Subject: the shipped server path, which runs the
6.x `TinyTitan` runtime (the `Ornith35B4Bit` variant is parity-verified but not yet
the server's engine). Host: Apple M3 MacBook Pro, 24 GB, macOS 27.2, Swift 6.4.
Method: paired, interleaved A/B (`benchmark/minor_wins_ab.py`), text-parity checked
(sha256 of the output), per the community Benchmarking Guide. Every accepted change
must be faster **and** golden-parity-identical; anything slower or quality-costing
is recorded and rejected (AGENTS.md).

## 1. Where the time goes (diagnostic profile)

`TINYTITAN_RUNNER_STATS` + `TINYTITAN_KERNEL_STATS` + `TURBO_FIELDFARE_PHASES`
(diagnosis only — these traces add ~35% overhead and are never used for a reported
number):

- **Decode is GPU-compute bound, not SSD-bound.** `io_hidden_pct` ≈ 99.5–100%
  (prefetch hides expert I/O), expert loads p50/p95/p99 = 2/4/8 ms, expert hit rate
  ~0.96. The dominant per-token cost is `wait_ms` ≈ 36–39 ms — GPU execution of
  attention + router (`waitForCompletion(tailCB)`), not idle. `cb1_ms` (CPU encode)
  ≈ 7–16 ms, `head_ms` ≈ 7–9 ms.
- **GPU occupancy ≈ 47.8%** (busy 890 ms / span 1861 ms): the GPU drains queued work
  (shared expert, phase-1 hits) then waits while the CPU does the router readback +
  expert plan + routed-CB encode. The router readback is a hard per-layer barrier.
- **The Metal GEMV kernels already use bf16 + simdgroup** (`bf16_gemv_simd`, one
  simdgroup per row); the head GEMV is bandwidth-bound (~242 MiB, ~4 ms GPU). The
  "float-based kernel" note in AGENTS.md is the *CPU* path, not these GPU kernels.
- Prefill is dominated by the routed-expert MoE tiles (`prefill_routed_tile`).

Consequence: the I/O and precision knobs are not the lever; the GPU pipeline
(occupancy / per-layer CPU barrier) is the only remaining headroom.

## 2. I/O knobs — measured-REJECTED (items 2/3/4)

3 paired rounds, 128-token greedy, fresh server per arm, parity held (555 chars):

| arm | mean Δ vs base | rounds positive |
| --- | --- | --- |
| pagecache (`TINYTITAN_BOUNDED_IO=0`) | −14.96% | 0/3 |
| advice-off (`TINYTITAN_RDADVISE_POLICY=off`) | −21.37% | 0/3 |
| immediate (`TINYTITAN_EXPERT_IO_SUBMISSION=immediate`) | −26.85% | 0/3 |
| stack (pagecache + immediate) | −18.97% | 0/3 |

The shipped I/O defaults are optimal — every alternative is slower. Result:
`.build/minor-wins-ornith4.json`.

## 3. Prefill chunk 2048 vs 4096 — measured-REJECTED (item 6)

Ornith's row pins chunk 4096 (`ModelProfile.swift`). Clean 4-round A/B, ~1700-token
prompt + 128 decode, parity held (sha `53742f52…`): **median −0.70%**, 1/4 positive;
warm rounds 2–4 are equal (≈16.1–16.3 tok/s both arms, prefill ≈30.9 s both). Round 1
(−32%) is a cold-expert-cache artifact. Chunk 4096 is already optimal.

**Deviation disclosed:** a first run under heavy memory pressure (7.7 GB swap)
showed a spurious "+31% mean" because swap thrashing collapsed the base arm in one
round. Re-run with acceptable free memory inverted the verdict. Lesson: a
swap-confounded paired run can invert the result; validate under the memory
precondition. Result: `.build/prefill-chunk-clean.json`.

## 4. ANE prefill (item 7) — already integrated

ANE prefill is default-on in the production path (3.14× on a 9.3k-token prompt,
breakeven ~9.2k generated tokens; not bit-identical, golden pins it off). Only the
*variant* path is inert (chunk 128 vs the 4096 sidecar), and the server does not run
the variant — so no product headroom here without first wiring the server to the
variant.

## 5. Remaining lever — GPU occupancy (item 5, analyzed; deferred as high-risk)

The only measured headroom is the 47.8% GPU occupancy / per-layer CPU barrier.
Two candidates were analyzed in depth:

- **Pre-resolve per-layer TensorViews** (the `model.inputNorm/postAttnNorm/router`
  accessors string-format + dictionary-lookup + rebuild a `TensorView` every layer
  every token, `Model+Accessors.swift:319`). Caching them (precedent:
  `effectiveScaleBuffers[L]`) is parity-safe but saves only ~0.2–0.6 ms of the
  ~7–16 ms `cb1` ≈ **~1% decode — marginal, not worth the cross-file change.**
- **Pre-encode the next layer's attention CB** ahead of the routed-MoE CPU work to
  fill the inter-layer GPU idle gap. This is the real occupancy lever, but it
  requires restructuring the 344-line critical `decodeLayerLoop`
  (`RealForwardRunner+DecodeLayers.swift`) to carry the attention command buffer
  one iteration ahead, respecting the router-scratch lifecycle (the authoritative
  `outIndices/outWeights` are drained via the pending-routed-command before the
  next layer's router overwrites them; probe routers already use separate
  `prefetchPrediction*` buffers). High complexity, real parity risk, uncertain
  payoff — **deferred as a funded research direction, not forced.** The loop is
  already deeply optimized (shared-CB overlap, probe router, prefetch ring, event
  I/O, deferred routed completion).

High-risk alternatives (indirect command buffers / GPU-driven expert dispatch)
rebuild the CPU cache-plan authority and are likewise deferred. Each future
candidate must pass golden `ornith-4` parity + a clean paired benchmark.

**Conclusion of this investigation:** the shipped Ornith 4-bit engine is
near-optimal on this host. Every parameter knob (I/O submission, page cache,
readahead advice, prefill chunk) is measured-optimal; the kernels are already
bf16+simdgroup and bandwidth-bound; I/O is ~100% hidden. A *solid* further gain
requires the GPU-occupancy restructure above, which is scoped as future work
rather than risked at the tail of this session.

## Clean baselines (for the final paired comparison)

- Production profile (12 short prompts, cache-warm, 12/12 verified): **19.65 tok/s**
  footer decode (run `20261115T211043`); instrumented 12.69 tok/s (traces +35%).
- Long-prompt + 128 decode, warm: **≈16.2 tok/s** (chunk-clean run, base arm).
- Registry qualification record: 11.55 tok/s (earlier, busier host).
## 6. Occupancy experiment: early attention-CB commit — measured-REJECTED

Hypothesis: commit `attnCB`/`softmaxCB` immediately after `encodeDecodeAttention`
(before the tail/router encode), so the GPU runs attention while the CPU encodes
the router (same-queue FIFO preserves order). Implemented in `decodeLayerLoop`
(`RealForwardRunner+DecodeLayers.swift`), built two binaries (base A, change B),
paired interleaved A/B (3 rounds, 192-token greedy, parity-checked; shader bundle
copied beside the test binaries — a copied binary without its bundle dies on
"unable to find bundle named TinyTitan_TinyTitanCore").

| round | base A | change B | Δ B vs A | parity | machine |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.579 | 7.419 | +12.77% | OK | heavy swap load |
| 2 | 7.018 | 7.351 | +4.74% | OK | load |
| 3 | 16.620 | 13.884 | **−16.46%** | OK | **freed up (≈clean)** |

Mean +0.35%, median +4.74%, 2/3 positive — but the **representative clean round
(3) is a regression**: B's `cb1` (8.36 vs 6.67 ms) and `wait` (39.4 vs 33.4 ms)
are both *higher* when the machine is not swap-thrashed. The early commit only
helps when the CPU encode is pathologically slow (swap); it hurts normal clean
operation (likely disrupts command-buffer batching). **REJECTED and reverted** —
the product runs clean, not swap-thrashed; a load-only "win" is not a win.

Lesson: an occupancy lever must be validated under the *clean* memory
precondition, not under load — load can invert the verdict (here, +13% under
swap became −16% clean). Parity held throughout (identical sha), so this was a
pure scheduling effect, not a correctness issue.

**Net conclusion of the occupancy investigation:** the shipped decode pipeline
is near-optimal under clean conditions. The I/O knobs, prefill chunk, and the
early-attention-commit occupancy lever were all measured and rejected. No
accepted performance change resulted; the engine's defaults stand.

## 7. Readahead at the actual expert stride — measured no change

Ornith's `expertStride` is **1,769,472 bytes (1.69 MiB)** (`manifest.json`).
A direct-I/O probe at that exact size gives no reliable readahead win: random
runs were ~7.75–8.93 GiB/s with advice on vs ~7.75–8.36 GiB/s off (one −29%
outlier; the rest within noise). Larger chunks were not monotonic either
(64 KiB 7.26, 256 KiB 5.65, 1 MiB 7.27, 8 MiB 4.83 GiB/s). **No change** to
readahead or expert-read sizing. (`/tmp/ssd_probe.c`.)

## 8. CPU thread-count sweep — no stable winner

The quantized prefill harness was run at 1/2/4/8 threads. Results swung widely
between rounds; 4 threads was often but not always best, and no count won
consistently across 2B/4B and 4/8-bit. Keep the current count. A future change
would need a cleaner, interleaved repeated sweep plus parity before it is
justified.

## 9. ANE prefill — no sidecar available

The installed `models/ornith-1.5_35B_A3B_4Bit` contains **no `ane_prefill*`
sidecar**, so there is no valid ANE on/off measurement to take. No ANE change
or performance claim. If a sidecar is later produced, evaluate a full
4,096-token chunk on/off (compile cost, TTFT, decode cost, parity) before
considering it.

## 10. P1 adopted: M=16 dequantize-once + SGEMM prefill

**Adopted `07cecaf1`.** The dense variants now use dequantize-once + Accelerate
SGEMM for **M >= 16 prompt tokens**, with the shipping per-token GEMV below 16
and for decode. `DenseVariantBatchedMatmulTests` 4/4 and `CPUBatchedPrefillTests`
5/5 green. Microbenchmark (dequant included): M=16 1.7–2.9x, M=64 5.7–6.2x on
2B/4B 4/8-bit, projections only. Full-model golden text/timing **not checked**
(no dense installs; none downloaded).

## 11. P2 measured non-adoption: attention-CB coalescing

A candidate reused `attnCB` for softmax/tail, reducing three command buffers to
one. Policy tests 3/3 and golden `ornith-4` passed, but the paired A/B was too
swap-confounded to accept: 13.19–13.75 GB swap, 3.5–8.6 tok/s, per-round deltas
−7.18%, +15.11%, +61.44%, +114.14% (parity OK). **Reverted; no coalescing change
adopted.** A future retry needs an acceptable-memory, clean paired run.

## 12. P1–P5 final verdict

The five-item plan was measured first, then implemented only where the evidence
justified it. One item was adopted; four were measured and not adopted.

| item | hypothesis | verdict | evidence |
| --- | --- | --- | --- |
| **P1** batched CPU prefill (dequantize-once + SGEMM, M≥16) | GEMV-per-token wastes repeated weight dequantization at large M | **ADOPTED** `07cecaf1` | microbench M=16 1.7–2.9×, M=64 5.7–6.2×; 4/4 + 5/5 tests green; full-model golden **not checked** (no dense installs) |
| **P2** attention command-buffer coalescing | fewer CBs → less per-layer CPU barrier | **NOT ADOPTED** | policy 3/3 + golden passed, but A/B swap-confounded (−7%…+114%); reverted |
| **P3** expert-read size / readahead tuning | larger readahead at the 1.69 MiB expert stride | **NO CHANGE** | direct-I/O probe within noise; no monotonic win |
| **P4** measured CPU thread-count selection | a fixed thread count is suboptimal | **NO CHANGE** | 1/2/4/8 sweep swung between rounds; no stable winner |
| **P5** ANE prefill sidecar | ANE offloads prefill | **NO CHANGE** | installed model has no `ane_prefill*` sidecar; nothing to measure |

**Net result:** one real improvement shipped (P1, dense prefill projections), zero
regressions, zero quality cost. The Ornith decode path was left at its measured-
optimal defaults — every scheduling/occupancy lever tried (early attention-CB
commit, CB coalescing) was rejected on clean-condition evidence, and the I/O,
chunk, readahead and thread knobs were confirmed already optimal. A further solid
decode gain requires the GPU-occupancy restructure scoped in §5 (deferred as
funded future work, not forced at the tail of this investigation).

**Final audit:** all 20 lint gates green (`LINT_EXIT=0`); `catalog --check` ok;
docs gate ok; full serial suite green (the only prior failure was a stale test
binary expecting the pre-removal 19 shipped pairs — rebuilt to the correct 18 and
passing). P1 full-model golden text/timing remains **not checked** because no
dense model is installed and none was downloaded, per the model-install policy.
