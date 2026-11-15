# Post-7.0 Ornith 1.5 35B-A3B 4-bit — performance investigation

Status: in progress (2026-11-16). Subject: the shipped server path, which runs the
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

## 5. Remaining lever — GPU occupancy (item 5, in progress)

The only measured headroom is the 47.8% GPU occupancy / per-layer CPU barrier.
Candidate parity-safe scheduling changes (ranked): pre-encode the next layer's
attention command buffer ahead of the routed-MoE CPU work (requires double-buffered
router scratch so the pending routed CB is not clobbered); reduce `cb1` CPU encode
cost; check the fused greedy head branch. High-risk alternatives (indirect command
buffers / GPU-driven expert dispatch) rebuild the CPU cache-plan authority and are
deferred. Each candidate is gated by golden `ornith-4` parity + a clean paired
benchmark.

## Clean baselines (for the final paired comparison)

- Production profile (12 short prompts, cache-warm, 12/12 verified): **19.65 tok/s**
  footer decode (run `20261115T211043`); instrumented 12.69 tok/s (traces +35%).
- Long-prompt + 128 decode, warm: **≈16.2 tok/s** (chunk-clean run, base arm).
- Registry qualification record: 11.55 tok/s (earlier, busier host).