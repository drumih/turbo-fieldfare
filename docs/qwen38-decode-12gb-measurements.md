# Qwen3.8 4-bit decode at 12 GB — measurements

Stage-0 record for `docs/plan-qwen38-decode-12gb.md`. Machine: MacBook M3
24 GB, macOS 26, Xcode 27 / Swift 6.4, release build, bounded pread reader.
Every number below was measured 2026-10-11; commands and exit codes included.

## 1. The capture

Model run (window free: `pgrep` guard empty, `memory_pressure -Q` 77% free,
install `models/qwen3.8-flash-next_125B_A6B_4Bit`):

```
TINYTITAN_ROUTE_TRACE=/tmp/qwen38-route.trace \
  .build/release/TinyTitanCLI \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing, …" \
  --max-new 384
# exit 0; footer: prefill 30 tok / 7.17 s, decode 88.24 s, tok/s=4.352
```

Trace: 19,824 lines = 413 positions (30 prefill + 383 decode) × 48 layers ×
top-10 experts. Committed as
`benchmark/expert-cache/route-trace-qwen38-4bit-20261011.txt.gz` so every
figure below re-runs without a model:

```
gunzip -c benchmark/expert-cache/route-trace-qwen38-4bit-20261011.txt.gz > /tmp/qwen38-route.trace
python3 benchmark/flat_pair_cache_sim.py /tmp/qwen38-route.trace --slots-per-layer 64 --layers 48
```

The measured 4.352 tok/s reproduces the 4.0–4.7 baseline.

## 2. Control calibration (does the simulator reproduce the engine?)

Simulator uniform-LFU at 64 slots/layer over **decode-only** events (first 30
positions dropped): **65.7%** hit. The engine measures 71.7–78.1% at this
budget; the gap is prefetch adoption, which the simulator does not model —
the depth-1 predictor turns would-be misses into hits, worth ≈6–12 points
here. Calibration through the engine's own measured point (≈55 ms exposed I/O
at ≈78% hit, i.e. ≈40.4k misses of 183,840 decode expert-references) puts
uniform-64's 62,976 misses at ≈86 ms exposed ≈ 211 ms/token ≈ **4.7 tok/s** —
against 4.352 measured. The control reproduces within run-to-run noise once
calibrated; the raw 3.6 GB/s projection in the CLI output understates exposed
time ~3x and its tok/s figures are not the ones to quote.

## 3. Policy comparison at the fixed 12 GB budget (3,072 pairs)

Decode-only, same budget:

| policy | hit rate |
| --- | --- |
| uniform (today's semantics) | 65.7% |
| flat (layer, expert) pool | 65.9% |
| heat-proportional per-layer | 65.5% |

**Flat and heat are within noise of the control (+0.2 / −0.2 points).** B1's
cache-semantics hypothesis is dead on this trace, and the reason is
structural, not a tuning miss:

- Per-layer demand is nearly uniform: distinct experts per layer 146 / 259
  (median) / 409, sum 12,544 pairs over the decode. There is no hot layer to
  favor, so heat allocation has nothing to exploit, and a flat pool's extra
  freedom buys nothing a per-layer pool cannot.
- **The working set outruns the cache at any window**: a sliding window of
  16 decode tokens touches a median of **3,234** distinct (layer, expert)
  pairs — more than the entire 3,072-slot budget; 32 tokens ≈ 4,906; 64
  tokens ≈ 6,811. The cache is in constant churn; hit rate is set by
  capacity vs window, not by allocation cleverness.

## 4. Ceilings at this budget

| quantity | value |
| --- | --- |
| compulsory-miss bound (infinite cache) | 93.2% |
| Belady oracle @ 3,072 pairs (8.28 GiB) | **85.4%** |
| Belady @ 4,608 | 89.8% |
| Belady @ 6,144 (16.6 GiB — swaps) | 92.1% |
| Belady @ 9,216 | 93.2% (saturated) |
| uniform LFU @ 96 / 128 slots | 76.5% / 83.5% |

Best case for *any* online policy at 8.28 GiB sits between uniform's 65.7%
and Belady's 85.4%. Even at the full Belady 85.4%, calibrated exposure is
≈36 ms/token → ≈6.2 tok/s — short of the 7 target, and Belady is unreachable.

## 5. Verdict (per the plan's §2 acceptance rule)

**B1 is dead as modeled: the plan jumps to §4/§5 (PLE attribution, split
reads) and the §6 merge decision moves to the operator.** No cache-semantics
change reaches 7 tok/s at 12 GB on this workload; the levers that remain are
shrinking the corpus (expert merging 512→256–320) or the bytes per miss
(split gate+up/down reads, hc-factored QKV, lm_head 4-bit) — the first is
numerics-changing and needs sign-off, the second is byte-identical but
bounded (≈+0.3–0.5 tok/s at today's hit rate).

The trace also kills the §3 LFU-warmth sub-hypothesis for this shape: with a
working set that exceeds the cache within 16 tokens, cross-request warmth
evicts within a handful of tokens regardless.

## 5b. B2 (PLE decode-time traffic) — closed by arithmetic, no model run

The plan's §4 asked for a capture; the manifest answers it. From
`models/qwen3.8-flash-next_125B_A6B_4Bit/ple_constants.json`:
`ple_head_dim=160` → `rowBytes = 160 × 2 = 320`, `ngram_size=3`,
`ple_n_heads=16` → `gatherPLERows`
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+PLE.swift:13-26`)
reads **16 rows × 320 B = 5,120 B per token** from `ngram_table.bin`
(102.4 GB). Against 293–368 MB of expert traffic per token that is 0.0015%
— 1.4 µs at the 3.6 GB/s wall. The comment on the gather already said
"three orders of magnitude under"; the manifest pins it. The reader uses an
ordinary `pread` (page-cached), so those 5 KiB are usually not even device
reads. **B2 is not material; closed with these numbers.**

## 6. Provenance

- Simulator: `benchmark/flat_pair_cache_sim.py` (22 tests, CI-registered).
- Belady and window analyses: ad-hoc scripts over the committed trace; the
  Belady implementation used for §4 is the corrected form (evict the
  farthest-next-use resident unconditionally; a first version that skipped
  the insert when the newcomer's next use was nearer produced the nonsense
  46.7% and was discarded before any conclusion was drawn from it).
- The trace is one workload (one essay prompt, 383 decode tokens). A second
  capture over a different workload is cheap while the window stays free and
  would strengthen §5's verdict; the working-set arithmetic above is the
  load-bearing fact and is a property of the routing distribution, not the
  prompt.

## 5c. B3 (split blob reads + early miss fixup) — measured, accepted

Reader split (`fetch_ranges`), streamer `phase1Bytes` from the manifest's
`down` offset, first-stage publication on `ExpertLoadOperation`, and the
runner's early miss fixup: phase 1 over the miss slots is encoded and
committed as soon as the front range (gate+up) is resident and runs on the
GPU while the tail (down) fetch is still in flight; `routedCB` carries
phase 2 only. Committed at `87d072c`, `af26fc5`, `c4a431c`, `53d7b8d`.

Golden gate: `tools/golden-baseline.sh --check qwen38-4` — output identical
to baseline on the full chain (model run, window free). Model-free suites
and `tools/lint.sh` all green.

A/B, same binary pair, same prompt, `--max-new 384`, `TINYTITAN_RUNNER_STATS=1`
(two pairs, back to back; base = pre-3b `c4a431c`, treatment = `53d7b8d`):

| arm | tok/s | hit rate | io_hidden_pct | wait_ms/token |
| --- | ---: | ---: | ---: | ---: |
| base, run 1 | 4.162 | 74.67% | 14.09 | 86.55 |
| base, run 2 | 4.197 | 74.97% | 14.31 | 85.90 |
| B3, run 1 | 4.286 | 74.35% | 99.90 | 76.76 |
| B3, run 2 | 4.286 | 74.05% | 99.83 | 75.97 |

The plan's falsifier (§5): hit rate held equal (74.8% → 74.2%, inside the
pair's spread), and tok/s **and** `io_hidden_pct` rose together — satisfied.
The `io_hidden_pct` jump is the accounting seeing the fixup CB complete
inside the I/O window; the wall-clock number is the honest one: **+2.5%
(4.180 → 4.286 mean), ≈ +0.11 tok/s**, with per-layer wait down 11.5%
(86.2 → 76.4 ms). Below the §5 projection (+0.3–0.5 tok/s) but positive and
mechanistically explained: the fixup fills only the GPU-idle slice of the
tail fetch, and the front fetch's window remains unfilled at this hit rate.

Text outputs were not compared across arms: the CLI's default sampling is
temperature 1.0, so run-to-run text differs inside each arm; the golden
gate is the numerics verdict and it passed.

## 5d. Reader throughput vs the device wall — closed (2026-10-11)

The one remaining byte-identical candidate was raising the reader's raw
throughput. Premise checked before touching code: what does this SSD
actually deliver on the reader's exact access pattern — 2.64 MiB
expert-stride-aligned preads scattered over 48 layer files? A thread-pool
probe (`stride = 2,768,896`, random file + expert-aligned offset per read,
~2 GB per configuration, 48 layer files ≈ 68 GB so the page cache never
answers):

| workers | GB/s |
| ---: | ---: |
| 1 | 2.01 |
| 2 | 2.84 |
| 4 | 3.57 |
| 8 | 3.71 |
| 16 | 3.69 |

Saturation is ~3.7 GB/s at 8–16 workers; the engine's 4-worker reader
(3.57 GB/s in this probe, matching its in-engine effective throughput) is
already at the wall. The sequential contrast kills the alternative shape:
`dd` over a whole layer file measures 3.15 GB/s (`layer_47.bin`) and
2.61 GB/s (`layer_31.bin`) — sequential is *slower* than the reader's
strided parallelism, so "read more sequentially" is not a lever either.
The earlier assumption of a ~7 GB/s M3 SSD was wrong for this access
pattern; scattered 2.64 MiB reads are the binding constraint.

**The reader-throughput lever is closed with these numbers.** Combined
with B1 (cache policy dead) and B2 (PLE = 0.0015%), no byte-identical
lever of material size remains; what is left is §6, gated on an operator
decision.

## 5e. The ANE/CPU survey — every remaining co-executor closed or bounded (2026-10-11)

The question: with the SSD at its wall and the GPU's per-token busy time
~103 ms, can the ANE or the CPU cores take enough work off the GPU to move
tok/s? Re-surveyed against the post-B3 profile (4.286 tok/s ≈ 233 ms/token,
wait 76 ms, busy ~103 ms, `head_logits` 13.49 ms/call) before building
anything; what was already measured in `docs/cpu-coexecution-plan.md` stays
closed, and the two candidates that record never tested at *this* profile
got measured now.

Closed by earlier measurement (do not reopen):

- **CPU co-execution of experts:** GPU busy +44.9%, tok/s −22.6%; no split
  ratio won (`cpu-coexecution-plan.md:601`).
- **ANE concurrent load:** GPU busy +89.3%, tok/s −45.4% — the ANE is a
  *faster bus consumer* on the same die, and this profile is bus-bound
  (`cpu-coexecution-plan.md:692`).
- **ANE expert ops:** the GPU runs them 2.8x faster; the ANE collapses
  below 1 MiB of weights.
- **CPU spinning during I/O wait:** −15%.
- **Fused greedy head:** −3% (benched, not shipped).

Closed by arithmetic against the post-B3 profile:

- **CPU head GEMV:** the head is 606 MiB of 8-bit weights; at the M3's
  ~17 GiB/s CPU memory bandwidth that is ~35 ms of reads against the GPU's
  13.49 ms measured. The CPU is 2.6x slower on the one op it could
  plausibly take, before counting any transfer.
- **CPU or ANE expert work during the I/O window:** dead after B3, not
  before it. The early miss fixup already runs phase-1 compute *while* the
  tail fetch is in flight and `io_hidden_pct` is 99.90 (§5c run 1) — the
  overlap those co-executors were meant to supply is already banked on the
  GPU. What remains of the token is GPU-busy compute at 233 ms cadence,
  and both co-executors measurably make *that* slower (the two closure
  rows above).
- **ANE at all, at decode cadence:** the token's ~40 ANE-eligible ops at
  8–12 ms of per-handoff cost is 320–480 ms of handoff per 233 ms token.

**New measurement — the head's in-engine cost is cadence, not the kernel.**
The head GEMV is the largest per-call GPU cost in the profile (13.49
ms/call, 606 MiB stored → 46.6 GB/s) and the only op never probed at the
qwen38 shape. `TinyTitanBench head_affine8` now has
`TINYTITAN_BENCH_HEAD_CADENCE_MS` (committed `2c3fc36`): one launch per
command buffer with a wall-clock gap, mimicking decode's once-per-token
cadence. The identical kernel, identical buffers, identical PSO:

| arm | per launch |
| --- | ---: |
| back-to-back steady state | 9.42 ms (71.7 GB/s) |
| 50 ms gaps | 12.73 ms |
| 200 ms gaps | 15.49 ms |
| in-engine (`head_logits`) | 13.49 ms |

The in-engine 13.49 sits between the 50 and 200 ms cadence arms: the ~4 ms
delta over steady state is GPU-state (DVFS/power between launches), not
kernel inefficiency, not memory layout, not the mmap. Recoverable bound if
the state could somehow be kept warm: ~4 ms/token ≈ +1.8% — and nothing in
the engine controls GPU clock state, so this is a measured bound, not an
actionable lever.

**New measurement — the head on CPU and ANE, at the exact shape.** A
synthetic coremltools probe (8-bit kmeans-palettised `linear` at 248320 x
2560, batch 1, slope harness N=2 vs N=4, median of 40 predicts) measured
the head op on every unit pool coremltools can address:

| arm | per op |
| --- | ---: |
| ALL / CPU_AND_GPU / CPU_ONLY / CPU_AND_NE (N=2 graph) | 78.4–79.1 ms whole-graph, no arm differs |
| slope per op (any arm) | 38.95 ms → 16.3 GB/s of stored bytes |
| GPU in-engine (`head_logits`) | 13.49 ms → 46.6 GB/s |

Two verdicts, one honest caveat. (1) **CPU head: measured dead** — 38.95
ms/op is 2.9x the GPU's 13.49 ms, confirming the ~35 ms bandwidth
arithmetic above with a real number. (2) **ANE: no demonstrable
engagement.** The arm contrast did not discriminate (ALL vs CPU_AND_GPU
identical at both N), and even with the GPU excluded (`CPU_AND_NE`,
GPU_ONLY's complement — the only setting that forces an ANE-or-CPU choice)
the rate is unchanged and matches CPU bandwidth — so the ANE either did
not take this op (8-bit palettised batch-1 linear not ANE-eligible in this
form) or fell back silently; `~/.venvs/coreml-py311/bin/python` ran both
probes. The caveat: a non-engagement is not a measured ANE *rate*, so the
closing statement leans on the earlier record — ANE batch-1 is bounded at
the same ~64 GB/s wall (the head alone would then save ~4 ms/token ≈
+1.8% at best, before handoff cost and a mandatory golden re-capture for
the ANE's different accumulation numerics), and every measured ANE
co-execution so far made the token slower, not faster.

**The survey is closed.** Every ANE/CPU involvement option is now either
measured dead at this profile or bounded below 2% by a mechanism nothing
in the engine controls; decode tok/s at 12 GB remains SSD-bound, and the
open levers are the operator decisions in §6.

## 5f. B4 algebra check — the HC-factored QKV premise is false (2026-10-12)

Plan §6 item 1 ordered the algebra check before anything else: `W_eff =
W·B` precomputed offline is only valid if the hyper-connection read map
`normed → blockInput` equals a fixed linear `B @ normed`, bit-exact modulo
float order. The runtime (`HyperConnection.swift`) computes
`h = silu(Wdown @ normed/4)`, `mix = sigmoid(Wup @ h)`,
`input = mean_st(mix[st] ⊙ normed[st])` — so the map is linear only if the
gates are degenerate. The check ran over the real install's dequantized
weights (`tools/ssdai_reader.py`, layers 0/1/2/47, attn + mlp arms and the
model-level mixer):

| arm | \|Wdown\|max | \|Wup\|max | mix std/mean | linearity viol. | scale viol. |
| --- | ---: | ---: | ---: | ---: | ---: |
| layers.0.attn | 1.28 | 3.94 | 0.24 | 0.19 | 0.33 |
| layers.1.attn | 6.47 | 2.94 | 0.41 | 0.44 | 0.45 |
| layers.2.attn | 1.84 | 4.50 | 0.52 | 0.36 | 0.30 |
| layers.47.attn | 1.06 | 2.89 | 0.34 | 0.40 | 0.31 |
| layers.0.mlp | 2.78 | 8.56 | 0.44 | 0.29 | 0.23 |
| layers.1.mlp | 2.34 | 4.49 | 0.39 | 0.26 | 0.33 |
| layers.2.mlp | 4.13 | 6.19 | 0.49 | 0.32 | 0.23 |
| layers.47.mlp | 0.94 | 2.95 | 0.37 | 0.27 | 0.24 |
| model mixer | 1.35 | 4.16 | 0.71 | 0.39 | 0.36 |

The gates are live (weight maxima O(1)–O(10), mix moves 24–71% of its mean
across random tokens) and the linearity violation is 0.19–0.44 relative on
every arm — against a float32 noise floor below 1e-6. **No fixed `B`
exists; the B4 premise is false and the lever is closed at the algebra
gate.** No golden re-capture was spent, no engine code was touched; this
is the outcome the plan's check-first ordering exists to catch cheaply.
The plan's §6 item 1 and status header record the closure.

## 5g. gate|up sub-split — premise measured false (2026-10-12)

The mechanism hypothesis after §5c: at 74% hit rate × 10 routed experts ×
48 layers, a token misses ~2.6 experts per layer, so the B3 front phase
issues only ~2.6 ranges of 1.76 MiB to the 4-worker reader — possibly
undersaturated — while splitting gate|up at the real blob seam (up's
offset, 921,600 B) would double the range count. Premise checked with the
§5d thread-pool probe at the real geometry (F_NOCACHE preads at
expert-aligned offsets over the 48 real layer files, random file per job,
~1.5 GB per arm, 4-bit front = 1,843,200 B, blob = 2,768,896 B):

| arm | GB/s |
| --- | ---: |
| front as-is: 3 concurrent × 1.76 MiB | 3.25 |
| front as-is: 4 concurrent × 1.76 MiB | 3.36 |
| sub-split: 3 × 2 × 0.88 MiB (6 concurrent) | 3.22 |
| sub-split: 3 × 2 × 0.88 MiB (4 workers) | 3.33 |
| control: 6 × 0.88 MiB | 3.36 |
| control: 3 × 2.64 MiB whole blob | 3.38 |
| tail-split blob: 3 × 2 × 1.32 MiB | 3.32 |

Every arm lands in 3.2–3.4 GB/s, inside arm-to-arm noise, and the
sub-split arms are not faster than the as-is front. **Per-pread size
between 0.88 MiB and 2.64 MiB does not move this SSD's throughput, and the
front phase at ~3 concurrent ranges is already at the same wall as
everything else — the "undersaturated front" premise is false and the
sub-split lever is closed at its probe gate.** A pinned single-file run
(layer_00 only) reproduced the same verdict (2.92–3.17 GB/s, sub-split
slowest), so the conclusion is not an artifact of the scatter. No engine
code was touched, no golden gate was spent. Combined with §5d, the reader
is closed in both directions: more workers buy nothing (§5d) and finer
ranges buy nothing (this section).

## 5h. Expert merging — the 512 experts are mutually orthogonal (2026-10-12)

Plan §6 item 2 was the last lever with a big ceiling: merge 512 experts to
256–320 so the decode working set fits the 3,072-pair cache. The operator
set the constraint first: **no merge that makes the model dumber.** The
study therefore asks whether the experts cluster at all, on the real
install's dequantized 4-bit weights (`PackedExperts`, layers 0/6/12/24/36/
47, all 512 experts, gate+up+down). Method: pairwise relative Frobenius
distance `d(i,j) = ||Wi−Wj||_F / sqrt(||Wi||_F ||Wj||_F)` — for
uncorrelated equal-norm matrices this is √2; then a greedy closest-pair
matching (the merge schedule), then the functional cost: SwiGLU output of
the weight-averaged expert vs both originals over unit-RMS gaussian inputs
(RMSNorm makes the hidden state unit-RMS; real activations would need a
model run), plus the milder "drop expert i, route its tokens to j"
contrast.

Per layer, 131,072 pairs each; pooled medians across the six layers:

| quantity | median | p90 | max |
| --- | ---: | ---: | ---: |
| all-pair relative distance | 1.4143 | 1.4159 | 1.5282 |
| matched-pair (closest 256) distance | 1.4131 | 1.4135 | 1.4438 |
| merge→256: merged-expert output error | 0.947 | 0.971 | 1.162 |
| merge→320 (closest 192 pairs) error | 0.946 | 0.963 | 1.123 |
| drop contrast (route i's token to j) | 1.411 | 1.419 | 1.470 |

The verdict is structural, not marginal. **The experts are pairwise
orthogonal: the all-pair distance is √2 = 1.4142 with almost no spread,
and even the closest matched pairs (1.4131) sit a rounding error below
random.** There is no cluster structure to merge. The functional numbers
agree: weight-averaging the *closest* pair still destroys ~95% of the
routed expert's output, and the drop contrast is worse than the output
itself (1.41×). Merging 512→256–320 would make the model measurably
dumber — it fails the operator's constraint at the study gate, and the
quality-eval-on-a-merged-install step is never reached because there is
no merge schedule worth building.

Caveats, stated plainly: the study measures weight geometry and
gaussian-input function, not task quality; and a merged install would add
re-quantization error on top. Both push the same direction — neither could
rescue a √2 distance. With this, §6 item 2 is closed; the remaining
§6 items are lm_head 4-bit (B5, quality-gated) and the MTP re-verdict
(B6). The study's script was a one-off (`/tmp/merge_distance.py`); its
pooled-summary print crashed on a scalar shape after every per-layer row
had printed, so the per-layer table above is complete.

## 5i. B5 offline gate — lm_head 4-bit is not near-lossless (2026-10-12)

Plan §6 item 3 hypothesized that 4-bit group-64 affine might be
near-lossless for the vocabulary head. Measured on **real activations**:
one decode run (175 positions, Hamlet prompt, `--max-new 192`) with
`TINYTITAN_ACT_DUMP` captured the true pre-head hidden state (`mixer_out`)
and the engine's own 8-bit logits per position; the study
(`/tmp/b5_head_4bit.py`, model-free afterwards) simulated the repack's
4-bit affine scheme (asymmetric uint4, groups of 64 along the input dim,
BF16-rounded scale and bias) on the real 8-bit head weights and compared
logits.

Reference sanity first: the offline fp32 recomputation of the 8-bit head
matches the engine's dumped logits to 0.00021 relative — the proxy is
sound, so all the error below is the 4-bit step itself.

| metric (175 real decode positions) | value |
| --- | ---: |
| relative logit error (4-bit vs 8-bit) | 0.079 |
| greedy top-1 flips | 2 / 175 (1.1%) |
| positions where err@top1 exceeds the top-1/top-2 margin | 6 / 175 |
| ΔNLL on the generated token (perplexity multiplier) | mean +0.005 nats on greedy, p90 +0.59 on generated |
| KL(p8‖p4), T=1 | 0.0065 mean, 0.020 p90 |
| entropy shift | 0.345 → 0.334 |

**The "near-lossless" premise is false: 4-bit injects 7.9% relative logit
noise — ~370x the 8-bit reference error — and flips the greedy argmax on
1.1% of positions.** Over a 300-token greedy generation that is a ~96%
chance of at least one divergence, and every divergence is a visible
quality event (the golden text changes, which is why the plan priced a
golden re-capture). The perplexity-side damage is small (≤0.5% on greedy)
and the KL is modest, so this is not obviously a dumber model — but it is
measurably a *different* one, against an operator constraint that says no
quality reduction.

**Decision: declined at the operator decision (2026-10-12)** — "quality
loss cannot be accepted." Advancing B5 would have meant a repack change
(the engine already supports a 4-bit head end to end —
`lmHeadWeightBits` resolves per-role and the decode/prefill head kernels
are width-keyed, so no kernel work), a fresh ~68 GB install from the
pinned source checkpoint, the full quality panel (perplexity + prompt
panel), a golden re-capture, and the runner-stats A/B for what B5 buys
(the head GEMV read halves, ~640 → ~320 MiB/token ≈ +2-3% ceiling at the
67.8 GB/s the plan measured). The offline gate's job was to kill a bad
premise before that spend; it did: the premise is false (7.9% logit
noise, 1.1% greedy flips), and the operator declined any numerics
change regardless of the small perplexity damage. No repack build, no
install, no quality panel. This closes the §6 list and with it the
decode-12GB campaign, standing at the measured 4.286 tok/s.

## 5j. B6 re-affirmed closed by its own gate (2026-10-12)

The MTP re-verdict is conditional on the hit rate rising; with B1, the
reader (§5d/§5g), B4, §5h and the sub-split all closed, nothing in the
campaign moves it off 74.2–74.8%. The condition never fires; B6 stays
closed without a re-measurement.


## 5k. The CPU L1/L2 cache is not an accelerator for this engine (2026-10-13)

Asked whether the CPU cores' L1 (192+128 KB per P-core) or 16 MB P-cluster
L2 could be used as an accelerator for a critical area. Attribution from
the code, no model run needed:

- **No tensor math runs on the CPU in the main decode loop.** Router,
  GEMVs, attention, expert FFN and sampling are all Metal kernels; the
  `Sampler` touches the logits on the host only when a repetition/presence
  penalty or a structured constraint is active, which the default path is
  not (`sources/TinyTitan/Runtime/Generation/Sampler.swift:84-106`).
- **The one real CPU cost per token is dispatch latency, not data.** The
  profiled "≈20 ms/token turnaround" is GPU idle during readback, plan,
  pin, encode and commit (`docs/qwen38-decode-profile-2026-09-05.md`
  §C) — command-buffer scheduling, on which a cache has no effect. The
  early-hits work (profile doc §12) already shipped most of it down.
- **The CPU data the decode loop does touch is already trivially
  resident**: kernel arguments, expert plans, PLE bookkeeping — kilobytes
  against a 128 KB L1d.
- **The only CPU tensor machinery in the tree is the closed side
  engine.** `CPUEngine/` + `Int8AffineGEMV`/`CPUExpertFFN` exist for CPU
  co-execution, measured at **−22.6%** tok/s (`cpu-coexecution-plan.md`),
  and CPU spinning during I/O wait measured −15%. This profile is
  bus-bound: moving compute onto the CPU cores turns them into another
  memory-bus consumer competing with the SSD streaming that is the wall.

Verdict: the cache-residency lever prices at **~0%** here. The L1/L2
option only pays when a kernel repeatedly re-reads a working set that
fits the cache, and the only such compute available would have to be
*added* to the CPU — which is the measured-loss co-execution lever. Not
re-openable on the evidence.

## 5l. Low Power Mode re-baseline and turnaround decomposition (2026-10-13)

A runner-stats re-baseline captured **inside Low Power Mode** (`pmset -g`
`lowpowermode 1`, on battery), at the operator's explicit direction after
an LPM throttle was found to have invalidated an earlier attempt to
re-measure. **Scope caveat: these numbers are valid for comparison
against each other and against the code's structure only — not against
the 4.286 tok/s campaign record or any mains-power number.** Method is
the §5c CLI command, run twice back-to-back (run 1 warms, run 2 is the
measured arm):

| | prefill | decode | tok/s | ms/token |
| --- | --- | --- | ---: | ---: |
| LPM run 1 (warm) | 58 tok / 8.53 s | 384 tok / 112.59 s | **3.411** | 293.2 |
| LPM run 2 (measured) | 58 tok / 8.35 s | 384 tok / 128.28 s | **2.993** | 334.1 |
| campaign B3 arm, mains, for orientation only | — | — | 4.286 | 233.3 |

Runner-stats decomposition (all values ms/token):

| segment | run 1 | run 2 |
| --- | ---: | ---: |
| cb1 (encode) | 8.28 | 12.77 |
| io (miss I/O) | 106.86 | 117.98 |
| cb2 | 1.78 | 2.99 |
| rdadvise (issue) | 17.53 | 26.81 |
| wait (GPU + hidden I/O) | 118.96 | 124.38 |
| **body** | **262.13** | **299.48** |
| head | 21.54 | 22.91 |
| cache_plan | 5.37 | 8.65 |
| io_completion_to_fixup | 2.99 | 5.15 |
| gather | 2.96 | 3.74 |
| loop_sample | 1.92 | 1.92 |
| loop_other | 1.97 | 1.98 |
| embed | 0.70 | 0.90 |
| expert hit rate | 76.0% | 74.4% |
| io_hidden | 99.79% | 99.09% |
| expert load p50/p95/p99 | 4 / 8 / 16 | 4 / 8 / 16 |
| rdadvise calls / MiB per token | 102.5 / 272.4 | 110.1 / 334.0 |
| prefetch issued / adopted per token | 43.5 / 23.6 | 44.0 / 26.1 |

Findings:

- **The token's structure is unchanged under LPM.** Hit rate (74–76%),
  I/O hidden fraction (99+%), expert load percentiles and the segment
  ordering all match the campaign's B3 arm; the throttle scales every
  segment roughly together — clock throttling, not a new bottleneck.
- **LPM run-to-run spread is ±14% on decode time** (112.6 vs 128.3 s)
  while every CPU-side dispatch segment moved 30–50% between runs
  (rdadvise 17.5→26.8, cb1 8.3→12.8, plan 5.4→8.6). **This baseline
  cannot resolve a single-% candidate**: the noise floor is an order of
  magnitude above one percent. It prices structure, not micro-levers.
- **CPU-side dispatch segments total ≈44–66 ms/token (15–20% of the
  token)**, the same magnitude as the mains-power turnaround finding
  (profile doc §C). rdadvise issuing is the largest single segment, but
  the §8 closure covers the *policy* A/B (`7cc8b5e`: default beats off
  by ~10.6% at 4-bit, neutral on M5/8-bit; `db1c102`: off costs 8.5%,
  adaptive/bounded no win) — not the issue count. A call-count reduction
  (~105 `F_RDADVISE` syscalls/token for ~300 MiB advised) is the one
  uncovered seam this baseline names, and it is unpriceable at LPM's
  noise floor; it would need a mains window and an A/B to go further.

Verdict: no new lever opened. The LPM baseline confirms the delivered
build's structure and re-identifies the turnaround as the remaining
non-GPU cost; any single-% work against it requires mains power.

## 5m. rdadvise on/off on this family — measured, closure re-affirmed (2026-10-14)

§5l named the rdadvise issue count (~105 `F_RDADVISE` syscalls/token) as the
one seam §8's closure did not cover, and noted the closure's provenance is
ornith-era (`7cc8b5e`, `db1c102`) — this family at 12G had never been
arm'd. Mains window, §5c CLI command, `--rdadvise default` vs
`--rdadvise off`, two back-to-back runs per arm, `TINYTITAN_RUNNER_STATS=1`,
`--max-new 384`, prompt "Write a detailed essay about the history of
computing." (10 prefill tokens):

| arm | run | tok/s | hit rate | io_hidden_pct | wait_ms/token | rdadvise_ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 1 | 4.313 | 77.79% | 99.84 | 89.02 | 11.60 |
| default | 2 | 4.805 | 77.74% | 99.93 | 74.83 | 9.72 |
| off | 1 | 4.380 | 77.26% | 84.26 | 81.17 | 0.00 |
| off | 2 | 4.538 | 77.53% | 84.90 | 79.42 | 0.00 |

Findings:

- **The mechanism is confirmed and the two effects roughly cancel.** Off
  removes the 9.7–11.6 ms/token issue cost — and io_hidden falls from
  99.8–99.9% to 84.3–84.9%, i.e. ~15% of the ~100–120 ms/token miss I/O
  stops hiding behind GPU work. Wall-clock: default mean 4.559 vs off
  mean 4.459 (+2.2% for default), but the run-1 pairing went the other
  way (4.313 vs 4.380). The difference is inside the run spread — the
  default arm's own pair spans 11% (4.313 → 4.805).
- **The ornith-era "off costs 8.5%" does not transfer to this family at
  this budget.** Here the policy is worth ≤2%, not 8.5%. The §8 closure
  ("family-dependent") is re-affirmed on measurement, and the shipped
  `default` policy stays: it is never worse and its hiding is real.
- **Issue-count coalescing prices at a theoretical ceiling of the 10–12
  ms/token it could recover while keeping the hiding — ≈2–3%, single-%
  territory**, below what a 384-token single run can resolve here. Parked
  as unproven; an A/B would need many more repeats per arm than four.
- **Warmth keeps climbing across back-to-back runs** (4.313 → 4.805
  inside one arm), so short single-run A/Bs on this model carry a warmth
  confound the campaign's more stable pairs did not show. Today's
  no-route-trace runs also reach 4.805, above the 4.286 record — that
  record was captured with `TINYTITAN_ROUTE_TRACE` writing ~19,824 lines
  to /tmp during the run; the trace capture itself is part of the gap.
  The 4.286 campaign number stands as recorded; a no-trace run is a
  lighter-instrumented environment, not a faster build.

Verdict: no lever opens. Default rdadvise stays; the issue-count seam is
priced at ≤2–3% and parked.

## 5n. Context length at 12G — decode pays ~20% for a long prompt (2026-10-14)

The wiki-derived recommendation that was not already closed: measure decode
at more than one context length. Run with `benchmark/tinytitan_longctx.py`,
qwen3.8 4-bit, `TINYTITAN_EXPERT_CACHE_BUDGET=12` (the harness passes it to the
launcher as `--ram 12`, which forwards `--ram-budget 12G` to the server), mains
power, build `d26d97a`, server v5.19 on
127.0.0.1:8091. One generation per cell, driver cells in order A, B, C:

| cell | prompt | max_new | prefill s | decode tok/s | finish |
| --- | --- | --- | ---: | ---: | --- |
| A: long+128 | 1,697 tok | 128 | 74.4 | **3.962** | length |
| B: short+1024 | 22 tok | 1024 | 4.4 | **4.271** | length |
| C: long+1024 | 1,697 tok | 1024 | 67.8 | **3.423** | stop @ 549 |

Runner stats, per-token, B vs C (the two 1024-token decodes):

| metric | B: short | C: long |
| --- | ---: | ---: |
| expert_hit_rate | 0.7588 | 0.6824 |
| expert_load p95 | 8 ms | 16 ms |
| wait_ms | 84.0 | 95.6 |
| cache_plan_ms | 3.68 | 5.64 |
| io_hidden_pct | 99.88 | 99.63 |
| rdadvise MiB/token | 300.2 | 191.7 |

Findings:

- **A long prompt costs ~20% of decode throughput at this budget**
  (4.271 → 3.423 on the same 1024-token decode length). The mechanism is
  the §B/§5n cache mechanism, not attention: the expert hit rate falls
  0.759 → 0.682 and the expert-load p95 doubles (8 → 16 ms) as the 96-slot
  per-(layer,expert) cache turns over against a larger live working set.
  I/O stays almost fully hidden (99.6–99.9%) — the loss surfaces as GPU
  wait, not as exposed I/O.
- **Prefill is ~40 ms/token at this context** (67.8–74.4 s for 1,697
  tokens) and warms ~9% on the second long prompt of the session
  (74.4 → 67.8), the same warmth climb §5m saw.
- **Cell C stopped at 549 tokens on EOS** — greedy on the long prompt ends
  its essay there; the 3.423 figure is measured over those 549 tokens, not
  truncated at 1024.
- Single run per cell: these are first-measurement figures, not settled
  means. But the direction is structural (hit rate is a counted
  quantity, not a clock reading), and the cache-churn mechanism is the
  same one §5/Stage-0 priced: capacity, not policy, bounds the long case.

Verdict: the wiki proposal's premise is confirmed — context length is a
real decode variable at 12G, worth ~20% between short and ~1.7K-token
prompt. It does not open a lever: raising cache capacity at this budget
is §8-closed (slots_128 swaps at 68% cost; the flat-pair cache is at
capacity by design), and no routing or policy change recovers the hit
rate the larger working set spends. The number is recorded as the
long-context figure future proposals have to argue against.

## 5o. Budget × context sweep — the working-set cliff, and what actually sits on it (2026-10-14)

Two questions in one protocol. The sweep question (#147): does any
(cache slots × max-context) combination beat the derived default on this
machine? The regression question (#152): v5.0 shipped qwen38 at a 12 GiB
budget — 96 slots — and measured 6.957 tok/s (wiki Changelog);
`808cea3` then clamped the budget to a third of physical memory, which
is 8 GiB — 64 slots — on this 24 GB machine, and the whole campaign ran
there. Did the clamp cost the configuration?

Protocol: release build `33b3a06`, CLI (no budget flag: the CLI takes only
`--expert-cache-slots`, so the derivation picks the slots the server would
pick), essay prompt, `--max-new 384`,
`TINYTITAN_RUNNER_STATS=1`, mains power. Mirrored arms, and the mirror
of the decisive arm was run after two *healthy* arms so a predecessor's
swap debt cannot be the explanation. 129.6 MB per slot per layer group
(expert stride × 48 layers), so 64 slots is 7.92 GiB, 96 is 12.15,
112 is 14.51, 128 is 16.58, 160 is 20.74; the resident floor is 3.72 GiB.

| arm | slots | max-context | decode tok/s (pair) |
| --- | ---: | ---: | --- |
| base | 64 (derived) | 262144 | **4.551 / 4.481** |
| c32 | 64 | 32768 | 4.543 / 4.175 |
| s96 | 96 | 262144 | 3.052 / 3.201 |
| s96_32k | 96 | 32768 | **4.929 / 4.966** |
| s112_32k | 112 | 32768 | 3.622 |
| s128_32k | 128 | 32768 | 0.999 / 1.473 |
| s160_32k | 160 | 32768 | 0.498 / 0.579 |

Findings:

- **96 slots at the shipped 262144 context is a 30% loss, mirrored**
  (3.05/3.20 against 4.55/4.48). #152's hypothesis is false on the
  current build: restoring the tuned 12 GiB budget — the exact experiment
  `808cea3`'s comment deferred — would be a regression, not a recovery.
- **96 slots at a 32K context cap is the only winning arm: +9%**
  (4.93/4.97 against the base pair's 4.52). Hit rate rises 0.775 →
  0.850, confirming the extra 32 slots genuinely hold experts.
- **The cliff is the machine's working set, and the QSA indexer is what
  the context cap actually frees.** Runner stats, s96b vs s96_32kb:
  io_ms identical (66.2 / 65.7), hit rate identical (0.8525 / 0.8531),
  io_queue_ms identical (0.38) — but wait_ms 201 vs 98 and body_ms 290
  vs 184. The disk does the same work at the same speed in both arms;
  the slow arm pages. The accounting: the QSA indexer preallocates its
  per-layer key buffers at *full context* on first use
  (`QSAIndexer+Scratch.swift`: "Sized to the full context on a layer's
  first use, unlike the KV cache which grows on demand"). From the
  installed manifest (indexerHeadDim 128, compressRatio 4, 12
  full-attention layers of 48): **0.94 GiB at 262144 against 0.12 GiB
  at 32768**. That 0.82 GiB is the entire difference between the two
  96-slot arms: 12.15 + 3.72 + 0.12 ≈ 16.0 GiB fits, +0.82 ≈ 16.8 GiB
  does not. The KV cache is not the variable — it starts at 8192 tokens
  and grows (`min(maxContext, initialCapacityTokens)`), identical in
  both arms.
- **64 slots is on the right side of the cliff either way**, which is
  why the context cap measured neutral there (4.54/4.18 against
  4.55/4.48, inside the ~8% mirror noise this prompt shows).
- **112 slots is over the cliff even at 32K** (3.62), and 128/160 swap
  to collapse as before. The last good rung on this machine is 96, and
  only with the 0.82 GiB freed.
- **Deviation noted:** the s128/s160 arms left 9.4 GB of swap debt that
  persisted through the sweep; the mirrored pairs (and s96b running
  after two healthy arms at the same 3.2 as s96) are what make the
  cliff attribution safe anyway.

Verdicts:

- **#147 closed.** At shipped defaults the derived 64 slots is the best
  runnable configuration on this 24 GB machine. The binding constraint
  is physical working set, not budget policy.
- **#152 closed — hypothesis false, mechanism found.** The third-of-RAM
  ceiling is the *right* answer for qwen38 at the shipped context on
  this machine; v5.0's 6.957 is not reachable by any configuration this
  sweep turned. What grew since is named: the QSA decode indexer
  (`f5b1720`, 2026-08-30) wires ~0.94 GiB at full context and postdates
  the v5.0 measurement window. The rest of the gap to 6.957 is not
  attributable to a single knob and is not chased here.
- **#149 closed by the base arm's own runner line.** io_ms = 102.0 per
  token for ~403 MB moved ≈ 3.95 GB/s against the device's measured
  3.92 GB/s four-thread ceiling (`ParallelExpertReader`); io_queue_ms
  0.41 — no queueing to buy back. The disk is at its wall; every
  alternative allocation of that bandwidth (deeper rings, more staged
  reads, two-layer lookahead) is already measured loss-making in the
  prefetch study. The miss-read path has no headroom.
- **#151 closed without an experiment.** The serial tail per token is
  head 12.2 ms + gather 2.5 + sample 1.0 + embed 0.45 against a ~220 ms
  body; overlapping it across tokens requires starting the next token's
  embedding before this token is sampled, which is speculation — and
  speculation is §8-closed (MTP), with the fused head architecturally
  excluded for this family (hyper-connections on). The residue is ~1%,
  below the 8% mirror noise floor.
- **The lever the sweep did open:** the QSA preallocation is the file's
  own deferred work ("growing on demand is worth doing… and is not
  worth the bookkeeping yet"). Growing the indexer buffers on demand,
  exactly as KV does, frees ~0.82 GiB at short contexts at *any*
  max-context — which is precisely the headroom the +9% arm needs at
  the shipped 262144 default, with numerics untouched (the buffers are
  storage; selection runs over visible keys only). That is the next
  task.

## 5p. The lever taken: QSA grows on demand, the ceiling floor, and the default is now 96 slots (2026-10-15)

§5o's open lever, implemented and measured. Two commits:

- **`45f55bb` — the QSA indexer buffers grow on demand**, exactly as the KV
  cache does: 8192 cells at first touch, doubling with a `memcpy` of the live
  prefix, capped at `maxContext`. The full-context preallocation (0.94 GiB
  across the 12 full-attention layers) becomes 0.12 GiB at short contexts at
  any max-context. Growth is safe for the same reason KV's is: every access
  funnels through the one accessor, growth completes before the needing
  token or chunk is encoded, and Metal retains an old buffer for any command
  buffer still in flight. Selection runs over visible keys only, so the
  numerics are storage-independent. Model-free suite:
  `QSAIndexerGrowthTests` (start small, whole when small fits, doubling and
  capping, contents preserved across a grow).
- **The ceiling floor** — the experiment `808cea3`'s comment deferred ("a
  floor at the tuned budget, so only machines below the tune are cut").
  `affordableExpertCacheBudget` now keeps `wanted` on any machine with at
  least the 24 GiB the budgets were tuned on, and keeps the third-of-RAM
  clamp below it. The test amendment is stated in the test itself: the old
  32 GiB assertion expected `(32<<30)/3` because the 96 slots it buys paged
  under the QSA preallocation; with that gone, the cut buys nothing on a
  large machine and the assertion now expects the tuned 12 GiB through. A
  new boundary test pins the floor inclusive at 24 GiB and the third still
  cutting at 20.

Verification, all on this 24 GB M3, release build, greedy essay prompt,
`--max-new 128`, temperature 0:

| arm | slots | decode tok/s | text |
| --- | ---: | ---: | --- |
| old build (pre-`45f55bb`), default | 64 | 4.811 | reference |
| new build, default | 64 | 4.720 | byte-identical |
| new build, `--expert-cache-slots 96`, full 262144 context | 96 | **5.156** | byte-identical |
| floor build, **no flags** — shipped default derivation | 96 | **5.045** | byte-identical to both |

The floor run's own log: `ram profile cache=11.88G slots=96
resident_floor=3.72G estimate=15.60G` — the default now lands on the rung
the sweep measured as the last good one, with no operator flag. Greedy text
is byte-identical across all four arms, and `tools/golden-baseline.sh
--check qwen38-4` reports the output identical to the recorded baseline
(ornith-8 not checked: no install on this machine, reported rather than
fetched).

Effect by family on a ≥24 GiB machine: qwen38 4-bit gets its tuned 12 GiB
(64 → 96 slots, **4.48–4.81 → 5.05–5.16**, +7–14% depending on the base
arm); qwen3.6 35B-A3B 4-bit gets its tuned 10 GiB back (the ~+1.3%
`808cea3` priced); Ornith 8-bit is unchanged (its 8 GiB is a third of 24
GiB exactly, so floor and clamp agree); machines under 24 GiB are
unchanged — the 8 GB mini evidence for the third still stands and still
applies there.

Campaign accounting against the ≥6 tok/s goal: the shipped default moves
from 4.48–4.55 (mirrored base, §5o) to 5.05–5.16. That is the largest
single measured step of the campaign, taken with numerics untouched and
greedy output byte-identical. It is still short of 6: §5o/#152 recorded
that the remaining gap to v5.0's 6.957 is not attributable to a single
knob, and this step does not change that record. Single-run caveat: the
96-slot figures are one run each against a prompt whose mirrored spread is
~8% — but the pre/post gap at 96 slots (3.05/3.20 → 5.16, +61–69%) and
the default-derivation move (64 → 96 slots at equal text) are structural,
not clock noise.

## 5q. The same tier-sizing rule applied upward: budgets that scale with the machine (2026-10-15)

§5p floored the expert-cache ceiling at the tune, so a machine with at least the
24 GiB the budgets were measured on keeps what it was tuned to. That fixed the
machine this repository has, and left the other half of the answer unfinished: a
tuned budget is still a **constant derived on one machine class**, so a 64 GiB
Mac was handed the same 96 slots as a 24 GiB one and left ~40 GiB of the tier it
could have filled. `docs/lmcache-evaluation.md` named the one concept worth
taking from LMCache — size the resident tier to the host rather than to a number
measured elsewhere — and this section applies it at home, to both resident tiers.

**Change 1 — the third becomes a floor as well as a ceiling.**
`affordableExpertCacheBudget` is now `min(wanted, RAM/3)` below the 24 GiB tune
and `max(wanted, RAM/3)` at or above it. The measured machine classes are
byte-identical: below 24 GiB the rule is untouched, so the 8 GB mini evidence
still stands; at 24 GiB, `max(12, 8)` is the tuned 12 GiB, which is §5p's 96
slots; at 32 GiB a third is 10.67 GiB, so the tune still wins. What changes is
everything above, where the third exceeds the tune and the machine can hold it:

| RAM | a third | qwen38 4-bit (its row asks 12 GiB) | 35B-A3B 4-bit installs (ask 10 GiB) | prompt-cache RAM tier |
| ---: | ---: | --- | --- | ---: |
| 8 GiB | 2.67 | 2.67 → 24 (min, unchanged) | 2.67 → 40 (unchanged — the measured mini optimum) | 256 MiB |
| 24 GiB | 8.00 | 12.00 → **96** (measured, §5p) | 10.00 → 160 (unchanged) | 256 MiB |
| 32 GiB | 10.67 | 12.00 → 96 (tune wins) | 10.67 → 160 (same rung) | 256 MiB |
| 48 GiB | 16.00 | 16.00 → 128 (new) | 16.00 → 256 (new) | 2 GiB (new) |
| 64 GiB | 21.33 | 21.33 → 160 (new) | 21.33 → 256 (new) | 2 GiB |
| 128 GiB | 42.67 | 42.67 → 256, the ladder's top rung | 42.67 → 256 | 2 GiB |

Slot counts are the nearest rung on the 8…256 ladder, then stepped down while
the footprint exceeds 1.15 × the budget — the same rule that produced the 96 §5p
shipped. qwen38 4-bit is 129.6 MB a slot (stride 2,768,896 × 48 layers), the 35B
4-bit geometry 67.5 MiB (1,769,472 × 40). The `wanted` figures are the installs'
own `ModelProfile` table rows, which is what the server feeds the derivation
(`ServerModelSession+Loading.swift:118`) — not the family fallback. Nothing
changes at 32 GiB in either family; 48 GiB is where the third first exceeds the
tune.

**Change 2 — the prompt cache's RAM tier is derived, not constant.** Its default
was a flat 256 MiB, which is a small machine's number. The cost of prompt state
is now measured rather than assumed: this run stored a 152-token prefix as
`state_bytes=117442560` (112 MiB), so **~0.74 MiB per prefix token** on qwen38
4-bit — which means the 256 MiB tier holds roughly 350 tokens and pushes any
longer tool-loop prefix into the SSD tier, a second streaming consumer beside the
expert reads that already own the disk, while the RAM above it sat idle. 2 GiB is
the same rule at a big machine's size: about 2,800 prefix tokens resident. The
default stops there rather than scaling on, because past a few GiB what binds is
the store's own 4 GiB per-snapshot capture ceiling and the retained-entry count,
not RAM; an operator who wants more names the flag, whose range reaches 4 GiB.
Below 48 GiB the 256 MiB stays, because every machine this project has measured
lives there and it is what those measurements priced.

**Why this is the TTFT lever and not only a decode one.** Prefill streams the
experts a prompt activates, so a resident slot that holds one of them removes a
disk read from the path to the first token; a prefix restore served from RAM
rather than the SSD tier does the same. The profile prices prefill at 17.9 s for a
262-token qwen38 prompt (`docs/qwen38-decode-profile-2026-09-05.md`), and §5n
measured that the coupling between the KV reservation and expert residency is
real but indirect — it runs through RAM, not through the disk.

**Safety properties, each one a place where §5o's cliff was reached by the other
direction:**

- No measured class changes: below the tune `min`, at the tune `max(12, 8) = 12`,
  and the prompt-cache default on every machine measured here (< 48 GiB) is still
  256 MiB.
- A third cannot exceed the half-of-physical line the launcher warns on
  (`BatchedMemoryBudget.slotBudgetBytes` refuses to wire more than half), so the
  growth can never trip that guard by construction. Dense families are inert:
  `expertCacheHeldBack` is 0 when `numExperts == 0`.
- Every operator override still wins: `--expert-cache-slots`, `--ram-budget`,
  `--prompt-cache-memory-mib`.
- Slot counts and tier sizes change residency, not arithmetic. No numerics path is
  touched, so greedy output is unchanged by construction, and the golden check is
  the proof rather than the promise.

**Not measured.** This repository's install is the 24 GiB M3 — exactly 24 GiB, so
the floor is inclusive and the rule is a deliberate no-op here. There is no 48, 64
or 128 GiB machine available, so every row at or above 48 GiB is arithmetic on
the constant ladder, pinned by test (`aboveTheTuneTheBudgetGrowsWithTheMachine`,
`promptCacheMemoryTierGrowsWithTheMachine`) rather than measured. §5o bounds how
far to trust them: on 24 GB the rungs above 96 collapsed on *physical working
set*, not budget policy. A 48 GB machine has the headroom that rung needs (15.84
GiB of cache + the 3.72 GiB resident floor + 0.12 GiB of QSA at short context,
against 48 GiB), so the arithmetic is not contradicted by anything measured — but
nobody should call 128 slots a measured number until it runs on a machine that
big, and a long-context prompt there should be re-checked against the same
accounting before it is believed.

**Verification.** Three changes, each with its test run failing first.

- **Change 1** — `aboveTheTuneTheBudgetGrowsWithTheMachine` failed against the
  floor-only rule (`sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift:199`)
  before the `max` arm existed, and passed after. The existing
  `clampNeverGrowsABudget` had to be amended rather than left alone: its single
  assertion *was* the old premise — 8 GiB of budget on a 128 GiB machine stays
  8 GiB — which this change reverses on purpose. It is renamed
  `clampNeverGrowsABudgetBelowTheTune` and now pins the side that must not move
  (16 GiB and 8 GiB machines, `physicalMemory == 0`, the tune itself), and the
  reversed claim is asserted by the new test rather than dropped. The derivation
  has one production caller (`sources/TinyTitanLib/ServerModelSession+Loading.swift:118`),
  so nothing else could have drifted.
- **Change 2** — the order here was wrong and is recorded as such: `parse`
  already derived the tier when the assertion was written, so the RED had to be
  produced by temporarily reverting the seed to a flat `256`, watching
  `tests/TinyTitanServer/OpenAIValidationTests.swift:566` fail on
  `big.promptCacheMemoryMiB == 2_048`, and restoring the line. `defaults()` also
  gained a pinned `physicalMemory: 24 << 30`: an unpinned assertion over a
  derived default reads whichever host ran the suite, which is the AUD-312
  defect class.
- **The launcher pinned the tier away** — found by asking where change 2 is
  actually *reachable*. `tools/server_launcher.sh` put
  `--prompt-cache-memory-mib 256` into every GPU command line, and the launcher
  is how the owners of the big machines this was written for start a server: a
  dry run against a synthetic 128 GiB host showed the pin, so the derivation
  would have reached no one who used the supported path. `test_launcher_ram.py`'s
  new `ThePromptCacheTierIsLeftToTheServer` failed on both the 24 GiB and the
  128 GiB sub-arm before `tools/server_launcher.sh:1332` changed, and passes now
  that the launcher names only the mode. `benchmark/tinytitan_profile.py`'s
  `DEFAULT_PROMPT_CACHE_MEMORY_MIB` had one reader — `test_tinytitan_profile.py`,
  which asserted the launcher's pin as a property of the benchmark protocol. That
  test was the wrong side, because the pin was the defect: it now asserts the
  launcher's command carries no tier, and the constant is gone.- **Suites and gates** — `swift test --no-parallel` after the last source edit:
  7 bundles, 1765 tests, exit 0. `tools/lint.sh` full battery: all
  18 gates ok (exit 0). `swift build -c release`: exit 0, `lipo -archs` reports `arm64`
  alone. Python: `test_launcher_ram` plus `test_tinytitan_profile` — 35 tests,
  OK — and ruff clean at the pin.
- **On the real machine, configuration identity** — a no-flag server launch on
  this 24 GiB M3 logged `TinyTitan ram profile cache=11.88G slots=96
  resident_floor=3.72G estimate=15.60G` and bannered `expert_slots=96 …
  prompt_cache_memory_mib=256`: byte-identical to what the shipped build
  banners, so both derivations are the deliberate no-ops they claim to be where
  they must be. `tools/golden-baseline.sh --check qwen38-4` → *ok — output
  identical to baseline*, exit 0. Ornith 1.5 8-bit is **not checked**: this
  machine has no 8-bit install and a gate may not fetch one.
- **A protocol deviation, recorded** — the same live run clocked 0.617 tok/s and
  a 20.895 s TTFT where §5p measured 5.045 tok/s at the identical configuration.
  Root-caused before anything was concluded from it: the Mac was on battery at
  34–35% with `lowpowermode 1`, 8.2 GB of swap in use against 2.9 GB before the
  run, 8% of RAM free, load 12.4 left over from the suite and builds that
  preceded it. The protocol requires mains, so those two figures are **not
  measurements** and are quoted nowhere else in this file; the banner and
  `ram profile` lines above are clock-independent, which is why they stand. §5r
  takes the mains re-run of decode, TTFT and `wait_ms` this left open, and
  corrects one use of those lines: `ram profile` prints the *derived* budget
  even after an operator's `--expert-cache-slots` has replaced it, so it is not
  proof of the rung that actually ran.
- **One claim corrected by measurement** — this section first said the 2 GiB
  tier "holds the retained four prefixes at the store's 4 GiB capture ceiling".
  The run's own log disproved it: `prompt_cache stored tokens=152
  state_bytes=117442560 ram_bytes=117442560 disk_bytes=0` is 0.74 MiB per prefix
  token, so 256 MiB holds about 350 tokens and 2 GiB about 2,800 — not four
  multi-gigabyte prefixes. The doc comment on `defaultPromptCacheMemoryMiB`, this
  section and the wiki row now carry the measured figure.

## 5r. The mains re-measurement of the shipped 96-slot default (2026-10-15)

§5q left three numbers open and one of its own pieces of evidence invalid. All
four are settled here. Provenance: commit `0242f65`, tree clean, 24 GiB M3
(`hw.memsize` 25769803776), macOS 27.0.1, Swift 6.4
(swiftlang-6.4.0.34.1 clang-2100.3.34.1), `swift build -c release` exit 0 in
85.21 s with `lipo -archs` = `arm64`, **AC power with `lowpowermode 0`**,
`pgrep` guard empty before every arm, one model process at a time. Seven CLI
arms, all exit 0, all producing stdout sha256
`95aa4531b4ea2304a10dde8d079c68bb3b2bf2eea1335b21814fe3fca5c2d0f0` (2,065 bytes)
— greedy text identical across both rung and both batch.

Arm command, batch A (three identical arms of the shipped default, 15:21–15:26Z):

```
TINYTITAN_RUNNER_STATS=1 .build/release/TinyTitanCLI \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0
```

No `--ram`, no `--expert-cache-slots`: the derivation lands on 96 slots, and each
arm logged `ram profile cache=11.88G slots=96 resident_floor=3.72G
estimate=15.60G`. Batch B (15:27–15:33Z) is the same command with
`--expert-cache-slots 64` interleaved as 64, 96, 96, 64 so machine drift cancels
in the paired difference and the baseline runs first *and* last.

| batch | run | slots | decode tok/s | prefill s | wait_ms | io_ms | body_ms | hit | read MiB | evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 96 | 4.075 | 6.080 | 118.84 | 75.26 | 218.43 | 0.8456 | 75,868 | 31,293 |
| A | 2 | 96 | 4.406 | 3.824 | 111.73 | 73.41 | 207.10 | 0.8459 | 75,709 | 31,293 |
| A | 3 | 96 | 4.632 | 3.226 | 102.59 | 73.65 | 196.44 | 0.8452 | 76,047 | 31,293 |
| B | 1 | 64 | 4.245 | 3.171 | 84.64 | 104.76 | 216.67 | 0.7704 | 112,768 | 48,582 |
| B | 2 | 96 | **5.021** | 3.808 | 91.67 | 72.10 | 181.19 | 0.8448 | 76,245 | 31,293 |
| B | 3 | 96 | **5.074** | 3.069 | 93.02 | 68.76 | 179.41 | 0.8462 | 75,554 | 31,293 |
| B | 4 | 64 | 4.690 | 2.820 | 75.61 | 100.11 | 195.61 | 0.7716 | 112,190 | 48,582 |

**1. §5p's step is confirmed at §5o's own arm length.** Paired 96−64 deltas are
**+0.776** (runs 1→2) and **+0.384** (runs 4→3): two rounds, both positive. The
mechanism is in the counters, not the clock — 76 GB of expert reads against 112
GB (−32%), 31,293 evictions against 48,582, hit rate 0.845 against 0.771, and
`io_ms` 69–72 against 100–105. §5p measured 5.045 at `--max-new 128` with n=1 and
flagged that as a caveat; the caveat is discharged for the direction and the
magnitude at 384 tokens is +0.4 to +0.8 tok/s. This still does not reach 6.

**2. A finding that outranks the headline: the first arm after a load is
systematically slow.** Batch A rose monotonically on one identical configuration
(4.075 → 4.406 → 4.632, +13.7% from first to last); batch B's two 64-slot arms
differ by 10.5% in the same position order; and the server arm's first request
decoded 4.099 tok/s where its second reached 5.431 (+32%). `wait_ms` carries it —
102–119 in batch A against 92–93 in batch B at near-identical `io_ms` and hit
rate, so the engine did the same work and waited longer. The likely mechanism is
file-cache warmth of the expert stream; it is **untested**, and named as a
hypothesis rather than a conclusion. The consequence is procedural and applies
backwards through this whole file: cross-*batch* qwen38 tok/s comparisons are not
usable, and the "~8% mirror noise" §5o recorded understates this host — the same
configuration measured up to 24% apart across batches. Within-batch pairing is
what a qwen38 number has to rest on.

**3. `wait_ms` at the shipped default is 91.7–93.0 ms of a 179–181 ms body.** §5o
priced the double-buffered-plan axis against 98 ms at 96 slots with a 32K cap; at
the shipped 262144 context and the QSA fix the number is essentially unchanged,
and it is ~51% of the token. `io_hidden_pct` is 99.6–99.8% and `io_queue_ms`
0.28–0.41, so the disk is not queueing and the miss I/O is already hidden — what
remains exposed is the wait for the *next* layer's plan, which is exactly axis 1.

**4. TTFT at the shipped default on mains is 6.882 s.** The server arm launched
no flags on port 8085, bannered `model=qwen3.8-flash-next_4-Bit … expert_slots=96
prompt_cache=multi-prefix prompt_cache_memory_mib=256`, and served a 128-token
greedy chat request: first content delta at **6.882 s** (`prefill_s=6.808`),
second identical request 4.674 s (`prefill_s=4.514`). §5q's 20.895 s was 3.0×
this, which is what "battery artifact" means in numbers. The server was launched
and stopped by the arm's own script; no other model process existed.

**5. An identical replay is not a prompt-cache hit, and that is the design.** The
repeat request logged a *second* `prompt_cache stored tokens=149` entry and no
`hit` line, so this looked like a reuse failure until `ServerPromptCache.match`
was read: `sources/TinyTitanLib/ServerPromptCache.swift:172` requires
`renderedPromptIDs.count >= entry.kvPosition`, and the stored entry's KV-backed
prefix is 149 tokens — the prompt *and* the generation — against a 25-token replay,
so the direct-prefix arm cannot fire; `:180` then requires more messages than a
single-turn replay has. Reuse is for continuations (the tool loop the wiki
documents), where the next request *extends* the stored prefix. §5q's `cached=0`
is the same mechanism, not a regression — and it bounds what §5q's bigger RAM tier
buys: a long retained prefix in a multi-turn loop, not a replayed one-shot prompt.
The per-token cost re-measured here, 117,403,392 bytes over 149 tokens = 0.738
MiB per prefix token, matches §5q's 0.74 within 0.5%.

**6. The operator cap, and the machine.** On this 24 GiB M3 the rule is that no
run may take a RAM budget above 12 GiB, because above it the machine destabilises;
larger Macs keep the §5q host-sized derivation, which is what they were written
for. Every arm here complied: 11.88 GiB derived cache, nothing above it named, and
the 64-slot arms deliberately *under* it. Stability held across all seven CLI arms
and the server — every exit 0, swap peaking at 9.41 GB without a watchdog, a crash
report or a lost request. Note that 64 slots is the rung whose whole footprint
(7.92 + 3.72 = 11.64 GiB) fits under 12 GiB including the resident floor, while the
shipped 96 sits at 15.60 GiB estimated: the cap is on the cache tier, and which
reading an operator wants is their call, not a measurement's.

## 5s. The two RAM diagnostics fixed test-first, and the wiki arm refused (2026-10-15)

§5r recorded the `ram profile` defect as authorised-for but not granted; the
operator then said fix it. Two commits: `2c2a30b` (the applied count in both
lines) and `0355d2c` (the warning names the override when the override is what
misses the target). No clock in this file changes — this section is about what
the load-time log is allowed to be used as evidence of.

**1. Root cause.** `ram profile` and `ram target` printed the rung the
*derivation* chose, while the allocation used `requestedExpertCacheSlots ??
slotOverride ?? derived`. That precedence existed only as an inline expression
*below* the diagnostics, so the printed number and the allocated one were
computed twice by different code and had drifted: an operator who passed
`--expert-cache-slots 96` under `--ram-budget 12G` got 96 slots wired and a load
line asserting `slots=64`. A second defect surfaced only by probing the fixed
line against a live server — the warning called those 96 allocated slots "the
minimum cache" and said "the cache is already at its smallest". Both are false
when an override pinned the count, and both pointed the reader at the install
when the flag they set themselves is the thing they can change.

**2. The lines now** (verbatim: a no-flag CLI run with `--expert-cache-slots 64`,
and the `--ram-budget 12G --expert-cache-slots 96` server probe on port 8086):

```
TinyTitan ram profile cache=7.92G slots=64 derived_slots=96 resident_floor=3.72G estimate=11.64G (cache budget, not a process target)
TinyTitan ram target=12.00G cache=11.88G slots=96 derived_slots=64 resident_floor=3.72G estimate=15.60G
TinyTitan ram warning: 12.00G is below this install's 15.60G footprint (3.72G resident + the 96-slot cache named by --expert-cache-slots); the target alone allows 64 slots, so the gap is the override, not the install.
```

`slots=` is the applied count in every line; `derived_slots=` is emitted only
when an override replaced the derivation, so the shipped default line is
byte-for-byte what it was. `appliedSlots` is defined **once** in
`ServerModelSession+Loading.swift` and shared by the two diagnostics, the
`loadSlots` that sizes the cache, and the server banner. The text itself moved to
a new 76-line `sources/TinyTitanLib/ExpertCacheDiagnostics.swift` with six tests
pinning both branches of both lines, because these lines are the operator's only
record of the allocation and nothing downstream re-derives them.

**3. Which prior claims this corrects.** §5p's third arm (line 786,
`--expert-cache-slots 96` at full context) is the case the defect describes: the
table's 96 was right, but under the old code that run's own `ram profile` printed
`slots=64`, so the log could not have confirmed it — the rung was known only from
the flag. The line quoted at §5p (the floor run) and at §5q was legitimate,
because those were no-flag loads where derivation *is* the applied count. §5r's
finding 1 therefore closes: the identity evidence it needed is now produced
correctly rather than worked around. Nothing is re-measured for this, since every
rate here already rests on within-batch pairing plus rung-specific counters (hit
rate, evictions, `expert_read` MiB).

**4. Flag ownership, which two recipe lines here got wrong.** The runtime and
server take `--ram-budget <size>`; `tools/server_launcher.sh` takes `--ram <whole
GB>` and forwards it as `--ram-budget "${ram_gb}G"` (`server_launcher.sh:1339`),
and only when a budget was chosen at all — an unattended run passes nothing and
the install profile applies; the whole `TinyTitanCLI` target parses **no** RAM
flag of any spelling, only `--expert-cache-slots`. §5n and §5o cited a bare
`--ram` on the CLI, which is not
a command anyone could have run: it exits 2 with `error: unknown flag`. Both lines
now state what actually executes.

**5. Verification on the committed tree.** `swift test --no-parallel` → seven
target runs, 1,771 tests (815 / 408 / 157 / 189 / 41 / 63 / 98), each reporting
*passed*, no failure line. `tools/lint.sh` → exit 0, 18 gate sections clean.
`tools/golden-baseline.sh --check qwen38-4` → *ok — output identical to baseline*
(ornith-8 not checked: no install here, and a gate may not fetch one). Both lines
above came from real loads, the server probe launched by a pgrep-guarded script
that killed its own server on exit; `lipo -archs` on the release binaries is
`arm64`. The test-file changes are disclosed in `0355d2c`'s message: one existing
fixture (applied=8, derived=16) described a pair no loader can produce, so it was
retargeted to applied=derived=8 — the state it meant to pin — with its expected
text unchanged, and the override shape it accidentally covered became a new test.

**6. The wiki's Qwen3.8 decode row is not refreshed, and the refusal is the
finding.** The second authorised item was `.qwen/wiki/Benchmarks.md`'s 6.82 tok/s
for Qwen3.8-Flash-Next 4-bit (the 2026-08-30 batch) against §5r's 5.02–5.07.
Those are different protocols, so refreshing the row means running the wiki's own
instrument, not pasting a number: `benchmark/tinytitan_gate0_profile.py` with
`--quant 4bit --model-dir models/qwen3.8-flash-next_125B_A6B_4Bit --runs 5` at
shipped defaults (11.88 GiB cache, 96 slots). It refused before starting a server:

```
refusing to start: GPU is already 54% busy before TinyTitan launches (threshold 15%).
```

`Device Utilization %` held at 43–68% across a 30 s sample with load average
3.4–6.3, and the consumers are the operator's desktop, not this engine:
`Qoder Helper (Renderer)` 42–46% CPU, `WindowServer` 42%, three Chrome renderer
helpers, WhatsApp, Boom 3D. Everything else was compliant — AC power,
`lowpowermode 0`, macOS 27.0.1, Swift 6.4, 80% memory free, empty `pgrep` guard,
clean tree. `--allow-busy-gpu` exists to override the threshold to 100%, and its
own message says a contended result is not comparable with the published
benchmark, so it was not used and the row stands unchanged, reported as **not
measured**. Two procedure notes: this driver is the only one under `benchmark/`
that gates on GPU idleness, so §5r's CLI arms were taken without that check; and
its threshold is binary (15% or off) — there is no "record the contention and
carry on" mode, which is the right design for a published table but means a
wiki-protocol arm needs the GPU-holding apps quit first.

Two further things about that refused arm were found after the fact, and each would
have cost a quiet-machine window to discover the other way round.

- **The command as actually run named the wrong install.** It was invoked as
  `--quant 4bit --runs 5` with no `--model-dir`, and that driver's `4bit` key is
  `models/ornith-1.5_35B_A3B_4Bit` (`tinytitan_gate0_profile.py:75`) — the published
  workload was Ornith-only until `--model-dir` was added. This Mac has no Ornith
  install, so on a quiet GPU the arm would have died on a missing model directory
  after waiting all day for one. The refusal itself still stands, because the GPU
  preflight runs before the model is opened, but the Qwen3.8 row could never have
  been refreshed by that invocation. TT-049's next-step command on the tracker
  already carries the `--model-dir`; the command quoted above did not, and now does.
- **The budget claim is confirmed in the code rather than assumed.**
  `server_command` appends `--ram` only when `ram_budget` is not `None`, and the
  harness default is `os.environ.get("TINYTITAN_BENCH_RAM_BUDGET")` with no fallback
  — `None`, so the shipped derivation applies (`tinytitan_profile.py:29-30`,
  `:211`). That is what "no flag, 96 slots" rests on. The gate0 docstring still
  describes the profile as using an "8 GiB budget", which was true of the v4.1
  published batch and is not what the harness does now; left unfixed, because this
  window was not authorised to touch the driver, but no reader should take that
  line as the arm's configuration.

**7. What the page now says instead (wiki `c2bb180`).** The 6.82 figure was not
touched — it is still the v4.7 measurement, and only a wiki-protocol arm can change
it — but the page's own provenance was checked and published, because the
contradiction is older and wider than one row. `git log -S` on the wiki: all five
medians arrive in `eb84741` (2026-08-30, "Add the v4.7 benchmark section") and none
has been edited since, while the peak paragraph was rewritten on 2026-09-29
(`5d74cd6`) from the README matrix, whose rows for these models all arrived in
`69dd683` (2026-09-05). Four of the five table medians are *above* the peak the same
page then quotes for that model and width — 23.28 and 22.65 against "17–22" at
4-bit, 12.72 against "7–12" at 8-bit, 6.82 against "5.5" — and only Ornith 1.5 8-bit
at 11.89 sits inside its range. A median at shipped defaults cannot exceed a peak
taken across every setting, so the two sets cannot both describe one build, and the
table is the older of the pair by six days. For the 125B row §5r settles it
independently: 4.08–5.07 tok/s at the shipped 96-slot default, against 6.82. The
note states the batch, the contradiction, that range and its protocol, points at
TT-049, and pastes nothing into the table.

**Amended 2026-10-17:** the last clause of that sentence is what the page no longer
does. Two days and one operator decision later the 125B row was withdrawn outright and
replaced by §5bn's and §5r's CLI arms published as their own labelled protocol
(median **5.40 tok/s** over 15 surviving controls), because §5bd had by then shown
that the gate0 bar this note refused to override is not a statement about this host at
all. What §5s did on 2026-10-15 — refuse to paste a number into a table it could not
certify, and open a row instead — stands as the right call; the row it left behind is
the part that has since moved. See §5t's amendment for the closing and §5bd's verdict
bullet for the instrument finding.

Two claims were caught and corrected while drafting that note, and are recorded
because both were plausible: that the README's `5.46` predated the table (it
postdates it by six days — the `5.46` in README on 2026-08-08 is a different
quantity, a "32 slots + pin" reference figure), and a positional "two lines below"
that the insertion itself invalidated. What the note does *not* establish is whether
the four 35B rows are also unreachable on the current engine: this Mac has no 35B
install, and no gate may fetch one, so on this machine TT-049 can only ever refresh
the 125B row.

The sibling sweep for the same defect found two engineering documents dividing by
that stale figure. `docs/ngram-table-sharing-plan.md` projected the 8-bit install at
"about 3.4 tok/s" as half of 6.82; corrected in the same commit as this section,
because the README already measures that build at **2.10 tok/s** — 0.39x the
README's own 4-bit row, so the projection was optimistic and its verdict (not worth
building) was right for a stronger reason than the arithmetic. The other is
`docs/qwen38-decode-20tps-concept.md:110`, which opens "Published decode is
**6.82 tok/s**" and reasons from it to a ~12 tok/s target. Left alone: it says
*published*, which remains true of the wiki row, it is a superseded concept
document, and §5r's 4.08–5.07 already displaces it for any live decision. Neither
file is guarded by a gate, so both readings are documents, not proofs.

**Amended 2026-10-17:** the first of those three reasons expired with the row it
pointed at — the wiki no longer publishes `6.82` — so that line carries its own dated
note naming **5.40 tok/s** and §5bn. `docs/ngram-table-sharing-plan.md` needed nothing:
it had already been retargeted onto §5r and the README's 8-bit row, and it still reads
correctly. `docs/release-notes-v5.20.md` also names the row as left in place with
TT-049 open, and is deliberately **not** edited: release notes are a record of what was
true at the tag, not a page to keep current.



## 5t. The wiki's own instrument, run on v5.20, on a contended GPU (2026-10-15)

§5s recorded the refused arm and left TT-049 open for a quiet machine. The
operator then said run it now, so it ran with the guard overridden. **This is not
a benchmark figure and the wiki row is unchanged** — see 4 and 5 for why.

**1. What ran, and against what build.**

```
python3 benchmark/tinytitan_gate0_profile.py --quant 4bit \
  --model-dir models/qwen3.8-flash-next_125B_A6B_4Bit --runs 5 --allow-busy-gpu
exit 0    # 1 warm-up + 5 measured runs, fresh server per run, 9m21s wall
```

Two launches were refused first, at GPU 53% and 70%, by the 15% preflight
(`tinytitan_gate0_profile.py:220`); the third went because `--allow-busy-gpu`
raises the threshold to 100%. Commit `b27f29e` (the v5.20 tag). The binary is
`.build/release/TinyTitanServer`, mtime 21:19, and carries the `5.20` version
literal; `b27f29e` changed only that literal and the notes file, so this is the
shipped code path. The rung is confirmed from the run's own server log rather
than assumed — `.build/benchmark-logs/gate0_4bit_5.log`:

```
TinyTitan ram profile cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G (cache budget, not a process target)
```

No `derived_slots=`, so the derivation applied with no override: the shipped
default, exactly what the wiki table claims to measure.

**2. The numbers.** Median over five runs; each run is 512 tokens of continuous
prose at native 262K context, multi-prefix prompt cache on, 8-bit KV, top-k 20,
temperature 0.6, MTP off.

| run | decode tok/s | busy/token ms | occupancy | expert hit | expert read MiB | wait_ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.450 | 104.858 | 52.1% | 0.8718 | 73,459.5 | 96.754 |
| 2 | 5.491 | 100.333 | 50.3% | 0.8722 | 73,277.3 | 97.026 |
| 3 | 5.713 | 96.823 | 50.4% | 0.8719 | 73,409.4 | 91.634 |
| 4 | 6.102 | 92.234 | 51.0% | 0.8718 | 73,464.8 | 84.661 |
| 5 | 6.026 | 94.920 | 51.7% | 0.8719 | 73,422.6 | 86.497 |

**median 5.713 tok/s = 175.04 ms/token**; `prefill_s` 7.598–7.912 (median 7.79);
`io_ms` 51.195, `io_hidden` 99.71%, `gpu_share_of_decode` 65.5%, router readback
0.0162 ms, cache plan 3.4161 ms. Driver verdict: `BANDWIDTH-BOUND (ceiling
~1.35x)`, which is the same verdict §5c–§5r reached from the other direction.

**3. The finding that outranks the level: all five runs are byte-identical.**
`completion_sha256` is `aded9bd20fbde145` for every run, 440 completion tokens
each, and the counters move with it — hit rate spans 0.8718–0.8722 and expert
reads 73,277–73,465 MiB, both inside 0.25%. The work was provably the same in all
five arms, yet decode rose 5.450 → 6.102 across them, **+12%**, and
`busy_per_token` fell from 104.9 to 92.2 ms. That is §5r's first-arm-after-a-load
effect reproduced inside a single protocol with the bytes held fixed, and it is
the strongest evidence in this file that a *level* from a batch is a property of
the machine settling, not of the build. Only paired within-batch deltas carry
magnitudes; a median of five sequential identical arms carries a range.

**4. The contention, measured rather than assumed.** `Device Utilization %` was
sampled every 30 s for the whole run (19 samples, `.build/benchmark-logs` aside):
50–65%, never below 50%, load average 2.7–7.4. Everything else complied — AC
power, `lowpowermode 0`, macOS 27.0.1, Swift 6.4, 82% memory free, empty `pgrep`
guard before launch, clean tree, and no model process left after it
(`_terminate_all` ran in the driver's `finally`). Note what contention did *not*
do here: the hit rate stayed at 0.872, so the expert cache remained resident and
this is not the 0.277-collapse shape the preflight's comment describes. The
sharing shows up as CPU/GPU service time, not as an I/O failure — which is
exactly why a contended run looks superficially healthy in its counters.

**5. What this licenses, and what it does not.** Three figures now exist for this
model and width and none of them is the same measurement: this contended
gate0 median **5.713**, §5r's greedy 384-token CLI arms **4.08–5.07**, and the
wiki table's **6.82** from the v4.7 engine on 2026-08-30. Note that 5.713 already
sits *above* the 5.5 this page quotes as the model's peak and the README's 5.46 —
so the table-versus-peak contradiction the wiki's batch note describes is not
resolved by taking the table's own instrument; it is a protocol difference between
the peak row and the table row, still standing. What this run licenses is the
statement that **the 6.82 row cannot be defended as a current figure at this
machine's defaults** — it was measured on another engine, and the current build at
the shipped default under contention does not reach it. What it does not license
is replacing it: the replacement must come from the same command without
`--allow-busy-gpu`, on a GPU under 15%, and that window has never been available
on this Mac while its desktop is running. TT-049 stays open for that arm.

**Amended 2026-10-17:** TT-049 closed without that arm. §5bd established that the
bar is not a statement about this host at all — the reader's floor *is* the
desktop's own compositing — and the operator then chose to publish the CLI protocol
instead of a gate0 median: the wiki's Benchmarks page withdrew the `6.82` row and
now carries §5bn's controls and §5r's arms as their own labelled protocol, with
this section's contended `5.713` explicitly named as not a level (wiki `c2591e7`,
its gate0 sentence re-worded in place by `5e13783`, the page's head).
A gate0-certified row for this model still needs a preflight that measures
TinyTitan's own occupancy rather than the desktop's.

## 5u. The 32K context cap re-measured on v5.20 — §5o's winning arm is gone (2026-10-15)

§5o found exactly one configuration that beat that batch's default: 96 slots with
`--max-context 32768`, **+9%** (4.929/4.966 against 4.551/4.481), and it named the
mechanism — the QSA indexer preallocating 0.94 GiB at full context against 0.12
GiB at 32K, that 0.82 GiB being the whole margin between the 96-slot arm paging
(3.05) and fitting (4.93). §5p then shipped the fix (`45f55bb`): the buffers start
at `initialCapacityCells = 8_192` and double on demand
(`sources/TinyTitan/Kernels/Attention/QSAIndexer.swift:34`), which frees the same
0.82 GiB at *any* max-context. The question this section answers is narrow and it
is the one the operator asked — after that fix, does the cap still buy anything?

**Provenance and protocol.** Tree `3f92d47` (the `v5.20` tag plus a notes-only
commit), `swift build -c release` refreshed before the batch (Build complete,
82.35 s), clean `git status`. 24 GB M3, macOS 27.0.1, Swift 6.4, AC power,
`lowpowermode` off, empty model-process guard before launch, one CLI process at a
time, none left after. Seven cold runs: three interleaved rounds with the arm
order alternated (A B · B A · A B) plus a closing baseline run to bound drift.

```
TINYTITAN_RUNNER_STATS=1 .build/release/TinyTitanCLI \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0 --max-context 262144   # arm A — shipped default
                                          --max-context 32768    # arm B — capped
```

§5o's prompt and length, greedy as §5p ran it. Both arms derived the same tier:
every one of the seven logs carries `ram profile cache=11.88G slots=96
resident_floor=3.72G estimate=15.60G`, so the cap is not changing the slot count —
this is a pure memory-pressure test at equal capacity. Driver
`/tmp/v520_ctx_ab.sh`, data `/tmp/v520-ctx-ab/summary.tsv`.

**Deviations, all of them.** (1) The desktop held the GPU at 56% at run 1's launch
instant; runs 2-7 launched at 0%. Round 1's delta is therefore contaminated and
only rounds 2-3 are clean. (2) A 16-token smoke run preceded the batch to pin the
footer field names — not an arm, but it warmed the page cache for run 1. (3) The
harness pipeline exited 1 because `tee` opened its progress log before the script
created the directory; every one of the seven CLI runs exited 0, so the batch
itself succeeded. (4) The arms went through the CLI, which accepts any count up to
262144, while the server refuses a non-rung; 32768 is a rung, so the two paths
agree on this value.

| run | arm | ctx | tok/s | decode s | hit | wait_ms | body_ms | io_ms | gpu at launch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | A | 262144 | 4.884 | 78.63 | 0.8460 | 96.086 | 186.469 | 70.619 | 56% |
| 2 | B | 32768 | 5.061 | 75.88 | 0.8446 | 90.753 | 179.916 | 72.640 | 0% |
| 3 | B | 32768 | 5.067 | 75.79 | 0.8461 | 92.876 | 179.349 | 69.449 | 0% |
| 4 | A | 262144 | 5.072 | 75.70 | 0.8463 | 93.603 | 179.082 | 68.707 | 0% |
| 5 | A | 262144 | 5.111 | 75.13 | 0.8454 | 90.028 | 177.664 | 71.504 | 0% |
| 6 | B | 32768 | 5.086 | 75.50 | 0.8452 | 90.767 | 178.425 | 71.394 | 0% |
| 7 | A | 262144 | 5.031 | 76.33 | 0.8458 | 94.640 | 180.768 | 68.603 | 0% |

**Paired deltas, per round.** Round 1 (contended) **+3.62%**, round 2 **−0.10%**,
round 3 **−0.49%**: median **−0.10%** (−0.005 tok/s). The identical-arm band is
wider than the effect — the four A runs spread 4.884-5.111 (4.6%), or 5.031-5.111
(1.6%) once the contended run is excluded, and the three B runs spread 0.5%. A
direction needs two rounds of its own; here the two clean rounds point opposite
ways and both are inside the drift band. **The cap is neutral on v5.20.**

**The counters say the mechanism is gone, not just the clock.** `wait_ms` is the
quantity §5o used to prove the cap was relieving paging: it measured **201 ms
uncapped against 98 ms capped**. This batch measures **90.03-96.09 uncapped** (A
median 94.12, or 93.60 with the contended run excluded) against **90.75-92.88
capped** (B median 90.77); the per-round paired wait deltas are +5.33, +0.73 and
−0.74 ms, so the arms are not separable on it either way. The point is not the
1.4% between these two arms — it is that the ~100 ms/token the cap used to buy
back is already absent from the *un*Capped arm, which is `45f55bb` working, seen
from the other side. `io_ms` is flat across both arms (68.6-72.6) and the hit rate
does not separate them either (0.8446-0.8463), so there is no second effect hiding
under the first.

**Output invariance.** `sha256` of the generated text is `95aa4531b4ea2304…` in all
seven runs, 384 tokens each, `stop=maxTokens` throughout. The cap changed no token,
so nothing here is a numerics question.

**Verdicts.**
- **§5o's +9% arm is closed as a lever.** Do not recommend `--max-context 32768`
  for decode speed on a machine that keeps its tuned budget: the headroom it was
  buying is now released by default. It stays what the wiki documents it as — a
  capacity ceiling — and it still has the one real effect the code shows: because
  `capacity: maxContext` (`sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift:310`),
  a cap bounds how far the indexer and KV buffers may grow, so a request above the
  window is refused rather than allowed to page. On a long-prompt workload that is
  a blast-radius control, not a speed control.
- **The configuration search for ≥6 tok/s on this Mac has no untried flag left.**
  The shipped default measures 5.03-5.11 here (median 5.05), agreeing with §5p's
  5.045 and sitting under §5t's contended 5.713 server-protocol median. Every
  candidate this campaign turned — budget (§8), context (§5u), rdadvise (§8),
  knobs (§8), MTP (§8), co-execution (§8) — is now measured closed.
- **New reading worth carrying:** during each run `memory_pressure -Q` falls to
  **8-11% free** (77-80% at the launch instants) while the 11.88 GiB tier is
  resident. That is the margin the 12 GiB budget is chosen to fit, and it is direct
  evidence the budget sits at the edge rather than comfortably inside: anything
  else opening on this machine is a few hundred megabytes away from pushing the
  expert cache out — the collapse shape the gate0 preflight hunts. It also means
  the paired-batch free-memory guard is measuring the wrong instant for this model:
  a launch-time 77% free says nothing about the 9% the run runs at.

## 5v. The continuous weight feeder, tested on the drive itself — the drive is already continuous (2026-10-15)

The operator's idea, stated as the premise to be tested: *"it makes no sense to
make it read for a couple of seconds, then stop, compute and read again. the SSD
can read constantly while a user has an active session… the idea is to have a
constant weight feeder instead of a chunk-by-request reader."* That is a claim
about duty cycle, so it was measured as one rather than argued from §8's closed
lever list. Nothing in the probe touches the engine: the samplers are external,
and the decode arm runs the shipped default with **no flags at all** — the 12 GiB
tier in its logs is the host-sized derivation on this 24 GB Mac, not an override.

**Provenance and protocol.** Tree `b8002b0`, whose only commits past the `v5.20`
tag are docs-only (`git diff --stat v5.20..HEAD` = one `docs/` file), so the
binary is the released engine. 24 GB M3 (Mac15,3), macOS 27.0.1, Swift 6.4, AC
power, empty model-process guard before launch, one CLI process at a time. Two
arms, each a cold run: **A** = `TINYTITAN_RUNNER_STATS=1` decode, **B** = the same
plus `TINYTITAN_ROUTE_TRACE` for the offline simulator. Both sampled every second
by external `iostat -d -w 1` and an `ioreg` AGX poller. Prompt as §5u, `--max-new
512 --temperature 0`, exit 0 in both arms. Driver `/tmp/feeder_probe_run.sh`, data
`/tmp/feeder-probe/`. Preconditions at launch: 81% memory free, **swap already
5.47 GB used**, GPU reader 61%.

**Result 1 — the drive never stops, at one-second resolution.** Taking
arm A's own clock window (decode begins 9 s after launch, runs 113.8 s), the
`disk0` series over 112 one-second samples reads:

| mean | median | min | max | samples at zero | samples under 400 MB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1359.5 MB/s | 1353.0 | 753.0 | 1685.7 | **0** | **0** |

Against the four-thread `ParallelExpertReader` ceiling recorded in §5o and
`docs/v4-core-design.md` (3.92 GB/s) the drive sits at **34.7% duty**, so
the operator's instinct that the SSD has spare capacity is correct. What this
table does *not* show, and what §5w found, is that "no sample at zero" was a
statement about the sampler's resolution: a 1 Hz series averages a 51 ms burst
into the second around it, so alternation at the scale that matters to a 217 ms
token is invisible here. Reading 2 was the mistake that followed from that.

**Result 2 — what a perfect feeder could still buy is 0.075%.** *Refuted by §5w,
kept because the arithmetic is the reason the refutation matters.* `io_ms=72.254`
is the time spent in miss reads per token and `io_hidden_pct=99.77` says how much
of it already runs under compute, so the exposure is `72.254 × 0.0023 = 0.166` ms
of a 222.27 ms token (1000/4.499), and a feeder that removed *all* remaining
exposure moves 4.499 → 4.502 tok/s. The engine's `expert_load_p50/p95/p99` of
2/8/16 ms and the `io_host_waits_avoided=8944` of 15,584 waits are the machinery
already spending that idle capacity. The flaw is the instrument, not the
arithmetic: `io_hidden_pct` compares the read's span with the completion
timestamps of the command buffers that ran alongside it, so it answers "did the
*host* have to block", and the routed command buffer is event-gated
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift:372`) —
the *device* blocks instead, and that is the time the token pays. §5w measures
the device side.

**Result 3 — the continuous feeder already exists, and its limit is RAM, not the
drive.** `prefetch_issued_per_token=39.54` against `prefetch_adopted_per_token=18.23`
means the ring issues ~20,244 speculative reads over the run (`spec_ops=20243`)
at full stride width — about 493 MB/s of pure prediction, on top of the 920 MB/s
the demand misses cost. Demand plus speculation is 1413 MB/s against the 1359.5
measured, a 4% cross-check that the two series account for the whole drive load.
The ring reports `free=0.99` — it runs with about one spare slot — so it is
throttled by where a read can *land*, not by bandwidth. This is the sentence the
feeder concept dies on: a constant reader already has the bandwidth it wants, and
cannot use more of it because the 96-slot tier has no room for the bytes.

**The misses are retention-shaped, not cold-read shaped.** Of 37,807 demand misses,
**33,239 are reloads** of experts already loaded once, against 42,535 evictions, at
hit rate 0.8472. Reading earlier does not help an expert that was evicted because
nothing was holding a slot for it; that is a placement and capacity question, and
capacity at 12 GiB is the operator's own ceiling on this machine.

**The offline feeder simulator, and why its rates are withheld.**
`benchmark/continuous_feeder_sim.py` (new) replays a route trace as a timeline —
one serial drive, the engine's LFU-with-LRU-tiebreak placement, and speculative
reads that may only issue in the intervals the demand reader leaves free — under
five arms: `burst` (control), `prev_token`, `recency3`, `hot_set8`, `recurrence`
(read an expert back before its settled recurrence interval says it is due). 26
tests in `benchmark/test_continuous_feeder_sim.py`, registered in CI's model-free
python list. On the real capture — committed as
`benchmark/expert-cache/route-trace-qwen38-4bit-v520-20261015.txt.gz`, arm B's,
24,528 layer steps after `--skip-steps 480`, 96 slots/layer — it reports:

```
burst       demand= 56433 spec=      0 cov=0.000 waste_MB=     0.0 exposed_ms=   9375.2 duty=0.446 REFUSED
prev_token  demand= 54908 spec=   2732 cov=0.027 waste_MB=   257.5 exposed_ms=   9117.1 duty=0.457 REFUSED
recency3    demand= 53136 spec=   7098 cov=0.058 waste_MB=   888.8 exposed_ms=   8682.3 duty=0.480 REFUSED
hot_set8    demand= 54158 spec=   4267 cov=0.040 waste_MB=   473.5 exposed_ms=   8658.0 duty=0.465 REFUSED
recurrence  demand= 54291 spec=  22533 cov=0.038 waste_MB=  9228.7 exposed_ms=   8138.5 duty=0.616 REFUSED
calibration: REJECTED (burst projects 5.72 against measured 4.50 tok/s, 27.1% off)
```

The control arm is 27.1% too fast against the clock measured in this same
section, which means the timeline is missing ~47 ms/token of per-step cost. The
tool now says so and exits 1 instead of printing quotable rates — the gate is tested
(`test_refuses_every_projection_when_the_control_arm_misses_the_measurement`).
Two readings survive the rejection, because they are within-arm and structural:
the `prev_token`/`recency` shape removes 0-6% of demand reads while the resident
set is the last-seen set — a due-soon predictor is the only shape that spends idle
drive time usefully — and `recurrence` buys its 13% exposure reduction at 9.2 GB
of wasted reads, 20× the waste of the next arm. Neither is a token-rate claim.

**Deviations, all of them.** (1) Arm B's 4.733 tok/s is **not** a clock: the trace
capture does an unbuffered `write()` per layer per token (~480 syscalls/token), so
its rate is perturbed by design and only its trace was used. (2) The `ioreg` AGX
"Device Utilization%" reader showed 61% with the machine idle and 69.6% mean during
decode, so it is not an occupancy instrument on this host and no conclusion here
rests on it — §5u's <15%-at-launch requirement cannot be met by that gauge. (3) One
run per arm, not a paired batch: the finding is a counter and a duty cycle, not a
few-percent clock delta. (4) Swap was already 5.47 GB used at launch. (5) The `disk0`
series is whole-volume, so it includes this machine's other I/O; the 4% agreement
with the engine's own byte counters is what bounds that error.

**Verdicts.**
- **The chunk-by-request → continuous-feeder change is closed by measurement, not
  by §8.** *Amended by §5w:* the prize was 0.075% only under `io_hidden_pct`, which
  measures host blocking, and the routed command buffer is event-gated so the device
  blocks instead. The device-side ledger prices the same concept at **52.8 ms of a
  217 ms token (24%)**, and §5w's verdicts keep the closure — the burst is already at
  97% of the drive's ceiling — on a different and correct basis. "0 idle samples in
  112" stands as a statement about a 1 Hz sampler, not about the drive.
- **The live headroom is host-side, and it is bigger than everything the SSD path
  can still give.** Named per-token spans in arm A read `head_ms=11.849`,
  `rdadvise_ms=7.863` over 35,430 calls, `wait_ms=109.113`, `body_ms=202.936`,
  `cache_plan_ms=4.877`, `gather_ms=2.329`, `loop_*=2.82` — and because `body_ms`
  overlaps `wait_ms` these cannot be summed into a budget. That the footer's spans
  do not decompose a 222 ms token is itself the next instrument to build, and it is
  where a CPU lane belongs: bookkeeping off the critical path, not tensor compute
  (which §8 and the operator's no-quality-loss rule both keep closed).
- `free=0.99` is the number to watch if this is ever reopened: it says the ring is
  one spare slot from starved, and any change that frees expert-cache slots without
  giving up residency converts directly into feeder room.


## 5w. The device-side token ledger — the feeder premise was right and §5v's instrument could not see it (2026-10-15)

**What this section overturns.** §5v priced the continuous-feeder idea at 0.075% of a
token on the strength of `io_hidden_pct=99.77`, and reported the drive's 1 Hz duty
series as proof that there was no burst/idle alternation to exploit. Both readings are
wrong, and they were wrong for one reason: each measured a *host* or a *sampler* where
the question is about the *device*. This section measures the device. It lands on the
operator's side of the argument — the SSD really does burst and then wait, and the wait
is 52.8 ms of a 217 ms token — and it also shows the prize is not collectable the way
§5v assumed, because the burst is already running at 97% of the drive's measured ceiling.

**Why `io_hidden_pct` cannot answer this question.** The counter is defined as
`100 × (1 − exposedIo/missIo)` (`sources/TinyTitanLib/ServerModelSession+Diagnostics.swift:324-329`),
and `totalExposedIoNanos` is built by comparing a read's span against the *completion
timestamps of the command buffers that ran alongside it*
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift:298-321`,
`RealForwardRunner+DecodeMoEFinalize.swift:44-58`). So it answers "did the **host**
have to block". It cannot answer "did the **GPU** go idle", because the routed command
buffer is event-gated on the device —
`routedCB.encodeWaitForEvent(token.event, value: token.value)` at
`sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift:372` — which
means that when bytes are late, the device waits and the host does not, and the counter
reports the token as essentially fully hidden. The engine already exposes the device-side
truth: `recordKernelGPU` feeds `kernelGPUOccupancy()`, which merges intervals so
overlapping buffers are not double-counted, and `kernelGPUGaps()`, which charges each
true idle interval to the role pair it separates
(`sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift:42-140`).
Nobody had published those two lines for this build.

**Provenance and protocol.** Tree `8c61f14`; `git diff --stat v5.20..HEAD` touches
`benchmark/`, one workflow file and `docs/` and **no file under `sources/`**, so the
binary is the released v5.20 engine. 24 GB M3 (Mac15,3), macOS 27.0.1, Swift 6.4
(`arm64-apple-macosx27.0.0`), AC power, `System-wide memory free 83%`, empty
model-process guard before launch, one CLI process at a time, **no engine flags** — the
12 GiB tier is this host's own derivation. Prompt "Write a detailed essay about the
history of computing.", `--temperature 0`, greedy. Capture arm `--max-new 512` with
`TINYTITAN_RUNNER_STATS=1 TINYTITAN_KERNEL_STATS=1`; the four A/B arms `--max-new 384`
with the same two env vars. Exit 0 in all five. Swap was **7.63 GB used before the
batch and 7.90 GB after**, which is well above the comfort line and is reported as a
deviation below. Drivers `/tmp/occupancy_probe_run.sh` and `/tmp/gap_slot_ab.sh`, data
`/tmp/occupancy-probe/` and `/tmp/gap-slot-ab/`.

### The ledger of one decode token (capture arm, 4.601 tok/s)

`busy_ms=58689.240 span_ms=115393.980 occupancy=50.9% busy_share_of_decode=52.7%`,
against `decode_s=111.283` for 512 tokens:

| part of a 217.34 ms token | ms/token | how it was obtained |
| --- | ---: | --- |
| GPU busy | 113.00 | `busy_ms/512`, minus the 832 ms of kernels whose role names are `prefill_*` |
| GPU idle, in named gaps | 101.98 | sum of the six gap pairs whose counts are decode-sized (table below) |
| unattributed | 2.36 | 1.1% of the token |

The ledger closes to 1.1%, and it is a *partition* rather than a pile-up: `kernelGPUGaps()`
advances a single frontier, so no interval is counted twice and overlapping command
buffers manufacture no gap. The caveat is that `busy_per_token_ms` as printed divides the
whole session's busy time by *decode* tokens, so the prefill rows would be smeared across
512 tokens; the 832 ms subtraction is what makes the first line honest, and without it the
closure would look better than the measurement is (the printed figure is 114.63).

**GPU occupancy is 50.9%, and the idle is not spare worker capacity — it is two
mechanisms, and only one of them is the drive:**

| gap pair | ms/token | count | ms per occurrence |
| --- | ---: | ---: | ---: |
| `moe_phase1_hit → moe_phase1_miss_fixup` | **52.803** | 15,549 | 1.739 |
| `shared_expert → moe_phase1_hit` | 26.766 | 15,499 | 0.884 |
| `shared_expert → moe_phase1_2_routed` | 16.070 | 8,964 | 0.918 |
| `embed → attn_norm_qkv` | 3.008 | 511 | 3.014 |
| `moe_phase2_routed → attn_norm_qkv` | 1.842 | 15,298 | 0.062 |
| `moe_phase1_miss_fixup → moe_phase2_routed` | 1.493 | 15,564 | 0.049 |
| *(prefill-only pairs, excluded above)* | 6.238 | 295 | 10.83 |

The largest single line is the answer to the feeder question: **a quarter of every token
(52.8 of 217.34 ms) is the device sitting idle between the hit-phase kernels and the
miss-fixup kernels, waiting for bytes.** Of that, `io_completion_to_fixup_ms=1.6549` is
host submit latency — 3.1% — and the remaining 51.1 ms is the read itself.

### The burst is at the ceiling, so the bill is the bytes, not the pipe

`expert_read_mib / tokens` divided by the four-thread `ParallelExpertReader` ceiling
recorded in §5o and `docs/v4-core-design.md` (3.92 GB/s) predicts the gap without using
it. Five arms, spanning 1.87× in read volume:

| arm | slots | tok/s | MB read/token | predicted gap | measured gap | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| armA | 96 (default) | 4.601 | 203.55 | 51.93 ms | 52.803 | 1.017 |
| r1a | 96 (default) | 5.015 | 207.63 | 52.97 ms | 56.348 | 1.064 |
| r2a | 96 (default) | 5.050 | 207.01 | 52.81 ms | 53.872 | 1.020 |
| r1b | 48 | 4.574 | 380.19 | 96.99 ms | 96.056 | 0.990 |
| r2b | 48 | 4.398 | 380.72 | 97.12 ms | 102.999 | 1.061 |

The prediction is `bytes ÷ 3.92 GB/s` and nothing else — no tuning, no fitted constant —
and it lands within 6.4% on every arm. Two more identities pin the mechanism:
`expert_read_mib / expert_misses` is **2.7689 MB in all five arms, exactly `expertStride`**,
so a miss costs one full stride and read volume is fully explained by the miss count; and
`gap_total / gap_count ÷ (experts_per_miss_layer × stride)` puts the *delivered rate inside
each burst* at 3.85, 3.68, 3.84, 3.95 and 3.69 GB/s — **mean 3.80 GB/s, 97% of the ceiling**.

So the operator's description of the machine is correct and the diagnosis that followed
from it is not. The SSD does stop and start — §5v's 1 Hz sampler simply could not see a
1.7 ms burst — but when it is asked for bytes it already delivers them at essentially the
full rate the drive was ever measured capable of, in bursts of 1.74–2.51 ms. There is no
idle pipe to fill that would make these particular bytes arrive sooner. What would help is
moving them *earlier*, which needs them to be known earlier, and the router decides which
experts a layer needs only after that layer's routing readback. The ring already tries to
beat that clock and lands 18.56 of 73.51 demand misses per token (25%) with `free=0.99` —
about one spare landing slot.

### The falsifier, run before measuring

The claim under test was *"the gap is the drive delivering bytes."* The falsifier written
down first: cutting the cache to 48 slots must raise the miss count and the gap together;
if the gap stays put while misses move, the reading in this section is wrong. Arms ran
interleaved `r1a(96), r1b(48), r2b(48), r2a(96)` — the shipped arm first and last, as the
drift check — and the counters were near-deterministic while the wall clock was not:

| paired round | miss/token 96 → 48 | gap/token 96 → 48 | tok/s 96 vs 48 | Δ |
| --- | ---: | ---: | ---: | ---: |
| round 1 | 74.99 → 137.31 (×1.831) | 56.348 → 96.056 (×1.705) | 5.015 vs 4.574 | +0.441 (+9.6%) |
| round 2 | 74.76 → 137.50 (×1.839) | 53.872 → 102.999 (×1.912) | 5.050 vs 4.398 | +0.652 (+14.8%) |

**The falsifier did not fire.** Misses ×1.83, gap ×1.70–1.91, same direction in both
rounds. Mean paired advantage of the shipped 96-slot default: **+0.546 tok/s, +12.2%**,
with the two rounds agreeing in direction and disagreeing in magnitude by 5 points of
percent — the spread that §5v's single runs would have had to guess. The response hash is
`95aa4531b4ea2304` in **all four** arms, so the slot count moves no numerics at all: this
is a scheduling and residency change with byte-identical output, and the 12.2% is pure
queueing.

**And the instrument correction, in its strongest form:** across those same four arms
`io_hidden_pct` reads 99.84, 99.90, 99.86 and 99.85 — a spread of 0.06 points — while the
clocks differ by 12.2% and the read-bound gap by 70%. `wait_ms` even *falls* (92.562 →
74.139) as `io_ms` *rises* (70.939 → 105.441), because at 48 slots the host is never the
one blocked. An instrument that varies by 0.06 points across a 12.2% effect cannot bound
that effect at 0.075%; §5v's Result 2 was reading the wrong clock.

### The two idle families trade against each other

The second family — `shared_expert → moe_phase1_*`, 42.835 ms/token in the capture arm —
is host turnaround: router readback, `cache_plan_ms=4.898`, `gather_ms=2.248`,
`rdadvise_ms=7.624` over 35,260 calls, `cb1_ms=5.079`, encode. The shared MLP
(`shared_expert` = 5.142 ms/token of GPU work) is what is queued to cover it, and it is
far too small. But this family is **not independent of the first**: at 48 slots it halves
to 19.930 and 20.386 ms/token while the read-bound gap doubles (those totals include a
`shared_expert → moe_phase1_miss_fixup` pair worth 0.975/1.124 ms/token that does not
appear at 96 slots at all). Per occurrence the dominant pair *falls*
from 0.730 to 0.393 ms. The reason is queue depth — when the drive is behind, the GPU has
already-queued work to chew on and the host's bookkeeping disappears under it; when the
drive is ahead, the host is exposed at the front of an empty queue. So the 102 ms of idle
is one queueing system, not three prizes, and a change that "fills" family 2 with work
that then has to wait on the same bytes buys nothing.

### Answer to the tri-worker premise

The premise was: GPU is busy but the ANE and CPU are idle, so reassign slices. The ledger
says the machine is 50.9% occupied and the whole idle half is attributable to (a) bytes
that are not yet known and (b) host bookkeeping the dependency chain will not let the GPU
start earlier. A second worker could only take a slice that does **not** depend on the
router readback, and the ledger names what those are: the shared MLP (5.1 ms/token,
already on the GPU, already queued ahead of the wait — `RealForwardRunner+DecodeLayers.swift:211-223`),
the sampler (0.285 ms/token), and the embeddings (0.106 ms/token). Together 5.5 ms of a
217 ms token, §8's CPU co-execution arm already measured that transfer at **−22.6%** on
bus contention, and the ANE decode path is unreachable. That is the honest size of the
tri-worker idea on this engine: the idle is a dependency chain and a drive, not an
unassigned worker.

**Deviations, all of them.** (1) The capture arm is a *capture* arm:
`TINYTITAN_KERNEL_STATS=1` records an interval per command buffer, so 4.601 tok/s is not
a published level and none of the rate claims above use it — the paired arms' 5.015/5.050
against 4.574/4.398 are the levels. (2) n=1 for the occupancy run; its role and gap
per-token figures are reproduced in shape by all four A/B arms but only the gap magnitude
was re-measured. (3) Swap was 7.63 GB used before the batch and 7.90 GB after, and the
48-slot arm ran with a smaller resident tier against that pressure — this is a plausible
cause of round 2's slower 48-slot pair, and it is why the paired *direction* rather than
the paired *magnitude* carries the finding. (4) The gap list is truncated at the engine's
own top-8, so the 2.36 ms residual includes small real gaps that were never printed; the
closure figure is therefore an upper bound on the attribution, not an exact balance.
(5) The `prefill_*` subtraction from `busy_ms` is by role name, not by timestamp, so any
prefill work recorded under a decode role name stays in the 113.00 line. (6) `--max-new`
differs between the capture arm (512) and the A/B (384); the counters are per token and
the two agree — 203.55 vs 207.63 MB/token, 52.803 vs 56.348 ms/token gap — but they are
not the same run and are not treated as one.

**Verdicts.**
- **The continuous-feeder premise is confirmed as a description and closed as a lever.**
  The drive does burst (1.74–2.51 ms) and idle, §5v's sampler could not see it, and the
  burst costs 52.8 ms/token — 24% of the token, not 0.075%. But the burst already runs at
  97% of the drive's measured ceiling, so the remaining lever is *earlier bytes*, not
  *faster bytes*, and earlier bytes need either prediction that lands (slot room, `free=0.99`)
  or a router result that arrives sooner. Neither is a feeder rewrite.
- **The slot count is worth 12.2% and it is the largest single knob measured in this
  campaign** — larger than every §6/§8 arm here. It is already at the maximum the
  operator's 12 GiB cap on this 24 GB Mac allows, and §5q's rule is what sets it. Nothing
  to ship; everything to remember if a bigger host appears, because this section says a
  host with room for ~200 more slots should be measured as a ~12% decode win, not a rounding error.
- **The open seam is the 42.8 ms/token of host turnaround, and it is the only part of the
  idle that is neither drive-bound nor RAM-bound.** §5m already priced the cheapest version
  of it (rdadvise issue-count coalescing) at ≤2–3% and parked it on paired evidence, so a
  real attempt needs a different shape: not fewer syscalls but bookkeeping issued a layer
  ahead of the wait. That is an implementation question, not a measurement, and it is where
  the next session should start.
  *Closed by §5bg on measurement, not elimination: the host holds 20.944 ms/token inside a
  layer iteration and 40.046 ms/token that is not a blocked wait at all, so 43.006 exceeds
  even the absolute bound by 2.960 — and the largest removable in-body term is the one §5m
  already priced negative. Do not reopen it without a counter this page does not have.*
- **`io_hidden_pct` is unfit for any future overlap claim.** Keep the counter — it is the
  right instrument for host blocking, and `io_host_waits_avoided=8964` of 15,564 waits is
  a real achievement — but a percentage that moves 0.06 points across a 12.2% clock change
  must not be used to bound one. The device-side lines (`TinyTitan kernel busy_ms=…`,
  `TinyTitan gap …`) are the ones to publish, and they are already in the build.

## 5x. The completion wait is already the cheapest primitive, and what an extra command buffer costs (2026-10-15)

§5w left ~0.44 ms/layer of GPU idle that no host counter names, and the decode loop blocks once
per layer on `cb.waitUntilCompleted()` (`RealForwardRunner+CommandBuffers.swift:16`). The obvious
suspect was the wait itself: a blocking wait sleeps, and waking a thread costs tens of
microseconds, so if the *wake* were the 0.44 ms, a spin-then-block wait would recover on the order
of 20 ms of a 217 ms token. Measured directly, it is not.

`/private/tmp/waitcost/main.swift` is a throwaway spike, not engine code: it commits a 4 KiB blit
and waits for it, 4 000 iterations per arm, on an idle machine on AC power with no model process
running (record in `/private/tmp/waitcost/run2.log`).

| arm | mean | p50 | p95 | p99 | min |
|---|---|---|---|---|---|
| `waitUntilCompleted()` | 126.3 µs | 101.6 | 179.2 | 850.0 | 74.3 |
| `cb.status` spin | 245.1 | 158.8 | 697.6 | 2438.1 | 80.0 |
| `sched_yield` spin | 253.1 | 165.4 | 721.9 | 2533.3 | 95.8 |
| blocking, repeated last | 157.7 | 103.1 | 252.1 | 904.8 | 76.5 |

**The blocking wait is the cheapest arm by 56% on p50 and its tail is 3× better than either spin.**
Polling `cb.status` costs more than sleeping on it: the driver's completion path beats a userspace
loop, and a spinning loop also holds a core the engine's own host bookkeeping needs. There is no
spin-first wait to ship. So the ~0.44 ms/layer from §5w is not a scheduler artefact — it is real work
the host is doing, which is where §5w's own verdict pointed, and it cannot be recovered by changing
*how* the engine waits.

### The floor the other direction has to pay

The same spike priced a command buffer with no wait at all: `encode` p50 3.9 µs (mean 22.3, p99
368.8), `commit` p50 0.5 µs, per-command-buffer host cost mean 22.9 µs — 6.59 ms/token at 6 command
buffers × 48 layers. So the open direction is the opposite of finer slicing: **fewer, larger command
buffers**. Every added commit-and-wait pair costs the ~74 µs wait floor this table shows plus ~23 µs
of host time, which is ~10 ms/token for one extra pair per layer — a tenth of the idle it would have
to explain.

**Deviations, all of them.** (1) The spike never loads a model, so it prices primitives, not the
token; the wait figures are a floor on wake cost with the GPU otherwise empty, not an estimate of a
decode layer's share. (2) An earlier, unsaved run of the same arms gave blocking p50 101.7 / min 73.3
and a `sched_yield` p50 of 137.2 against 165.4 here; the two spin arms trade places between runs, so
this section publishes only what is stable across both — the blocking arm is the cheapest, p50 ≈ 102
µs with a ~74 µs floor, and both spin arms are clearly worse. (3) `min` columns are the best of 4 000
samples, not a bound. (4) n=2 runs, same process shape, no interleaving — enough for an ordering, not
for a magnitude difference between the two spin arms.

## 5y. The miss replay is calibrated, and what it says first is that the slot *partition* costs more than the victim rule (2026-10-15)

§5w made capacity the largest knob on this machine (+12.2% decode for a 45% miss change, byte-identical
output), so the next feeder or placement idea should be screened offline before it costs a model window.
The screen already existed in `benchmark/continuous_feeder_sim.py` but was known-biased and loosely banded;
this section calibrates it, states what it may and may not be used for, and then runs the first experiment
through it.

### Calibration against the run that wrote the trace

Arm B of §5v is the capture that produced `route-trace-qwen38-4bit-v520-20261015.txt.gz`, so footer and
replay are one workload: 511 decode tokens, `expert_misses=37724`, `prefetch_adopted_per_token=18.40`.
The engine's own two counters bracket what a route trace can see — a replay must count more than the drive
*reads* (it cannot know which misses the prefetch ring had already staged) and no more than reads plus
adoptions (the placements it really made):

| quantity | engine | replay | error |
|---|---|---|---|
| drive reads/token (floor) | 73.82 | — | — |
| placements/token (ceiling) | 92.26 | **89.28** | −3.2% |
| 48-slot ÷ 96-slot miss ratio | 1.831 (§5w paired) | **1.794** | −2.0% |
| cold-cache consistency | placements − evictions = 4 610 | one cold cache = 48 × 96 = **4 608** | 2 slots |

The last row is the structural check: if the replay's placement count and the engine's eviction count do not
differ by about one cold cache, the two are not counting the same object. They do, to 2 slots in 4 610.

**Licensed — ranking placement rules at a fixed budget. Banned — quoting a token rate from it.** §5w's
0.637 ms of critical path per avoided miss was measured in drive *reads*; the ~15.5 placements/token the
ring served without one are invisible in a route trace and not fixed, because they depend on how deep the
ring ran. A replay delta has to be scaled by that unmeasured share before it becomes money, and the
timeline that would supply it is the refused model (§5v).

### Which detail of the engine's rule is load-bearing

The plan-faithful rule (`_PlanSlots`) and the textbook LFU-with-LRU-tiebreak (`_Slots`, the one the refused
timeline uses) differ by 69% on this trace — 89.28 against 151.20 placements/token. That gap was previously
described in this tree as belonging to three transcribed details of `makeExpertCachePlan`. Measured one at a
time, none of the three is it:

| arm | placements/token | vs shipped |
|---|---|---|
| plan rule as shipped | 89.28 | — |
| use count raised only on misses, not on hits | 88.24 | −1.17% |
| `useClock` advanced per expert instead of per plan | 89.31 | +0.03% |
| slots re-ranked for each miss instead of once per plan | 89.28 | +0.00% |
| expert's count **forgotten when its slot is evicted** | 102.17 | +14.4% |
| textbook `_Slots` rule, whole | 151.20 | +69.3% |

The load-bearing detail is count lifetime, and it is in the engine, not in the ranking: `expertUseCount` is a
per-layer array sized by `layout.expertsPerLayer` (`PreadExpertStreamer.swift:326`) and only
`resetExpertUseCounts()` zeroes it — which is itself env-gated (`TINYTITAN_EXPERT_USE_RESET=1`,
`RealForwardRunner+Prefill.swift:24`) and therefore **off** in every flag-free arm measured in this document.
A returning expert is priced by everything it has ever done. The textbook rule prices it at one.

**Withdrawn figures.** Four numbers published earlier in this campaign's working notes — 109.10, 111.64,
113.39 and 157.11 placements/token for a "textbook" replay — are withdrawn. The first three were three
different miss *accountings* of the same rule (no warm-up; warm-up counting every visit; warm-up counting
decode only), and 157.11 came from a probe that leaked residency: `_Slots.contains(e)` is `e in use_count`,
so restoring a count for an evicted expert makes it look resident, which is also how a broken variant of the
same probe once collapsed a miss count to 22.92. None of them is reproducible from the tree. The band claim
is now pinned by a test rather than by prose: `MissReplayTests` asserts the plan rule sits inside
[73.82, 96.87] and a drift arm asserts the textbook rule breaches the ceiling at 151.20 — the same assertion
run against the plan rule fails at 89.28, so the band discriminates.

### The ranking: capacity distribution beats victim order

Two throwaway rounds screened victim rules and budget shapes at the shipped 4 608-slot total
(`/tmp/placement-screen/`, uncommitted — see deviations). Ranking only, no token rate anywhere in these rows.

| rule and budget shape | placements/token | vs shipped |
|---|---|---|
| oracle static per-layer split (sum 4 587 slots) | 82.66 | −7.42% |
| LFU capped at 3 + one flat 4 608-slot pool | 82.97 | −7.07% |
| LRU + flat pool (capped at 3 or 8 — identical) | 85.50 | −4.24% |
| oracle static split, plain LFU inside it | 86.08 | −3.59% |
| LFU capped at 5, per-layer 96 | 86.95 | −2.61% |
| `lru_then_lfu`, per-layer 96 | 87.46 | −2.04% |
| **shipped: LFU, per-layer 96** | **89.28** | — |
| LRU, per-layer 96 | 91.05 | +1.98% |
| LFU + flat pool, uncapped | 104.56 | +17.11% |
| FIFO / random, per-layer 96 | 106.64 / 106.43 | +19.4% / +19.2% |

Read the middle of the table, not the top: the shipped rule is already the best *victim order* at this budget
(every pure alternative here is worse), and the wins come from **how the 4 608 slots are divided**, not from
who gets evicted. The per-layer decode working sets are very unequal — 174 to 415 distinct experts per layer,
mean 285, against a uniform 96-slot partition — and an oracle split of nearly the same total recovers 7.4% of
placements. A single flat pool with LRU recovers 4.2% without knowing the future; combining it with a
saturating count (cap 3) recovers 7.07%, which is within noise of the oracle. The uncapped LFU in a flat pool
is much *worse* (104.56): without a cap, one expert's lifetime count can defend a slot for the whole session,
and in a shared pool that starves every other layer — which is why the cap and the pool help together and the
cap alone is a 2% story.

Capacity ladder for reference (per-layer, LFU / LRU): 48 slots 160.13 / 157.77, 64 131.22 / 131.59, 96 89.28 /
91.05, 128 61.81 / 63.52, 192 34.77 / 34.53.

**Deviations, all of them.** (1) One trace, one prompt, 511 decode tokens; the §5w ratio it is calibrated
against came from a 384-token run, so the calibration compares a *response*, not a level. (2) The screen
scripts and their logs are uncommitted throwaways under `/tmp/placement-screen/`; only the shipped arm
(89.28), the 48-slot arm (160.13) and the drift arm (151.20) are reproducible from the tree, via
`benchmark/test_continuous_feeder_sim.py`. A reboot costs the ranking rows. (3) Deltas of 2–4% are inside the
instrument's own 3.2% level bias, and §5w showed a 45% miss change buying 12.2% of clock — a 4% placement win
may express as ~1% or as nothing once the ring and `free=0.99` absorb it. (4) The oracle split uses future
knowledge (per-layer distinct counts read off the same trace); it is published as the ceiling a static split
could reach, not as a policy. (5) The `window` arms at 32 and 128 tokens returned exactly the baseline because
the cutoff (`window × 48` rows) covers the whole session — those two are **inconclusive, not "no effect"**;
the window-8 arm (88.22, −1.19%) is the only one that tested the shape. (6) An earlier `lfu_decay` arm was a
no-op by construction — halving every count is a uniform rescaling and cannot change a ranking — caught before
any of it was reported. (7) The 128- and 192-slot rows exceed the operator's 12 GiB cap on this 24 GB Mac and
are offline arithmetic, not configurations to run here.

**Verdicts.**
- **The victim rule is not the open seam.** Shipped LFU beats LRU, FIFO and random at this budget, and every
  tiebreak reshuffle tried here moved placements by under 3%. Do not spend a model window on a cleverer LRU.
- **The partition is.** 512 experts per layer and a uniform 96 slots each, against per-layer working sets of
  174–415. Any real attempt should rebalance the *budget across layers* (or share one pool), not reorder
  evictions. Both are architectural here: `slotCount` is a `let` per streamer
  (`PreadExpertStreamer.swift:93`) and the plan guard assumes one uniform capacity per layer, so a non-uniform
  split is a load-time data-layout change, not a tuning knob. It needs the operator to ask for it.
  **— Corrected by §5z.** The streamers are already one per layer with their own `slotCount`, and the split is
  one expression (`Model+LayerStreamers.swift:168`); the "architectural" claim was wrong. §5z also shows this
  lever's own oracle ceiling is ~1 point of hit rate, so it is not worth the change.
- **One cheap live arm survives this screen**: a saturating use count (cap ~5) inside the comparator. It is a
  clamp in `makeExpertCachePlan`, changes only which slot is given up, and should be output-invariant —
  residency is a cache, not semantics, and §5w already showed a placement change holding the bytes identical.
  Offline it is worth −2.6% placements, which is *inside* the instrument's own 3.2% bias: that licenses one
  paired run, not a claim.
- **Nothing here is a tok/s number.** This section prices placements. The engine's clock is still what §5w and
  §5v measured, and no row above has been run on the machine.

## 5z. A heat split learned on one session does transfer to another — and the bar it has to clear is arithmetic, not empirical (2026-10-16)

§5y ended by naming the partition as the open seam and calling a non-uniform split architectural. Both halves of
that are wrong, and this section is the experiment that says so. It also answers the question §5y could not: a
split can only ship if it is learned from an **earlier** session, so the number that matters is not what an
oracle split wins on its own trace, but what a split carries across captures.

Everything below is reproducible from the tree. `benchmark/heat_split_sim.py` (new) plus
`benchmark/test_heat_split_sim.py` (17 tests) commit the screen that §5y left as a `/tmp` throwaway, together with
the third route capture it needed. Same licence as §5y: **ranking budget shapes at a fixed slot total. Banned:
quoting a token rate from any of it.**

### The captures

| capture | decode rows from | distinct experts/layer | demand | what it is |
|---|---|---|---|---|
| `20261011` | position 30 | 146..409, mean 261 | 480 req/token | pre-v5.20 build, the campaign's first trace |
| `v520-20261015` | position 10 | 174..415, mean 285 | 480 | §5y's arm B, the shipped 96-slot default |
| `v520-prompt2` | position 33 | 122..417, mean 269 | 480 | **new** — same build, a different prompt |

`v520-prompt2` is the falsifier's control: `benchmark/expert-cache/run-record-qwen38-4bit-v520-20261016-prompt2.txt`
fixes its provenance (machine, build, exit code, footer counters) and calibrates it the way §5y calibrated arm B.
Its replay is 102.10 placements/token against an engine bracket of 94.27 (drive reads) to 114.66 (reads plus
adoptions) — **−10.8%** from the ceiling. §5y's bias on its own capture was −3.2%. The replay's level error is
workload-dependent, which is a second, independent reason the tok/s ban is not caution but arithmetic: the bias is
larger than every effect this section measures.

Deviation on that run: the argv was never logged, so the run-record says so and the command line is treated as
unrecovered. The identity lines and counters in that file are the evidence; nothing else about the invocation is claimed.

### Do two sessions of the same model route the same experts?

| pair | profile correlation | shared rows | identical rows | picks in common |
|---|---|---|---|---|
| `20261011` × `v520-20261015` | +0.976 | 18 384 | **0** | 13.206% |
| `20261011` × `v520-prompt2` | +0.960 | 18 240 | **0** | 5.117% |
| `v520-20261015` × `v520-prompt2` | +0.957 | 18 384 | **0** | 3.209% |

Chance for a top-10 pick out of 512 is 1.953%. So the two same-prompt captures agree on 13.2% of picks — about 7×
chance, which is the prompt-memory an engine could in principle exploit — but the different-prompt pair sits at
3.2%, barely above chance, and **no row in any pair is routed identically**. Meanwhile the per-layer *shape*
correlates at +0.957 to +0.976 everywhere, including across the pair whose routes are nearly independent.

That is the result that decides whether a shipped table is data or memorisation: **which layer is busy is a property
of the model; which expert is busy is a property of the conversation.** A per-layer budget table is therefore
portable in the way a per-(layer, expert) table would not be.

### The transfer table

`split_slots` spends **exactly 4 608 slots** — the shipped tier — in proportion to one capture's heat profile, floor
10, ceiling 512. Each row replays that budget on a *different* capture. All figures are placements/token at cap
none, engine rule as shipped.

| scored on | shipped uniform | oracle (own trace) | transfer from the other two |
|---|---|---|---|
| `20261011` | 91.77 (80.88% hit) | 88.38 −3.69%, +0.71 pt | 88.58 (+0.66 pt, keeps 94.0%) / 88.25 (+0.73 pt, keeps 103.7%) |
| `v520-20261015` | 89.28 (81.40% hit) | 85.54 −4.19%, +0.78 pt | 85.32 (+0.82 pt, keeps 105.9%) / 84.91 (+0.91 pt, keeps 117.0%) |
| `v520-prompt2` | 102.10 (78.73% hit) | 97.69 −4.32%, +0.92 pt | 98.36 (+0.78 pt, keeps 84.9%) / 98.56 (+0.74 pt, keeps 80.3%) |

Six transfers, six better than the shipped uniform split, and four of them beat an oracle that read the future of
the very trace it is scored on. With the saturating count added (cap 3, cap 5) all eighteen transfers still win;
the keeps-fraction column, like-for-like at each cap, runs 74.4% to 128.3%.

### The bar, and why no budget shape reaches it

`docs/plan-qwen38-decode-12gb.md:150` sets B1's acceptance as **≥ +7 points of hit rate** over the reproduced
control. The measured transfers deliver **+0.66 to +1.52 points**. That is a factor of four to eleven short — and
the miss is not a transfer penalty, it is a ceiling: the *oracle* split, which knows the trace it is scored on, only
reaches +0.71 to +0.92 points. Every static partition of 4 608 slots is bounded by that.

The reason is units. Hit rate is `1 − placements / demand` with demand 480, so a percentage of placements buys
one-fifth of a point per percent: §5y's headline −7.4% of placements is 6.6 placements, 1.4 points. To buy +7 points
at this tier you must remove 33.6 placements/token, i.e. cut 89.28 by **37.6%**. The only arms in this campaign's
ladder that large are 128 slots/layer (−30.8%) and 192 (−61%), and §5y's deviation 7 already records that both blow
past the operator's 12 GiB cap on this 24 GB Mac.

So the plan's bar and §5y's percentage were never commensurable, and the honest reading is the strong one: **at a
fixed 4 608-slot budget, capacity *shape* is worth at most ~1 point of hit rate.** For contrast, §3/§4 measured
Belady at the same 4 608 slots at 89.8% hit — +8.4 points over the shipped 81.40%. The headroom at this budget is
real and it is nearly all in *when a slot is given up* relative to when it is needed, which is the one thing a static
partition cannot touch and a feeder can.

### Corrections to §5y, named

1. **"Both are architectural here: … a non-uniform split is a load-time data-layout change, not a tuning knob."**
   False, and the plan had already designed the seam. Verified in the engine:
   - `Model+LayerStreamers.swift:195-202` constructs **one `PreadExpertStreamer` per layer**, each with its own
     `slotCount:`, and each streamer allocates its own slot buffers from that number
     (`PreadExpertStreamer.swift:231-259`, `321-325`). Per-layer memory is already per-layer; nothing is a shared
     fixed-size table. The whole load-time change is line 168, `let effectiveSlotCount = slotCount`, which is the
     identity `docs/plan-qwen38-decode-12gb.md:182` says a heat table plugs into.
   - Two readers assume the count is uniform, and **both clear this split without modification**. The MTP pair
     schedule needs `slotCount >= 2 × topK` = 20 (`RealForwardRunner+MTPVerify.swift:67-68`); the prefill tile
     scheduler keeps its default 8 experts/tile for any `slotCount ≥ 16` at the shipped
     `maxPendingDepth = 1` (`PrefillRoutedTileScheduler.swift:36-56`, `RealForwardRunner.swift:238`). The three
     splits' minimum budgets are 53, 58 and 43 slots, so neither gate moves — which is why the floor in
     `split_slots` is inert: floors of 10, 20 and 32 produce byte-identical budgets, and only 48 changes anything
     (oracle 97.67 vs 97.69, keeps ±2%).
   - `Engine.swift:283-284` reports `slots × stride × layers` and would need `sum(budgets) × stride`; that is
     exactly equal, because `split_slots` conserves the total by construction (`test_heat_split_sim.py` pins it).
   - `ModelExpertIO.swift:147-152` documents the no-layer-argument property as deliberate; under a split it must
     return the **minimum** budget (or grow the argument), and `PreadExpertStreamer+Plan.swift:30`'s
     `experts.count <= slotCount` guard becomes per-layer — which it already effectively is, since it reads the
     streamer's own `slotCount`.
2. **§5y's top table row pooled two levers.** "oracle static per-layer split … −7.42%" was replayed with the
   saturating count *and* the split (`screen2.py`: `replay({"cap": 3}, budgets=scaled)`). The split-only arm in that
   same table is 86.08, −3.59%. Decomposed at the exact shipped total on `v520-20261015`: split alone **−4.19%**,
   cap 3 alone **−2.04%**, both **−8.04%**. The partition is a smaller share of §5y's headline than the prose implied.
3. **§5y's table has one arm listed twice.** "`lru_then_lfu`, per-layer 96 | 87.46" is the same rule as
   "LFU capped at 3 | 87.46" — the throwaway's rank key was `(min(count, 3), last)` (`screen.py:68`). It is not a
   second, independent victim order, so the ranking table's breadth overstated by one row how many distinct rules
   were tested.
4. **The cap is not a 2.6% arm.** §5y priced it from one capture and called it "inside the instrument's own 3.2%
   bias". Across three captures, cap 3 on the uniform split is −7.92% / −2.04% / −5.03%. On two of three it is well
   outside the bias. That strengthens, not weakens, the case for the one live arm §5y left open (task #177).
5. **Withdrawn:** the 92.79 placements/token figure used for `20261011`'s shipped arm in the working throwaways
   (`crossprofile3.py` carried the note). The committed instrument reads **91.77** on that capture and the tests pin
   it; `crossprofile3.py`'s number is not reproducible from the tree and is superseded.

### Deviations, all of them

1. Three captures, two prompts, one model. `v520-prompt2` is the only different-workload pair, so the shape
   transfer rests on two of three pairs being same-prompt.
2. Every transfer is scored with a cold cache and no warm-up beyond the prefill rows, the same convention as
   `replay_misses`; a live engine's cache at decode time is not cold.
3. The replay's level bias is −3.2% on `v520-20261015` and **−10.8%** on `v520-prompt2`, i.e. larger than the effect
   under test. Only the *sign* and the cross-capture agreement survive that; the magnitude does not.
4. `20261011` was captured on a pre-v5.20 build, so a shape agreement with the two v5.20 traces partly measures
   model constancy across a build change rather than session constancy.
5. No token rate appears in this section, and the one that could be inferred from §5w's 0.637 ms/miss must not be:
   that was priced in drive reads, and the ~15.5 placements/token the ring served without a read are invisible here.
6. The oracle column uses future knowledge of the trace it scores; it is published as the ceiling a static split can
   reach, not as a policy — which is the whole of its use here.
7. `v520-prompt2`'s command line is unrecovered (see the run-record). Its identity and counters are logged; nothing
   about its flags is claimed.

### Verdicts

- **The B1 heat split does not clear its own acceptance bar.** Six transfers all beat the shipped uniform split and
  four beat an oracle, and it is still a rejection: +0.66…+1.52 points against a required +7. It does not become a
  shipped default and does not get a model window on its own.
- **§5y's "architectural" verdict is withdrawn.** The change is one expression at
  `Model+LayerStreamers.swift:168` plus a table read, and the two consumers that assume a uniform count clear it at
  the split's own minimum. Anyone told in §5y that this needs a data-layout change was told wrong.
- **Do not spend the next screen on capacity shape.** At 4 608 slots the oracle ceiling of every partition is ~1
  point of hit rate, while Belady at the same slots is +8.4. The gap is temporal, not spatial. That is where the
  operator's constant-feeder idea actually points, and the counters already show a place to look: in the
  `v520-prompt2` capture the ring **issued 40.51 prefetches per token and had 20.39 adopted**, with `free=0.99` and
  `inflight=0.01` — it is issuing for twice the depth it keeps and still leaving slots idle. Nothing in this
  section measures why; that is the next question, and it is a feeder question rather than a budget question.
- **The live arm that survives unchanged** is the saturating use count (cap ~3–5) in `makeExpertCachePlan`: −2.04%
  to −7.92% placements across three captures, output-invariant by construction, and no model run yet. Task #177.
- **Nothing here is a tok/s number.** This section prices placements and hit rate, and says how many points of hit
  rate each lever can buy at this budget.

## 5aa. The constant feeder, screened in every shape it can take — it fails as a cache change, and the reason is eviction being zero-sum (2026-10-16)

§5z closed the *budget shape* lever and pointed the residual headroom somewhere else: Belady on the same 4 608 slots
is +8.4 points while every static partition is worth ~1, so the gap is in **when a slot is given up**, not in where
slots sit. The operator's premise is exactly that temporal idea — stop reading in bursts around each request and let
the drive stream weights continuously while a session is live. This section tests the premise instead of another
knob on the same lever.

The test follows from one constraint: **a feeder can only stream what it can name.** The engine ships no
per-conversation oracle, so the naming available is (a) an earlier session's statistics, or (b) this session's own
immediate future. Four arms, all inside the same 4 608-slot tier — none of them is a budget increase wearing another
name. Same licence as §5y and §5z: **ranking placement shapes at a fixed slot total; banned, quoting a token rate.**

One half of the premise was already measured on this machine, and it is worth naming before the arms. §5v sampled
`disk0` at one-second resolution across a whole decode run and got **0 idle samples** (mean 1 359.5 MB/s, min 753.0):
the drive does not stop and wait for compute. So the literal reading — bursty reads with dead wire between them — is
closed by counters, not by argument. What stayed open is the stronger reading: keep that continuously-busy wire, and
use it to read *different* weights, chosen ahead of the request. That is the thing the four arms test, and it is the
only version of the idea the route trace can price.

### What is in the tree

`benchmark/predictive_feeder_sim.py` (new) and `benchmark/test_predictive_feeder_sim.py` (new, 23 tests in 8 classes).
They reuse §5z's three committed captures, the engine's own `_PlanSlots` placement rule and `heat_split_sim`'s slot
tier, so every figure below is reproducible: `python3 benchmark/predictive_feeder_sim.py` prints these four tables
model-free and prints no token rate. Shipped-cache reference lines: hit
**80.9 / 81.4 / 78.7 %**, placements **91.77 / 89.28 / 102.10** per token, reads 242 / 236 / 270 MiB per token. Units,
again because everything here depends on them: demand is 480 requests/token, so **1 % of placements = 0.2 points of
hit rate**, and the plan's bar (`docs/plan-qwen38-decode-12gb.md:150`) of **+7 points** is **−33.6 placements/token**.

### Arm 1 — what an earlier session can name

Top-n per layer learned on one capture, scored on another. `demand cover` is the share of the other session's 480
requests per token; `miss cover` is the share of the placements the shipped rule actually makes — the only number a
feeder can act on, because a streamed expert still costs a read and the only thing that changes is whether that read
is exposed or already done.

| learned on → scored on | width | demand cover | miss cover |
|---|---|---|---|
| `20261011` → `v520-20261015` | 96 | 70.2 % | **26.5 %** |
| `20261011` → `v520-prompt2` | 96 | 42.4 % | **24.9 %** |
| `v520-20261015` → `20261011` | 96 | 72.0 % | **33.8 %** |
| `v520-20261015` → `v520-prompt2` | 96 | 36.8 % | **25.9 %** |
| `v520-prompt2` → `20261011` | 96 | 42.7 % | **28.7 %** |
| `v520-prompt2` → `v520-20261015` | 96 | 37.6 % | **28.6 %** |
| same six pairs | 192 | 61.2 – 89.9 % | 53.7 – 71.9 % |
| same six pairs | 288 | 73.0 – 95.8 % | 69.4 – 88.6 % |

At the tier's own width, **an earlier session names a quarter to a third of the misses the cache pays for**. A random
width-96 list would name 96/512 = 18.75 % of any requests by construction, so the signal is real but small. It grows
fast with width — and the width is where the idea dies: 192 and 288 named experts per layer are more than the tier
holds, so a wider list is not a cheaper feeder, it is a bigger cache under another name, and §5z's deviation 7 already
records that 128 slots/layer blows past the operator's 12 GiB cap on this Mac.

The two demand figures above 50 % are **not prediction**. They are the one *conversation* shared by two captures:
`20261011` and `v520-20261015` agree on 432 literal `(position, layer)` rows — positions 0..8 across all 48 layers —
and every pair involving `v520-prompt2` agrees on 0. Those shared rows are prefill rows, so they are not themselves
inside the scored demand; they are the marker that the two captures are the same conversation replayed on two builds.
§5z measured the same thing from the other side: 13.2 % picks in common for that pair against 3.2 % across prompts.

### The pre-registered bar, and the one disclosed rewrite

The bar was written before measuring as *"no earlier session's top-96 covers half of another session's 480 requests"*.
**It failed** (70.2 % and 72.0 %), and per the standing rule this names which side was wrong: the assertion, not the
premise. The demand denominator was the mistake — it credits a hot list for requests the live cache had already
resident-ised, which is why the *miss* column, not the demand column, is 25–34 %. The rewritten test
(`test_demand_coverage_tracks_how_much_route_the_two_captures_share`) asserts the **relation** in both directions:
a pair that shares literal rows must cover ≥ 50 %, and a pair that shares none must cover < 50 %. If a different-prompt
pair ever crosses 50 %, cross-session prediction is real and the test has to be rewritten again to say so.

The second falsifier — miss coverage < 50 % at tier width — was **not** rewritten and holds for all six pairs. Two
test-file changes are disclosed here: the rewrite above, and a wording correction in the same docstring made after
measuring, because the first version called the shared prefill rows the mechanism when they are only the marker. Both
are in a file that has never been committed; no earlier run is withdrawn, because none was published.

### Arm 2 — pinning a learned prefix costs slots the rest of the tier no longer has

P experts held for the whole session, taken **out of** the same 96-slot tier. Δ placements/token vs the shipped tier.

| learned on → scored on | P=12 | P=24 | P=48 | P=72 | P=96 |
|---|---|---|---|---|---|
| `20261011` → `v520-20261015` | −0.55 % | **−0.73 %** | +0.45 % | +1.96 % | +59.96 % |
| `v520-20261015` → `20261011` | +0.05 % | −0.05 % | **−0.71 %** | +0.10 % | +46.64 % |
| `20261011` → `v520-prompt2` | +4.05 % | +7.65 % | +16.42 % | +31.05 % | +170.89 % |
| `v520-20261015` → `v520-prompt2` | +5.11 % | +10.06 % | +19.90 % | +35.44 % | +196.92 % |
| `v520-prompt2` → `20261011` | +5.54 % | +11.22 % | +23.00 % | +41.12 % | +199.93 % |
| `v520-prompt2` → `v520-20261015` | +6.15 % | +12.70 % | +26.01 % | +49.24 % | +235.46 % |

The best honest arm in the whole table is **+0.14 points of hit rate** (−0.73 % and −0.71 %, the two same-conversation
transfers), and it is only reachable by a list learned from the same conversation. Every different-prompt transfer is
worse than shipping nothing, monotonically in P, up to 3.4× the reads at P=96 where no live slot remains. Cost, so the
ambition is legible: one pinned expert id held in all 48 layers is 0.124 GiB of pre-read, and a full 96-wide pin tier
is **11.88 GiB** — the entire expert-cache budget spent on a list learned from another session before a single row of
it is validated.

### Arm 3 — heat-biased eviction: same slots, no reservation, only a different victim

`_HeatedSlots` subclasses the shipped rule and changes exactly one thing: a protected slot's victim price is `+inf`
instead of its LFU rank, so it gives up its slot only after every unprotected one has. It claims no slot, so a list
wider than the tier is legal, and this is the arm that would have been free if nameability converted into placements.

| arm | W=96 | W=192 | W=288 | W=384 |
|---|---|---|---|---|
| `20261011` → `v520-20261015` | +1.88 % | **−3.04 %** | +0.87 % | +1.59 % |
| `v520-20261015` → `20261011` | **−6.81 %** | −6.18 % | −3.87 % | −3.20 % |
| `20261011` → `v520-prompt2` | +24.63 % | +23.33 % | +16.94 % | +16.46 % |
| `v520-20261015` → `v520-prompt2` | +22.43 % | +23.92 % | +16.71 % | +16.12 % |
| `v520-prompt2` → `20261011` | +11.52 % | +16.53 % | +9.61 % | +9.34 % |
| `v520-prompt2` → `v520-20261015` | +20.52 % | +24.22 % | +18.47 % | +17.85 % |
| **hindsight** `20261011` → itself | −15.24 % | −10.48 % | −2.82 % | −1.97 % |
| **hindsight** `v520-20261015` → itself | −16.15 % | −7.09 % | +1.15 % | +2.33 % |
| **hindsight** `v520-prompt2` → itself | −16.16 % | −11.31 % | −4.94 % | −3.89 % |

Best transfer **+1.30 points** (−6.81 %, same conversation). Best different-prompt transfer: there isn't one — the
worst is +24.63 % of placements (`20261011` → `v520-prompt2`), i.e. **−5.24 points** against its own control. And the
hindsight rows are the arm's ceiling with zero slot cost: protecting a list learned with perfect knowledge of the very
session being scored buys **+2.91 / +3.00 / +3.44 points**. Under half the bar, with the future already known.

That is the mechanism this whole screen found, and it is not a prediction-quality problem. **Eviction inside a fixed
tier is zero-sum.** Naming 62 % of a session's misses at width 192 does not reduce them, because holding those slots
means refusing to hold the ones the shipped rule would have chosen, and the shipped rule is adapting to the session
that is actually running. High nameability plus low placement reduction is the expected shape, not a surprise.

### Arm 4 — how much of THIS session's future the engine would have to own

The honest form of reading ahead: protect what this session routes in its next W tokens. It reserves no slot, so
`W=0` is exactly the shipped rule (pinned to 9 decimal places against `placements_per_token`) and every other window
reads as a delta on it. This is an **oracle** — the engine cannot see those rows yet — and it exists to bound what any
real predictor could ever be worth.

| capture | W=1 | W=2 | W=4 | W=8 | W=16 | W=32 |
|---|---|---|---|---|---|---|
| `20261011` | −4.19 % (+0.80 pt) | −6.88 % (+1.32) | −11.74 % (+2.24) | −18.92 % (+3.62) | −29.12 % (+5.57) | −40.90 % (**+7.82**) |
| `v520-20261015` | −5.08 % (+0.95 pt) | −8.26 % (+1.54) | −12.55 % (+2.33) | −18.34 % (+3.41) | −26.48 % (+4.93) | −37.91 % (**+7.05**) |
| `v520-prompt2` | −5.59 % (+1.19 pt) | −8.87 % (+1.89) | −13.75 % (+2.93) | −21.53 % (+4.58) | −30.73 % (+6.54) | −41.67 % (**+8.86**) |

Monotone and consistent across three captures: each doubling of the window buys more than the last, ≈ +0.8/+0.95/+1.19
points at one token and **+7.05 to +8.86 at 32** — which is where the plan's +7-point bar actually sits, and it
recovers most of the +8.4 that §5z attributed to Belady at the same slots. So the temporal headroom is real, and the
oracle prices it.

What the engine can actually own: nothing. MTP is off on the delivered path (§5t's config list, line 1259) and
speculation is §8-closed; and its verify pass is **width 2** — one draft token, `StreamingMTPTypes.swift:234` — so
even re-opening speculation buys W=1, worth **+0.80 to +1.19 points** (−4.2 to −5.6 % of placements). That is not close
to the bar on any reading, and §5y's licence note applies to it exactly as it applies to every other arm here: a
replay delta cannot be turned into a clock claim at all, because the ~15.5 placements/token the ring serves without a
drive read are invisible in a route trace and that share is unmeasured. The gap between what would clear the bar (32
tokens) and what is available (0, or 1 at the cost of re-opening a closed lever) is the whole result of this arm.

### One correction to §5z, named

§5z's last verdict bullet calls the ring counters "issuing for four times the depth it keeps". The two numbers it
quotes in the same sentence are 40.51 issued and 20.39 adopted per token — a ratio of **1.99**, not four. The wording
is corrected there; the counters themselves were right and are unchanged. The mistake does not soften §5z's point:
issuing 2× the depth it keeps while `inflight=0.01` is still a scheduling question that no budget arm can answer.

### Deviations, all of them

1. Three captures, two prompts, n = 1 conversation per prompt. The "same conversation" pair is one pre-v5.20 and one
   v5.20 build, so it partly measures model constancy across a build change (§5z deviation 4 applies unchanged).
2. Cold-cache replay, no warm-up beyond the prefill rows — §5z deviation 2. A live engine's cache at decode time is
   not cold, which flatters every arm that starts from an empty tier.
3. The replay's level bias is −3.2 % on `v520-20261015` and −10.8 % on `v520-prompt2` (§5z deviation 3). Only signs,
   monotonicities and cross-capture agreement survive that; no magnitude here is a machine number.
4. Arm 4 is an oracle by construction and cannot be implemented. It is published as a price, not a policy.
5. `_HeatedSlots` overrides only `_price`; `_PinnedSlots` composes the same rule at a smaller capacity. Both were
   checked to reproduce `placements_per_token` exactly at their neutral setting, so no delta in this section comes from
   a re-implemented cache rule.
6. Hot lists rank by request count with ties broken by ascending expert index, so a shipped prefix list would be
   byte-reproducible offline; that property is tested rather than assumed.
7. Heat widths above the tier (192/288/384) are legal *for that arm* because protection holds no slot, so those
   columns are judged on their own numbers. The same widths in Arm 1's table are not affordable, and are reported as
   nameability only.
8. The driver's pin-cost line first read "per layer" for a figure that is one pin spread over all 48 layers; the
   wording was corrected before commit. The first two sweep runs printed identical figures wherever their tables
   overlapped, so no number moved; deviation 10 records the third run.
9. No token rate appears in this section, and §5w's 0.637 ms per exposed read must not be used to infer one — same
   reasoning as §5z deviation 5: that was priced in drive reads, and the ~15.5 placements/token the ring served
   without one are invisible here.
10. The sweep has been run three times. The third run, after the footer rewrite and the formatter pass, reproduces the
    second **byte-for-byte across all four tables** and differs only in the two closing lines (the old one projected
    "3.6 GB/s → 274 experts/token" from an uncited rate; the new one names its two sources: §5d's 3.57 GB/s reader
    wall → 271 experts/token, and §5v's measured 1 359.5 MB/s drive → 103). Its own wall time did not reproduce: the
    same code path took 1 497 s wall / 883 s user the second time and 138.9 s / 136.3 s the third, an unexplained 6×
    on a machine that was idle in both cases, so no runtime is claimed anywhere in this section — and nothing in it
    depends on one, because the driver's output is a pure function of three committed files.

### Verdicts

- **The constant-feeder premise fails as a cache change**, in all three shapes it can take: pinning a learned prefix
  (+0.14 point best honest arm), heat-biased eviction (+1.30 across sessions, +3.44 even in hindsight), and prediction
  width (25–34 % of the misses the cache pays for at the tier's own width). None reaches +7 and none gets a model
  window.
- **Why it fails is the useful finding.** Not because misses are unnameable — at 192 names an unrelated session
  covers 54–72 % of them — but because a fixed tier makes eviction zero-sum: acting on that knowledge displaces the
  rule that was already adapting to the live session. Any future feeder screen should be judged against this
  asymmetry first, or it will re-report nameability as if it were savings.
- **The temporal headroom §5z pointed at is confirmed and is unreachable.** 32 tokens of lookahead recovers
  +7.05…+8.86 points, most of Belady's +8.4 on the same slots; one token, which is the widest thing this engine's own
  drafter could offer, is worth ≈ +1.
- **"Read constantly" survives only as scheduling, and this instrument cannot price it.** A read moved from exposed to
  overlapped is the same read, so an arm that changes no placement can still change the critical path — which is
  §5w's measurement, not a trace replay's. The ring's own counters say the burst-vs-stream question is not the one the
  engine is stuck on anyway: on `v520-prompt2` it **issued 40.51 prefetches per token and adopted 20.39**, with
  `free=0.99` and `inflight=0.01`. It is already issuing twice the depth it keeps, and half of what it reads is
  thrown away. That is a queue-depth and eviction-honesty question, and answering it needs the live paired A/B method,
  not more offline ranking.
- **The live arm that survives unchanged** is still the saturating use count (cap ~3–5) in `makeExpertCachePlan`:
  −2.04 to −7.92 % placements across three captures, output-invariant by construction, no model run yet. Task #177.
- **Nothing here is a tok/s number.** This section prices placements, says how many points of hit rate each feeder
  shape can buy at a fixed 4 608 slots, and closes the cache-change reading of the premise.

## 5ab. Four overnight premises, each closed by a measurement — and the empty scan that had to be re-run first (2026-10-16)

The overnight mandate was to try things out rather than re-run the plan's knob list. Five
premises were taken apart with model-free instruments — an SDK header inventory, a codec
entropy screen on real blob bytes, a codec *cost* screen on the same bytes, the committed
predictor study, and §5x's µs-scale host primitives. All five closed. None of them moved a
clock, and this section claims no clock.

### The scan that read nothing, and why it is first

The first version of the header inventory below was run as

```
find "$SDK/System/Library/Frameworks/Metal.framework/Headers" -name '*.h' | wc -l   # 0
grep -rli "accel" "$H"                                                              # (no output)
```

and it returned **nothing**, which is exactly what an absent API looks like. It was not
evidence: `Metal.framework/Headers` **is a symbolic link** (`stat -f '%HT'` → `Symbolic
Link`), and neither `find` nor `grep -r` follows a symlinked starting directory by default.
`ls "$H" | wc -l` over the same path reports **97** entries, and a direct `grep -c MTLIO`
on one of those files reports **18** hits, so the files were always readable — the walk was
blind. Re-run with `find -L`, which enumerates **96** headers, is the instrument every
figure below comes from.

This is the same defect shape `tools/lint.sh` hunts in `unbounded-read` and `test-hollow`:
*a scan that read nothing reporting a pass.* The earlier session's conclusion happened to
survive the re-run; the claim is nevertheless withdrawn and re-measured, because "right" and
"proven" are different properties and only the second one is reusable.

### GPU-side fetch planning is not expressible on this SDK

`docs/qwen38-decode-20tps-concept.md:96-102` leaves exactly two things that could move this
model, and the second is "GPU-side fetch planning, so the CPU never blocks on the router at
all… an architecture change". It is now answered from the API surface rather than from code,
on the toolchain this repository is required to use (macOS SDK **27.0**, Xcode 27 / Swift 6.4):

- `MTLIOCommandBuffer.h` exposes **three** load entry points — `loadBytes:` (46),
  `loadBuffer:` (56), `loadTexture:` (67) — and each one takes
  `sourceHandleOffset:(NSUInteger)`, a **host-supplied scalar baked in at encode time**.
- The device-selectable I/O family is absent: `MTLIOAccelTable`, `MTLIODevice`, `MTLHandle`,
  `MTLIOHandleDescriptor`, `MTLBufferIndirectHandleDescriptor` each appear in **0 of 96**
  headers. `MTL4AccelerationStructure.h` is ray tracing and has no I/O role.
- The complete `MTLIO*` symbol set in this SDK is the queue/buffer/status/compressor family:
  `MTLIOCommandQueue`, `MTLIOCommandBuffer`, `MTLIOFileHandle`, `MTLIOStatus*`,
  `MTLIOCompression*`, `MTLIOScratchBuffer*`, `MTLIOPriority*`. Nothing takes an offset from
  a buffer.

So the thing the architecture change needs — a kernel that decides *which file range* to
fetch after the router result lands on device — has no entry point here. Every read offset
must be known by the host before the command is encoded, which is the dependency the change
was meant to remove. What the SDK *does* offer and the engine already uses is device-gated
completion (`copyStatusToBuffer:` :81 then `signalEvent:` :139, wired together at
`MetalExpertReader.swift:114-117`), so the kernels can gate on bytes without a host
callback. That half is already shipped; the offset half is not buildable.

### The last predictor shape was already measured at 0.000

The recurrence idea — predict layer L's next-token experts from layer L's *previous* token's
selection — is the one prefetch-predictor shape not covered by §5aa's four feeder arms. It
turns out to be measured already, in the committed study, and its precision is zero on every
rank
([docs/qwen38-prefetch-predictor-study.md:44-45](qwen38-prefetch-predictor-study.md#L44-L45)):
`previous token's selection at that layer` = 0.000 / 0.000 / 0.000 / 0.000 P@1..4 and
0 / 0 / 0 adopted per token; `previous token's misses at that layer` the same; and mixing it
with the shipped probe changes nothing (0.552 / 0.450 / 0.379 / 0.328, identical to probe
alone). The mechanism is stated at lines 52-55: **the previous token's experts are always
already resident, which is why a temporal predictor stages nothing at all — the cache has
already absorbed that reuse.** §5aa arm 4 prices the same conclusion from the other side:
1 token of lookahead is worth +0.80…+1.19 points of hit rate and the plan's bar (+7) sits at
**32** tokens, which no part of this engine can own.

No test is needed and none was run: there is nothing left in the predictor shape to measure.

### Lossless expert compression — capacity measured, then cost measured, and the lever closes

This was the one genuinely new idea of the night: **gain slots without gaining RAM, without
touching numerics.** A lossless codec over the packed 4-bit blobs cannot change a single
value, so it is quality-exact by construction — the only capacity lever in this campaign that
is not in tension with the standing "never make the model dumber for tok/s" rule.

Screen on the real install (`python3 /tmp/expert_reducibility.py`, exit 0; layers
0/7/15/23/31/39/47 × experts 0/37/129/250/391/503 = 42 blobs = 116 293 632 B = 110.9 MiB,
i.e. 6 of 512 blobs per sampled layer):

| measured | value |
| --- | --- |
| codes used in the 4-bit space | **16/16** |
| order-0 nibble entropy | **3.7745 bits** = 94.36 % of 4 bits |
| order-1 conditional entropy | 3.7438 bits/nibble |
| single-valued code groups (32 B of codes each) | **0 / 3 225 600** = 0.000 % |
| zlib-9, whole blob | ratio **0.90555** → **+10.43 %** slots (96 → 106.0/layer) |
| zlib-9, code region only | 0.916 → +9.21 % (104.8) |
| zlib-9 per-blob spread | 0.901 … 0.907 |
| lzma-9, 8 blobs | **0.89493** → +11.74 % slots (0.8888 … 0.8956) |
| scale array / header | 0.945 / 0.707 |

Arithmetic pinned: `quantGroupSize 64` weights at 4 bits = 32 B of codes per group, so one
expert's 2 457 600 code bytes are 76 800 groups and the 42-blob sample is 3 225 600 groups.
The price comes from §5w's own measured slope — 48 → 96 slots (+100 %) bought **+12.2 %**
paired — so +10.43 % of capacity projects to **+1.27 % of clock**, under the 2.9 % drift
floor this campaign refuses to report as a win. lzma's +11.74 % projects to +1.43 %.

Then the cost, which is what actually kills it (`python3 /tmp/codec_cost.py`, exit 0, same
blobs, single thread, best of 5 rounds, Python's `zlib`/`lzma` i.e. the same C libraries a
Swift build would call):

| codec | decode throughput | a token's 203.55 MiB of reads costs | vs §5v drive demand 1 359.5 MB/s |
| --- | --- | --- | --- |
| zlib inflate | **458 MiB/s** (rounds 453 / 457 / 458 / 457 / 458) | **444 ms of CPU** | 0.35× |
| same, at perfect 8-core scaling (arithmetic, not measured) | 3 664 MiB/s | 55.6 ms = 26 % of the 217.34 ms token | 2.8× |
| lzma-9 inflate | 31 MiB/s | 6 572 ms | 0.02× |

One 2.64 MiB blob inflates in **5.207 ms**, which is 3× the 1.739 ms the drive spends
delivering a layer's *entire* miss burst, and inflating at the rate a burst arrives needs
**7.9×** the sustained pipe (3 622 MiB/s at §5w's measured 3.80 GB/s against 458 MiB/s of
inflate). So a lossless tier would move the bottleneck off the SSD and onto CPU cores that
already own the busiest host budget in the loop — and §5w's ledger says the token is 50.9 %
GPU-busy with 102 ms already accounted for by the drive and the dependency chain.

The one escape hatch was checked rather than assumed: the SDK's own compression surface is
**compress-only**. `MTLIOCompressor.h` declares three functions —
`MTLIOCreateCompressionContext` (:40), `MTLIOCompressionContextAppendData` (:46),
`MTLIOFlushAndDestroyCompressionContext` (:52), all macOS 13+ — with no decompression entry
point anywhere in the 96 headers. The hardware path that could beat a CPU inflate is not
nameable on this toolchain, so the codec lever is closed on both halves: +1.27 % projected
capacity for 2× a token's wall in CPU.

### What family 2's 42.8 ms is not, by arithmetic

§5w left the 42.835 ms/token of `shared_expert → moe_phase1_*` host turnaround open and
suggested "bookkeeping issued a layer ahead of the wait". §5x's µs measurements close that
suggestion off. Family 2 is 42.835 ms/token = **0.892 ms per layer**. §5x measured
encode p50 3.9 µs, commit p50 0.5 µs and a per-command-buffer host mean of 22.9 µs, so the
whole 6-CB-per-layer encoding budget is 6 × 48 × 22.9 µs = **6.59 ms/token** — 15 % of the
gap, and 0.892 ms per layer is **39×** the per-CB host mean.

Summing the named host counters on the same arm (line 1682) reaches the same place:
`cache_plan_ms=4.898` + `gather_ms=2.248` + `rdadvise_ms=7.624` + `cb1_ms=5.079` =
**19.85 ms/token**, i.e. 46 % of family 2 is named and the rest is queue serialization —
which §5w already explains: at 48 slots the same pair *falls* from 0.730 to 0.393 ms per
occurrence because the drive is behind and the host's bookkeeping disappears under already-
queued work. And §5x prices the proposed remedy: one extra commit-and-wait pair is ≈10 ms per
token, so a look-ahead command buffer spends ~10 to save ≤19.85, most of which is not waiting
on encoding. Task #179's premise is therefore answered as written, and the answer is no.

### Deviations, all of them

1. Both spikes live in `/tmp` and are **deliberately not committed**: the levers they tested
   are closed, so a new `benchmark/` driver would be a new file to audit for no usable gain.
   Their full printed output is embedded above so the numbers stay in the tree even when the
   scripts do not. They are throwaway probes, not engine code, and nothing under `sources/`
   changed in this section.
2. No model window was taken and no token rate appears in this section. Every clock-like
   figure quoted is a reference to an earlier section's measurement.
3. The entropy screen samples 6 of 512 blobs in each of 7 layers, and holds the same six
   expert ids fixed across those layers, so it is 42 blobs of 24 576 (0.17 %) and cannot
   separate layer variation from expert variation. The per-blob ratio spread (0.901 … 0.907)
   is that sample's variance, not a claim about the whole install.
4. The first entropy run was **broken and is withdrawn**: `entropy()` summed a `Counter`'s
   keys instead of its values and printed −29 141 861.85 bits (−728 546 546 % of 4 bits). The
   zlib/lzma ratios in that run were unaffected, the bug was fixed, and only the re-run's
   entropy lines are quoted.
5. The same run initially used a 128 B group where the format's is 32 B; the single-valued
   count and the group total changed with the fix. 76 800 groups/expert matches the manifest's
   scale count exactly, which is what confirms the corrected size.
6. Inflate throughput is single-threaded C-library throughput on this Mac. The 8-core row is
   labelled arithmetic and is **not** a measurement — no parallel inflate was run.
7. The capacity→clock projection uses §5w's 48→96 slope linearly across +10 %. The real
   slot/response curve is not linear and §5o's cliff sits elsewhere; the projection is quoted
   as an upper bound on a small step, not as a measurement of that step.
8. The SDK conclusion is about **this** SDK (macOS 27.0 from Xcode 27). A newer SDK could add
   the device-handle family; that is a toolchain question, and this repository's rule is that
   only the pinned toolchain's guarantees count as supported.
9. `nm -gU` over the framework binary returned nothing (the SDK ships a stub; the real image
   is in the dyld shared cache), so it is not part of the evidence. The header walk is.

### Verdicts

- **Four of the five overnight premises are closed by measurement, not by argument**: the
  recurrence predictor (already 0.000 on every rank), GPU-side fetch planning (no API for it
  on this SDK), lossless expert compression (+1.27 % projected capacity against 2× a token's
  wall in CPU), and look-ahead host bookkeeping (15 % of the gap it is meant to remove, and
  priced above its own saving).
- **The fifth is the instrument**: a blind `find`/`grep -r` walk over a symlinked SDK
  directory produced a confident, empty, wrong-shaped result. It is recorded here as the
  section's headline finding, not a footnote, because the conclusion it happened to support
  was already the right one — which is precisely when a broken gauge is dangerous.
- **No capacity route is left that does not need more RAM.** Slots can be bought with bytes
  (lossless: refused, sub-noise and CPU-expensive), with placement (5aa: zero-sum), with
  budget shape (5z: capped), or with hardware (not this Mac, and the operator's 12 GiB rule
  binds here). The concept document's remaining two levers are now both measured: more RAM,
  and nothing else.
- **The live work that survives this section unchanged** is #177 — the saturating use-count
  arm in `makeExpertCachePlan`, −2.04…−7.92 % placements offline, output-invariant by
  construction, still needing the paired live A/B — and #163, the wiki decode row.

## 5ac. Issue-count coalescing, priced by structure instead of by more repeats (2026-10-16)

§5m left one seam parked rather than closed: *"Issue-count coalescing prices at a
theoretical ceiling of the 10–12 ms/token it could recover while keeping the hiding —
≈2–3%, single-% territory … Parked as unproven; an A/B would need many more repeats
per arm than four."* That ceiling is the cost of the whole `F_RDADVISE` line item, and
an A/B was the only instrument proposed for finding how much of it coalescing can
actually reach. It does not need one. The merge condition is structural, so the
reachable fraction is computable from the committed route captures.

The mechanism, read rather than assumed: `sources/TinyTitan/Infrastructure/Streaming/RDAdvice.swift:33-38`
issues one `fcntl(fd, F_RDADVISE, &advice)` per call with a single `radvisory{ra_offset,
ra_count}`, so two expert ranges share a syscall only if their byte ranges touch —
i.e. only if their expert ids are consecutive inside the layer file. The live counters
confirm one call per miss read: `rdadvise_calls=53 712` over a 512-token decode is
104.9 calls/token against §5y's 89–102 placements/token at that prompt shape.

### What the bursts actually look like

`python3 /tmp/rdadvise_merge_probe.py` (throwaway, not committed, per §5ab deviation 1)
replays the shipped plan rule over the three committed captures with §5y's warm-up and
counts, for every `(token, layer)` miss burst, the maximal runs of ascending expert ids
at three hole tolerances. Gap 0 merges only strictly consecutive ids and therefore costs
no extra bytes; gap 1 and 2 buy fewer calls by reading the experts in between.

| capture | placements/token | bursts with 2+ misses | gap 0 calls/token | gap 1 net | gap 2 net |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20261011` | 91.765 | 8 809/13 198 = 66.7 % | 91.21 → **+0.059 ms** | −0.291 ms | −1.036 ms |
| `v520-20261015` | 89.284 | 11 842/17 951 = 66.0 % | 88.77 → **+0.054 ms** | −0.189 ms | −0.833 ms |
| `v520-prompt2` | 102.104 | 9 840/14 071 = 69.9 % | 101.37 → **+0.077 ms** | −0.279 ms | −1.119 ms |

The nets are `ms/token` after the wasted bytes are priced against the drive: a merge
that drags in one extra expert costs 2.64 MiB = **0.66 ms of device time** at §5o's
measured 3.95 GB/s, while the call it saves is worth **0.105 ms** (`rdadvise_ms`
10.958/token ÷ 104.9 calls/token). So a byte-costing merge loses by ~6:1 before it
starts, which is why gap 1 and gap 2 are negative and only gap 0 is a candidate.

**Two thirds of bursts do hold two or more misses — and 0.6 % of them are side by
side.** That is the whole lever: 0.054–0.077 ms/token = **0.025–0.035 % of a 219 ms
token**, against the 2–3 % §5m parked it at.

### The only way to buy contiguity is to move the weights, and that is capped too

Contiguity is a property of the on-disk id order, and the id order is the repack's
choice. Relabelling is numerics-free by construction — the same bytes at a different
offset, with the routing path resolving id → offset through a table — so a repack could
place co-routed experts side by side and manufacture the adjacency the router does not
supply. `python3 /tmp/expert_layout_screen.py` (throwaway) builds a per-layer
co-occurrence graph from the routed rows, takes a greedy maximum-weight path cover
(heaviest edge first, degree ≤ 2, no cycles — deterministic and cheap enough to run
inside a repack), writes each path to consecutive offsets, and replays the shipped rule
under that permutation.

| layout learned on | evaluated on | placements/token | calls/token | saved |
| --- | --- | ---: | ---: | ---: |
| shipped order | any capture | unchanged | 91.21 / 88.77 / 101.37 | 0.025–0.035 % of clock |
| `20261011` | `20261011` (self) | **91.765** | 88.35 | 0.164 % |
| `20261011` | `v520-20261015` | **89.284** | 88.17 | 0.053 % |
| `20261011` | `v520-prompt2` | **102.104** | 100.90 | 0.058 % |
| `v520-20261015` | `v520-20261015` (self) | **89.284** | 86.21 | 0.147 % |
| `v520-prompt2` | `v520-prompt2` (self) | **102.104** | 98.26 | 0.184 % |
| best cross-session arm | — | — | — | **0.063 %** |

The self-check is the `placements/token` column: it is identical, to the last digit,
under every permutation. That is the proof that a relabel cannot change what the cache
does — placement depends on the arrival sequence, not on the labels — so the only thing
moving is the syscall count, exactly as intended.

A learned layout does create adjacency, and it does transfer, but only at ⅓ of its own
value: 0.147–0.184 % against the session it was trained on, 0.044–0.063 % against
another. The reason is the same one §5h found from the other side — the 512 experts are
mutually orthogonal, so their co-routing is diffuse: a layer file has 511 adjacencies
available against 130 816 possible pairs, and no ordering can make many of them hot.

### Deviations

1. Both probes are `/tmp` throwaways and are deliberately not committed: they read
   committed captures and change no shipped code, so there is nothing to register.
   Their full text is reproducible from the tables above.
2. **My first run of the merge probe was wrong and is recorded as such.** It skipped
   the prefill rows instead of planning them, so decode started on a cold cache and
   reported 96.89 / 90.77 / 105.45 placements/token — 5.6 / 1.5 / 3.3 above §5y's
   calibrated figures. The self-check against `placements_per_token` caught it before
   any merge number was read; after the warm-up fix all three match to 1e-6. The
   discard's ratios were not wildly off (0.62 % vs 0.61 % adjacency), which is exactly
   why the absolute check, not the plausibility of the ratio, is what earns the result.
3. 0.105 ms per call is the engine's own aggregate: it attributes all of
   `rdadvise_ms` to the `fcntl`, including whatever the read thread pays around it. If
   the true syscall is cheaper, the ceiling is lower, not higher.
4. A 219 ms token is the shipped-default wall from §5r, used only as a denominator;
   nothing here is a clock claim.
5. The device side argues against the lever as well. §5d measured `dd` over a whole
   layer file at 3.15 / 2.61 GB/s — *slower* than the engine's strided 3.57 GB/s — so
   merging short scattered ranges into longer sequential ones moves toward the access
   shape this drive handles worse. A pure syscall win of ≤0.18 % would have to survive
   that to reach the wall clock at all.

### Verdict

**§5m's parked seam closes at two orders of magnitude below its estimate.** Merging
what is already contiguous reaches 0.025–0.035 % of clock; merging what a learned
relabel makes contiguous reaches 0.147–0.184 % on its own session and 0.063 % on
another's, and §5d's sequential contrast says the device may take some of that back.
None of it is within reach of the 2–3 % that made the A/B look worth many repeats, so
no repeats are owed: the lever is closed by structure, and the parked row can be
retired without a model run.

**What survives unchanged** is #177 — the saturating use-count arm, whose live paired
A/B is the running subject of §5ad — and #163, the wiki decode row.
## 5ad. #177 live: the capped use-count is a measured loss — and the replay that recommended it was a first-request proxy (2026-10-16)

§5aa's one surviving candidate was the saturating use-count cap in
`makeExpertCachePlan`: clamp the *ranking key* `min(useCount, C)` while the engine's
per-expert count keeps accumulating, so a defended slot is priced by recent demand
instead of by everything it has done since load. It was −2.04…−7.92 % placements
offline and output-invariant by construction, and §5ab closed the night with it as the
one arm still owed a live paired A/B. The arm has now run. **It loses, and the reason
it loses is a hole in the offline screen itself.**

Provenance: `models/qwen3.8-flash-next_125B_A6B_4Bit`, `.build/release/TinyTitanServer`
sha256 `fdaa34830d33a1e52d436376e30f821947fd0455e342aba4c26c1a944b0f948d`, built at
`2ef9d00`; `git diff --name-only 2ef9d00..HEAD` names `ci.yml` and two files under
`docs/` and **nothing under `sources/`**, so that binary is the engine at `1307a75`.
Working tree clean at measurement. 24 GB Apple M3, macOS 27.0.1, Swift 6.4,
target `arm64-apple-macosx27.0.0`, AC power at 100 % (recorded at launch),
`memory_pressure -Q` 85 % free. Command:
`python3 benchmark/tinytitan_usecap_ab.py` → exit 0. 3 rounds × 3 arms × 2 requests of
512 tokens at `--ram 12G`, one fresh server per arm per round (9 servers, never two
alive), arms interleaved within each round, `default` carried first as the control.
The cap is `TINYTITAN_EXPERT_USE_CAP`
(`sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Residency.swift:110-135`),
unset for the control arm — the driver *clears* it rather than leaving it, so an
operator's exported value cannot silently rename an arm.

### The wall clock, both pairings

| round | default r1 | default r2 | cap3 r1 | cap3 r2 | cap5 r1 | cap5 r2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.557 | 4.763 | 4.614 | 4.486 | 4.509 | 4.506 |
| 2 | 4.641 | 4.744 | 4.623 | 4.508 | 4.631 | 4.630 |
| 3 | 4.664 | 4.879 | 4.544 | 4.533 | 4.558 | 4.572 |

The driver pairs request *2* with request *2*, because that is the request a cache
policy can actually distinguish — and that pairing is uniformly negative:

| arm | r1 | r2 | r3 | mean |
| --- | ---: | ---: | ---: | ---: |
| cap3 (warm pairing) | −0.277 (−5.8 %) | −0.236 (−5.0 %) | −0.346 (−7.1 %) | **−0.286 (−6.0 %)** |
| cap5 (warm pairing) | −0.257 (−5.4 %) | −0.114 (−2.4 %) | −0.307 (−6.3 %) | **−0.226 (−4.7 %)** |
| cap3 (cold pairing, request 1) | +0.057 | −0.018 | −0.120 | −0.027 |
| cap5 (cold pairing, request 1) | −0.048 | −0.010 | −0.106 | −0.055 |

Six of six warm pairs negative; the cold pairs are inside ±0.12 tok/s and average
−0.6 % and −1.2 %. So the arm is not "slightly slower everywhere" — it is **neutral on
a cold cache and 5–7 % slower on a warm one**. That shape is the finding.

### The counters carry the mechanism, not the clock

| arm | hit rate range | mean hit | mean misses | mean read MiB |
| --- | --- | ---: | ---: | ---: |
| default | 0.7676 … 0.7872 | **0.7775** | 55 418 | 146 336.8 |
| cap3 | 0.7626 … 0.7643 | 0.7633 | 58 951 (+6.4 %) | 155 667.9 (+6.4 %) |
| cap5 | 0.7650 … 0.7667 | 0.7659 | 58 315 (+5.2 %) | 153 987.6 (+5.2 %) |

The default arm's second request gains **+1.8 points** of hit rate over its first
(0.7689 → 0.7864 at round 2): that is the cache remembering the previous request. The
cap arms gain nothing (cap3 0.7643 → 0.7626, i.e. −0.2 points). A clamp at 3 or 5 makes
every well-used expert indistinguishable from one used exactly that many times, so the
rule that defended the resident set across requests degenerates into recency — and
recency is exactly what §5y's `_Slots` textbook rule is, which lands at 151.20
misses/token where the shipped plan lands at 89.28. **The cap does not weaken the LFU
rank at the margin; it removes the memory that made the rank worth having.**
Response invariance held on every round: `b5c6c72d385f2f20` across default, cap3, cap5.

### Why the offline screen said the opposite, and what it now has to do

§5aa's replay ran each arm on **fresh caches over one session** — the analogue of
request 1, which is precisely where this arm is neutral-to-better. The second request
is not reproducible in that shape. Replaying the same route rows **twice over one cache
set** models it (greedy decoding makes the second request's routes identical, which the
hash equality above confirms), and the prediction flips:

| capture | pass | shipped | cap3 | cap5 |
| --- | --- | ---: | ---: | ---: |
| `20261011` | 1 | 91.765 | 84.496 (**−7.9 %**) | 84.734 (−7.7 %) |
| | 2 | 81.486 | 86.977 (**+6.7 %**) | 84.044 (+3.1 %) |
| `v520-20261015` | 1 | 89.284 | 87.458 (−2.0 %) | 86.951 (−2.6 %) |
| | 2 | 77.258 | 87.487 (**+13.2 %**) | 84.687 (+9.6 %) |
| `v520-prompt2` | 1 | 102.104 | 96.966 (−5.0 %) | 98.157 (−3.9 %) |
| | 2 | 90.752 | 95.397 (**+5.1 %**) | 93.493 (+3.0 %) |

Pass 1 reproduces the shipped baselines to the digit (91.765 / 89.284 / 102.104), so
this harness is §5y's, and only the second pass is new. The shipped rule's second pass
improves 11.1–13.5 % over its first — that is the warmth the live run measured as
+1.8 points of hit rate — and the cap arms' second passes do not improve at all
(+2.94 %, +0.03 %, −1.62 % for cap3). The offline second-pass penalty of
**+5.1…+13.2 %** placements brackets the live **+5.2…+6.4 %** miss penalty, which is
the agreement that matters: the counters predict each other across an offline replay and
a real server, and the wall clock is only the third witness.

### Deviations

1. **Swap was already 6 523.62 M of 8 192 M committed before the run started**, on
   mains, with the machine otherwise idle. Not removable without purging caches, which
   this repository forbids, so it is disclosed rather than fixed. It raises both arms
   equally and the counters move with the wall clock, not against it.
2. **Two single-threaded python screens totalling ~50 s of CPU ran inside the
   measurement window** — §5ac's merge probe and layout screen, while rounds 2–3 were
   in flight. Arms are interleaved *within* a round, so common-mode drift cancels in
   the paired delta, but round 3's losses are the largest of the three and I cannot
   exclude an asymmetric effect on it. That is why the verdict rests on the six-of-six
   sign, the counters and §5ad's own offline reproduction, not on the magnitude of
   −6.0 %. Re-running for a cleaner magnitude is not needed to decide the arm.
3. The driver's headline pairs the warm request. Quoted above alongside the cold
   pairing, because the difference between the two *is* the mechanism.
4. A 512-token request at ~4.6 tok/s is ~111 s per arm-request; 18 requests plus 9
   server starts is ~40 min of model time. Nothing else ran against the GPU.
5. `prefetch_issued_per_token` and `io_hidden_pct` are flat across arms (42.3–43.0 and
   99.7–99.9 %), so the cap does not disturb the prefetch ring — the loss is purely
   placement.

### Verdict

**The cap is refused.** `TINYTITAN_EXPERT_USE_CAP` stays unset by default — the shipped
rank remains the uncapped per-expert count that outlives its slot — and the flag and its
35 model-free tests stay in the tree as a measured-refused control, the same way
`--rdadvise off` stays after §5m.

**#177 closes as measured, and it leaves a rule behind**: for any candidate that changes
what the cache *remembers*, the admissible offline screen is the two-pass replay, not
the single-pass one. §5aa's numbers for this arm were not wrong — they described request
1, which the live run confirms is neutral. The campaign's other offline-screened policy
arms share the blind spot and should be re-read with that limit stated, which is what
this section exists to state.

**The rule is now code, not prose.** `two_pass_placements` in `benchmark/heat_split_sim.py`
replays the same rows twice over one shared cache set and returns both counts; its three
tests live in the already-CI-registered `benchmark/test_heat_split_sim.py` as
`TwoPassReplayTests`. The first pins pass 1 against `placements_per_token` over the same
events, so the two numbers cannot drift into different quantities and every §5y/5z/5ad
single-pass table stays comparable to a two-pass one; the second hand-traces the
remembering itself (a 3-expert working set over 3 slots costs 1.0 placements/token cold
and 0.0 warm); the third asserts the flip on the committed captures — `use_cap=3` beats
the shipped rank on pass 1 and loses on pass 2, on all three. One constraint inherited
from `_PlanSlots` and worth knowing before writing an arm: a row may not name more experts
than its layer has slots, which the engine cannot produce anyway (top-k 10 against 96) but
a hand-built fixture easily can — the failure is an `IndexError` inside the plan, not a
refusal.

## 5ae. The two-pass screen run over the campaign's backlog: the feeder family splits in two (2026-10-16)

§5ad left a rule and a backlog. This is the backlog. Three arms that were priced on fresh caches are
re-priced on both passes, at **cap none — the engine rule as shipped** — and every figure is placements
of token converted to hit-rate points on the 480-request denominator (1 placement = 0.2083 points, bar
= +7 points). Model-free: the three committed captures, python, no GPU, no clock, same licence as
§5y/§5z/§5aa.

Each arm was gated before it was read, by a self-check that **pass 1 reproduces the published
single-pass function to 1e-9**:

| arm | pass-1 gate | arms checked |
|---|---|---|
| §5z per-layer budget split | `heat_split_sim.placements_per_token` + §5z's own printed table | 9/9 |
| §5aa arm 3, heat-biased eviction | `predictive_feeder_sim.heated_placements` | 18/18 |
| §5aa arm 4, oracle look-ahead | `predictive_feeder_sim.lookahead_placements` | 21/21 |

That gate is not decoration — it is the reason this section's numbers are trustworthy. The first arm-3
screen counted with `placements += len(cache.plan(experts)) if decode else 0`, and Python evaluates only
the chosen branch of a conditional expression, so `plan()` never ran on a prefill row: the replay had
silently dropped §5y's warm-up and scored a cold cache. Its shipped baseline came out at 90.775
placements/token against the capture's published 89.284 — and 90.775 is *exactly* the discarded cold
figure §5ac disclosed and withdrew, which is what pointed the finger at my copy rather than at the
published value. Calling `plan` unconditionally and counting conditionally fixes it; all 18 arms then
reproduce to 1e-9. No number below comes from the broken copy.

### 1. The §5z heat split is warmth-neutral — and it never rested on the refused cap

| scored on | arm | pass 1 pts | pass 2 pts |
|---|---|---:|---:|
| `20261011` | uniform (shipped), 91.765 → 81.486 | — | — |
| | oracle: own profile | **+0.71** | **+0.87** |
| | transfer from `v520-20261015` | **+0.66** | **+0.75** |
| | transfer from `v520-prompt2` | **+0.73** | **+0.84** |
| `v520-20261015` | uniform (shipped), 89.284 → 77.258 | — | — |
| | oracle: own profile | **+0.78** | **+0.74** |
| | transfer from `20261011` | **+0.82** | **+0.80** |
| | transfer from `v520-prompt2` | **+0.91** | **+0.82** |
| `v520-prompt2` | uniform (shipped), 102.104 → 90.752 | — | — |
| | oracle: own profile | **+0.92** | **+0.99** |
| | transfer from `20261011` | **+0.78** | **+0.85** |
| | transfer from `v520-20261015` | **+0.74** | **+0.81** |

Pass 1 reproduces §5z's transfer table to the printed digit on all nine rows, which answers the second
question §5ad raised: **no, §5z never depended on `use_cap=3`.** Its own words already said "cap none,
engine rule as shipped" and the figures agree. The cap is only what `CrossCaptureTests` scores — a test
problem, not a documentation problem, filed as #185.

The first question: nine of nine split arms are positive on **both** passes. Six arms move up on the warm
pass (by +0.07…+0.16) and three move down (by −0.02…−0.09), so no arm moves by more than 0.16 points. That
is mechanism, not luck. `split_slots` changes **how many slots a layer holds** — a property of the cache's
size — and leaves the eviction rule's memory untouched, so it prices the same cold and warm. §5z's verdict
therefore stands unchanged by warmth: the six transfers land at +0.66…+0.91 on pass 1 and +0.75…+0.85 on
pass 2, against oracle ceilings of +0.71…+0.92 and +0.74…+0.99, which is every static partition of 4 608
slots sitting seven to eleven times under the +7-point bar.

### 2. Arm 4 keeps four fifths of its gain on a warm cache, and the bar it clears is one capture wide

| capture | W | pass 1 pts | pass 2 pts | retained | placements the 2nd request saves |
|---|---|---:|---:|---:|---:|
| `20261011` | 0 = shipped | — | — | — | 10.28 |
| | 1 | +0.80 | +0.66 | 82 % | 9.60 |
| | 8 | +3.62 | +2.90 | 80 % | 6.84 |
| | 16 | +5.57 | +4.47 | 80 % | 5.00 |
| | 32 | **+7.82** | **+6.40** | 82 % | 3.45 |
| `v520-20261015` | 0 = shipped | — | — | — | 12.03 |
| | 1 | +0.95 | +0.77 | 82 % | 11.21 |
| | 8 | +3.41 | +2.71 | 79 % | 8.64 |
| | 16 | +4.92 | +4.00 | 81 % | 7.56 |
| | 32 | **+7.05** | **+5.75** | 82 % | 5.78 |
| `v520-prompt2` | 0 = shipped | — | — | — | 11.35 |
| | 1 | +1.19 | +0.94 | 79 % | 10.18 |
| | 8 | +4.58 | +3.69 | 81 % | 7.08 |
| | 16 | +6.54 | +5.24 | 80 % | 5.11 |
| | 32 | **+8.86** | **+7.19** | 81 % | 3.30 |

Three results, and the middle one is the interesting one.

**The gain survives warmth.** Across 18 window/capture points the arm retains 79–82 % of its cold value
— a near-constant four fifths, independent of window size. So §5aa's temporal headroom was not an artifact
of cold caches, unlike everything else in the feeder family.

**The crossing is.** At W=32 the +7-point bar is cleared on pass 1 by 3 of 3 captures (+7.82/+7.05/+8.86)
and on pass 2 by **1 of 3** (+6.40/+5.75/+7.19). The bar is a threshold, so this does not soften the
conclusion into "still clears, roughly" — it moves two of three captures to the wrong side of it. And the
form the engine could actually own (§5aa: MTP is off, and its verify pass is width 2, so W=1 at best) is
**+0.66…+0.94 points warm** against +0.80…+1.19 cold.

**Protection and warmth are substitutes.** The shipped rule's second request saves 10.28/12.03/11.35
placements/token — offline warmth of 2.14/2.51/2.37 points against the +1.8 points §5ad measured live on the
same default arm. An engine that already protected this session's next 32 tokens saves only 3.45/5.78/3.30
on the same second request: the look-ahead had already consumed **66 %, 52 % and 71 %** of what the repeated
request would have delivered.
Both mechanisms buy the same thing — the right expert resident before it is named — so their gains are not
additive, and any future plan that prices "prediction" and "multi-turn warmth" together must subtract one
from the other.

### 3. Arm 3 is a cold-start discovery effect: every positive transfer flips sign

| learned on → scored on | W | pass 1 pts | pass 2 pts |
|---|---|---:|---:|
| **hindsight** `20261011` → itself | 96 | +2.91 | +1.12 |
| | 192 | +2.00 | −0.08 |
| **hindsight** `v520-20261015` → itself | 96 | +3.00 | +0.84 |
| | 192 | +1.32 | −0.92 |
| **hindsight** `v520-prompt2` → itself | 96 | +3.44 | +1.24 |
| | 192 | +2.41 | +0.07 |
| `v520-20261015` → `20261011` | 96 | **+1.30** | **−1.16** |
| | 192 | **+1.18** | **−0.96** |
| `20261011` → `v520-20261015` | 96 | −0.35 | −2.42 |
| | 192 | **+0.57** | **−1.67** |
| `20261011` → `v520-prompt2` | 96 | −5.24 | −8.49 |
| | 192 | −4.96 | −7.53 |
| `v520-20261015` → `v520-prompt2` | 96 | −4.77 | −7.73 |
| | 192 | −5.09 | −7.76 |
| `v520-prompt2` → `20261011` | 96 | −2.20 | −6.53 |
| | 192 | −3.16 | −5.95 |
| `v520-prompt2` → `v520-20261015` | 96 | −3.82 | −8.94 |
| | 192 | −4.51 | −7.75 |

All three transfers that were worth anything on a cold cache (**+1.30, +1.18, +0.57**) are worth *less than
nothing* on a warm one (−1.16, −0.96, −1.67), and the nine that were already negative get worse — up to
−8.94 points. The hindsight rows, the arm's ceiling with the session's own future already known and at zero
slot cost, keep 38 %/28 %/36 % at W=96 and **nothing at all at W=192** (+0.07…−0.92).

So §5aa's mechanism claim ("eviction inside a fixed tier is zero-sum") is not merely confirmed, it is
narrowed: what little arm 3 did convert was the cache **discovering its working set**, and a second pass
discovers it for free. A list learned from another session is redundant with the memory the shipped rule
already has — which is the same verdict §5ad reached for the cap from the live side, arriving here from the
offline side, and the reason both are now one finding rather than two.

### The rule the campaign gets out of this

The two passes separate policy arms by *what they change*, and that is worth stating before the next
candidate is screened:

- Arms that change **how fast the cache reaches its steady state** — cross-session victim protection
  (arm 3), a saturating count that flattens the rank (§5ad) — are priced almost entirely by pass 1 and
  collapse or flip on pass 2. Both arms of this class that the campaign measured are in that shape.
- Arms that change **what the steady state is** — per-layer budget size (§5z, ±0.16 points), own-future
  protection (arm 4, 79–82 % retained) — price similarly on both passes.

A single-pass offline screen therefore over-reports exactly the first class, and the first class is the one
that looks cheapest to build. Nothing of that class shipped, which is the only good news in the section.

### Deviations

1. **My first arm-3 replay dropped the warm-up** (the conditional-expression shape described above) and its
   figures are withdrawn, not adjusted. The corrected screen's 18/18 self-check is the evidence for §3.
2. **The arm-4 throwaway still contains the bug in its own arm-3 half**, so `feeder_two_pass_screen.py`
   exits 1 and the 12 arm-3 rows it prints are wrong — the corrected `/tmp/arm3_two_pass.py` rows are the
   ones used here. Its arm-4 loop calls `plan` first and counts second, which is why 21/21 of the arm-4
   checks pass. Disclosed because both scripts are in `/tmp` and may be re-run by someone who is not me.
3. **Arm 2 (pinned prefix) was not re-screened.** Its published best honest arm is +0.14 points, fifty times
   under the bar and reachable only from the same conversation; a sign flip cannot move that decision. This
   is the one backlog row closed by reasoning instead of measurement, and it is named as such.
4. **Pass 2 models a repeated identical request**, which is the shape §5ad confirmed live (greedy routes,
   hash `b5c6c72d385f2f20`). A multi-turn conversation whose next turn asks something different routes
   differently — §5z measured that agreement as 3.2–13.2 % of picks — so it lands *between* the two passes.
   Read both columns as a bracket on a session's value, not the warm one as the answer. For arm 3 that
   bracket is the whole story: its cold gain is real only to the extent turns repeat routes.
5. **Two §5aa intermediate points differ by 0.01** (+4.93 vs +4.92 at `v520-20261015` W=16; +2.93 vs +2.92
   at `v520-prompt2` W=4). Placements reproduce to 1e-9, so the difference is the doc's hand-converted
   rounded percentages: `predictive_feeder_sim.main()` prints points only at W=32. Not a measurement
   disagreement.
6. **Only `two_pass_placements` is shipped** (`85f810a`, and `test_heat_split_sim.py` is at 20 tests — §5z
   recorded 17, the three new ones are `TwoPassReplayTests`). §1 runs through it; §2 and §3 need their own
   two-pass loops, because a protection set is not a budget. If the campaign keeps pricing warm policies
   that loop shape belongs in `predictive_feeder_sim.py` — a decision, not a task for this section.
7. **No clock anywhere.** These are placement counts on three captures whose replay level error (§5z: −3.2 %
   to −10.8 % against the engine's own bracket) is larger than every effect above. The ban on quoting a
   token rate from them is arithmetic, not caution.
8. Provenance: tree `85f810a` clean, python 3.14.8, three committed route captures, no model loaded, no GPU
   work, mains power but nothing timed.

### Verdict

**The backlog §5ad named is re-priced, and the feeder family splits in two.** The budget-shape lever is
warmth-neutral and still seven to eleven times under the bar; arm 4's own-future protection survives warmth
at four fifths of its value but clears the +7-point bar on one capture of three rather than three of three;
arm 3's cross-session protection does not survive it at all. No shipped default changes: the 96-slot tier,
the uncapped rank and the flat per-layer budget are all steady-state levers, and this section is the check
that nothing in the cold-start class was quietly priced into one of them.

**What it does change** is the campaign's read on the temporal headroom. §5aa said the headroom is real but
owned by nobody; §5ae says the headroom is real *on a warm cache too*, which is the stronger claim — a
predictor that only paid off on cold starts would not be worth building, and one that keeps 80 % of its
points would. The blocker is unchanged and is not a measurement: the window that clears the bar is 32 tokens
and the window the engine can own is 1. **The reusable rule is the one to take forward**: price every future
policy arm on both passes and treat a pass-1-only gain as a cold-start effect until proven otherwise —
which is now the shape of two refusals (§5ad's cap, §5aa's arm 3) and would have saved the campaign a live
A/B if it had been stated first.

## 5af. Seven probes of the constant-feeder premise: the drive's wall is 3.3–3.4 GB/s, and the live session pays 17–53 % of decode for filling its idle clock (2026-10-16)

§5aa closed the feeder as a *cache* change (eviction inside a fixed tier is zero-sum) and §5v closed the
duty-cycle reading (34.7 % busy, and the idle clock is not spare bandwidth). Neither measured the thing the
operator's own words ask about: *"the sdd can read constantly while a user has an active session and can keep
on reading weights even if compute is doing stuff."* Whether that is worth doing depends on what the device
delivers when it is asked to read constantly, in the shapes a feeder would produce. Six model-free probes on
on the 48 `packed_experts/layer_NN.bin` files answer the device half of it. The seventh probe is not model-free: it runs
the same feeder beside a live decode session, because the offline arms can say what the device does under a constant
feed but not what a token does.

### The instrument

`/tmp/slice_sweep2.c` mirrors the shipped reader's terms and nothing else: `pread` on descriptors opened
`F_NOCACHE`, 2 MiB-aligned scratch, expert-stride-aligned offsets, a full-length read loop, per-read latency,
1–32 workers, and an arbitrary number of cooperating processes released by a shared gate file so their
windows overlap. Its aggregate for multi-process arms is `Σ bytes / (max end − min start)` — the union window,
not an average of averages. Worker seeds carry `getpid()`, because two processes chasing the same LBA
sequence would measure locality instead of the device.

### #186 — granularity × concurrency: the knee is in-flight bytes, and the shipped reader sits on it

Median of 5 interleaved rounds; control noise floor 3.258–3.385 GB/s (3.9 %). Rate at a given amount of
in-flight bytes:

| in-flight MiB | 0.17 | 0.66 | 2.64 | 5.28 | 10.56 | 21.12 | 42.25 | 84.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slice 2.641 MiB | — | — | 2.459 | 3.239 | **3.360** | 3.380 | 3.344 | 3.329 |
| slice 1.320 MiB | — | — | 2.132 | 3.073 | 3.318 | 3.352 | 3.359 | 3.274 |
| slice 0.660 MiB | — | — | 1.716 | 2.632 | 3.202 | 3.309 | 3.316 | 3.309 |
| slice 0.330 MiB | — | 0.999 | 1.676 | 2.347 | 2.856 | 3.052 | 3.039 | — |
| slice 0.165 MiB | 0.722 | 1.276 | 1.968 | 2.541 | 2.803 | 2.747 | — | — |

Three readings, all of them against the feeder idea:

1. **Throughput is set by in-flight bytes, not by how you cut them.** At 10.56 MiB in flight the three
   coarse slices deliver 3.202–3.360; at 21.12 they deliver 3.309–3.380. Granularity is free *down to
   0.66 MiB* and then costs: at the same 10.56 MiB, 0.330 MiB reads give 2.856 and 0.165 MiB give 2.541.
2. **The shipped shape is the knee.** Four workers × 2.64 MiB = 10.56 MiB is where the curve flattens; 8, 16
   and 32 workers move it by +0.020, −0.016 and −0.031 GB/s against that control. A reader with more
   concurrency in it would not read any faster.
3. **Extra workers buy latency, not bandwidth.** p50 per full read at slice 2.641 MiB: 1100 µs at w=1, 3260
   at w=4, 6534 at w=8, 13 182 at w=16, 26 793 at w=32 — linear in queue depth while the rate is flat.

The finer slices are not hypothetical: they are what B3's split blob reads issue. At w=4 a 1.32 MiB half-read
costs −0.019 (noise), and the campaign's own sub-0.66 MiB shapes cost −0.245 and −0.545 at their best.

### #187 — burst length: the wall is flat from 64 MiB to 6.4 GiB, so "constantly" is not slower, and not faster either

Control 203 MiB at 2.641 MiB × 4: 3.218–3.391 across 9 round-pairs (5.2 %).

| window MiB | 64 | 128 | 203 | 406 | 812 | 1625 | 3250 | 6500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.641 MiB × 4 | 3.265 | 3.317 | 3.356 | 3.374 | 3.373 | 3.374 | 3.388 | 3.376 |
| paired Δ vs control | −0.074 | −0.017 | n/a | +0.008 | +0.013 | +0.008 | +0.022 | +0.004 |
| 0.165 MiB × 16 | 3.044 | 3.213 | 3.242 | 3.260 | 3.291 | 3.292 | 3.314 | 3.300 |

A continuous feed runs 32 s at 3.376 GB/s with no decay — last-bin / first-bin is 0.974–0.993 for every arm
of 406 MiB or more. So the premise that the drive slows while it works is false, and the corollary matters
more: a feeder that keeps it busy forever earns the *same* 3.4 GB/s the burst already gets. There is no
sustained-read bonus to be had. (The summarizer prints the 0.165 MiB shape as `0MiBx16`; it is `%.0f` of
0.165, not a zero-length read.)

### #188 — a co-running feeder is strictly zero-sum, and it lands on the critical path

The question the operator's idea actually needs answered is not "can the drive do two things" but "does the
second thing cost the first anything". 200.7 MiB demand burst at the shipped shape, against a feeder reading
2 GiB of its own, 9 interleaved rounds, burst row is the measurement:

| arm | burst GB/s | burst wall ms | paired Δ vs same-round solo | feeder GB/s |
| --- | ---: | ---: | ---: | ---: |
| solo | 3.342 / 3.305 | 63 / 64 | — | — |
| thin (1 feeder worker) | 2.705 / 2.673 | 78 / 79 | −0.619 / −0.628 | 2.33 / 2.32 |
| full (4 feeder workers) | 1.710 / 1.694 | 123 / 124 | −1.642 / −1.633 | 3.08 / 3.07 |

All nine replication rounds are negative in both feeder arms. The two-averages figure this first suggested —
"aggregate 5.0 GB/s" — was an artifact: the feeder's 2 GiB window is mostly uncontended, so summing two
medians over two different windows invents bandwidth. #189's union-window instrument is what replaced it, and
the naive sums are labelled as naive wherever they appear.

### #189 — processes do not break the wall either; 3.31–3.38 at every count

Union-window aggregate, 2 GiB per process, 5 interleaved rounds, control noise floor 5.1 %, launch skew
sub-millisecond:

| p1w4 | p2w2 | p2w4 | p3w4 | p4w2 | p4w4 | p6w4 | p8w1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.385 | 3.378 | 3.383 | 3.375 | 3.370 | 3.334 | 3.331 | 3.308 |

Per-process share of the total falls as 1/P. Eight processes with one reader each get the same 3.31 that one
process with four readers gets. Whatever the 3.3–3.4 GB/s is, it is not a per-process limit, so no amount of
external feeding processes buys a faster device.

### #190 — splitting one demand burst across processes buys nothing

203 MiB delivered, gate-synchronised, 7 interleaved rounds: **63.2 ms** at 1×4, **63.0 ms** at 2×2 (paired
+0.023 — nothing), 64.5 ms at 4×1, 67.0 ms at 2×4, 71.5 ms at 6×2. A burst served by two cooperating
readers is not served faster than one.

### #191 — file scope: locality does not help, and sequential is the slowest arm

§5w reports the engine delivering 3.85/3.68/3.84/3.95/3.69 GB/s inside its own bursts (mean 3.80) at the
identical shape #186 measured at 3.36. The last structural difference between the two instruments is *which
files the concurrent reads touch*, so the address set was made the only variable — slice, worker count and
total bytes fixed at 2.641 MiB × 4 × 203 MiB — across two interleaved batches (9 rounds, then a 16-round
replication), with `spread` carried as the control in every round and the sharing arm's file rotated per
round so no arm is favoured by one platter range:

| scope | what it reads | GB/s (9 / 16 rounds) | paired Δ vs control |
| --- | --- | --- | --- |
| spread | random file, random expert | 3.310 / 3.310 | — |
| one | one file per worker | 3.320 / 3.317 | −0.005 / +0.029 |
| same | all four workers in one layer file | 3.218 / 3.234 | −0.100 / −0.049 |
| seq | consecutive expert strides in one file | 3.216 / 3.236 | −0.093 / −0.053 |

Two arms bracket the engine's own shapes: a token's ten routed experts live in one file (`same`), and a
token's 480 requests spread over all 48 (`spread`). Neither is faster, and maximal locality is ~3 % *slower*.
`seq` at −0.922 in one of its rounds is the same first-round warm-up shape seen in both batches — round 1's
control read 2.66–2.69 GB/s while every other arm of that round read high, which inflates all three of its
deltas; excluding round 1 the medians are `same` −0.06, `one` +0.03, `seq` −0.07. Short reads: 0 everywhere.

### The ceiling this section retires, and the one it replaces it with

Nothing the campaign can make this device do offline exceeds **3.39 GB/s** — across 30 slice × worker shapes
(#186), 8 burst lengths to 6.4 GiB (#187), 8 process counts (#189), 5 burst splits (#190) and 4 address sets
(#191). That contradicts three citations still standing in this doc and in `docs/v4-core-design.md`:

- the **3.92 GB/s** four-thread `ParallelExpertReader` ceiling (§5o),
- the engine's own **3.95 GB/s**,
- §5w's **3.80 GB/s "delivered inside each burst, 97 % of the ceiling"**.

§5w's number is a quotient, not a rate: `gap_total / gap_count` divided by `experts_per_miss_layer × stride`
credits the *whole token's demand volume* to the hit→fixup gap, while the ring's ~42.8 speculative reads per
token (§5v) are also flying in that window. Either the gap was not entirely demand-read time or not all of
those bytes flew inside it — both make the quotient larger than the device rate, and the device is what the
five scopes above just measured directly. **The planning number from here is 3.3–3.4 GB/s.** That does not
weaken §5w's conclusion, it strengthens it: if the bytes cost *more* device time than the section assumed,
the bill for a constant feed is bigger, not smaller, and moving them earlier is even more the only lever.

### Arithmetic on the idle clock the operator pointed at

How much idle clock there is depends on which of the campaign's two busy figures you believe: §5w's kernel
ledger leaves `217.34 − 95.7 = 121.6` ms of every token with the device not serving reads, while §5v's duty
sample (34.7 %) implies 141.9 ms. Take the smaller. Filling all of it at the wall measured here moves
`121.6 ms × 3.37 GB/s = 391 MiB` per token — roughly double the 203.55 MiB the token's demand misses actually
cost. So the idle clock is real and large enough; the reason the feeder still fails is not that there is
nothing to read, it is #188 (bytes read while a burst runs are taken from that burst) plus §5aa's landing
limit (42.8 issued per token, 18.56 adopted, `free=0.99`) — the tier has nowhere to put the extra bytes.

### #192 — the live session: the idle clock really is fillable, and filling it costs 17 % and 53 % of decode

The six offline probes share one limitation: their victim is a synthetic burst that never stops. A session is
different — §5v measured the device idle roughly two thirds of the time — so the same feeder was run beside ten
live greedy 384-token decodes at the shipped default (`--ram 12`, 96 slots, `--max-context 262144`,
`TINYTITAN_RUNNER_STATS=1`), in three interleaved rounds of control / 1-worker feeder / 4-worker feeder plus a
closing control, with `iostat -d -w 1 disk0` sampled across every arm:

| arm | tok/s median | paired Δ vs same-round control | hit rate | engine MiB/tok | device MB/s mean | % of the 3.37 GB/s wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 5.082 | — | 0.8457 | 197.3 | 1467.9 | 44 % |
| feeder ×1 | 4.216 | −0.870 (−17.1 %) | 0.8329 | 213.8 | 2724.9 | 81 % |
| feeder ×4 | 2.370 | −2.709 (−53.3 %) | 0.8112 | 241.5 | 3072.5 | 91 % |

Per-round paired deltas are −0.661, −1.008, −0.870 for ×1 and −2.717, −2.709, −2.650 for ×4: nine arm-pairs, all
negative, and no round in which a feeder helped. All ten arms emitted the same response hash
(`95aa4531b4ea2304…`), so this is scheduling and not numerics. The control's own spread is 4.6 %
(4.896–5.132), with round 1's control the low outlier — the same first-round warm-up #191 showed.

The premise is **confirmed** in the only sense the device can confirm it. Device utilisation rises from 44 % of
the wall to 81 % and 91 %, and the ×4 arm's busiest second is 3228 MB/s, 96 % of everything the offline probes
ever got out of this drive. The engine is not short of idle clock; it has never been using it. It is **refuted**
in the sense that matters: those bytes are not free, and they are worse than the fixed-pie split #188 measured.

**The demand volume is identical; only the classification moves.** `expert_hits + expert_misses` is 186 032 in
every arm of every round (484.5 per token), and round 1's `expert_evictions` (31 293) and `expert_reloads`
(22 986) are the same number in all three arms — the plan asked for the same experts in the same order and
chased the same slots. What changed is how many of them had already arrived: `prefetch_adopted_per_token` falls
19.01 → 12.52 → 4.66 while `prefetch_issued_per_token` barely moves, 39.45 → 37.98 → 35.93.

Three independent counters then agree to ±1 read, in all six feeder arms:

| round × arm | Δ demand misses | −Δ prefetch adoptions | Δ `expert_read_mib` ÷ stride |
| --- | ---: | ---: | ---: |
| R1 ×1 | +2 491 | 2 492 | 2 491.0 |
| R1 ×4 | +5 510 | 5 511 | 5 510.0 |
| R2 ×1 | +2 212 | 2 211 | 2 212.0 |
| R2 ×4 | +6 360 | 6 359 | 6 359.9 |
| R3 ×1 | +2 542 | 2 542 | 2 542.0 |
| R3 ×4 | +6 504 | 6 505 | 6 504.0 |

That is the mechanism, and it is in the code. A plan adopts a speculative read only if its buffer is ready when
the router's indices come back: `if let source = prefetched[...]` memcpys it into the slot, and anything without
a ready buffer falls through to `misses.append(index)`
([PreadExpertStreamer+Plan.swift:95-106](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Plan.swift#L95)),
while every demand miss charges `experts.count × expertStride` to the byte counter
([PreadExpertStreamer+Residency.swift:381](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Residency.swift#L381)).
A late speculative read is therefore **not used at all**: the same 2.641 MiB crosses the bus twice, once in the
window the feeder stole and once on the path the token waits for. That is why the engine reads *more* under a
co-tenant (197.3 → 213.8 → 241.5 MiB/token) — a fixed-bytes model like #188's cannot predict it, and it is the
reason a feeder is worse than zero-sum rather than merely zero-sum.

The time follows the bytes. Per token, `io_ms` goes 69.1 → 98.2 → 189.7 ms against token time 196.8 → 237.2 →
389.0 ms, while `wait_ms` (the host blocked on the GPU) only goes 92.3 → 98.2 → 124.0. The regression is the read
span, not a new stall — which is what a contention story should look like, and not what a scheduling-priority
story would.

What survives is the version of the operator's idea that the engine owns. Its ring issues 39.45 reads per token at
the shipped default and adopts 47 % of them, so about 21 reads — 55 MiB, 18 % of the 309.7 MiB/token this session
moves at the device — is read and then thrown away for arriving late. The argument that kills an external feeder
says the internal one is paying for its own lateness. Whether throttling it helps is #193, and
`TINYTITAN_PREDICTIVE_PREFETCH=0` has never been priced in this configuration:
`docs/qwen38-remaining-levers-plan.md:59` lists it as a check under hypothesis 1 and never ran it, while the
ring-depth study that retired two-and-three-ahead ran at `--ram-budget 8G` in September with no depth-0 arm.

### Deviations

1. **The probe's p99 was read out of bounds**, `all[n - n/100]` with `n < 100` evaluating to index `n`. It
   printed 0 for every short arm: #187's 64/128/203 MiB rows and #188's 76-read burst rows. Fixed to
   `all[n * 99 / 100]` and #188 re-run from the corrected binary (`/tmp/contention_rounds2.tsv`), so the
   #188 figures quoted here are from the fix. Rates, walls and p50 come from timestamps and a sorted array
   and were never affected. Note the converse degeneracy: with the fix, a 76-read row's p99 *is* its max, so
   #191's tail column is reported as max, not as a percentile.
2. **Binary provenance is not uniform.** #186 and #187 ran on the pre-edit binary (no pid in the seed, no
   gate, no bound on the gate wait). Both were single-process, so pid-in-seed and the gate are inert there;
   their rate/wall/p50 columns are valid. #188 was run once on the old binary and re-run on the corrected
   one; #189–#191 ran entirely on the corrected one.
3. **A `""` gate argument hung the smoke test.** An empty fifth argument was treated as a real path, so
   workers spun waiting for a file that never appears. Fixed (empty means no gate) and the spin is now
   bounded to 15 s with a loud exit, after it had cost two minutes. Probe defect, not a device finding.
4. **First-round warm-up in both #191 batches**, described above, is why per-round deltas are printed rather
   than only medians. #192's round-1 control is the same effect (4.896 against a 5.082 median), which is why
   its deltas are paired within a round and not against the batch median.
5. **Machine state before the live batch**: `sysctl vm.swapusage` read 6.19 GiB of 7.17 GiB already in use,
   `memory_pressure -Q` 82–83 % free, load 3.25, AC power, no model process. The swap is not something these
   probes created and it is sampled per arm in #192.
6. **Provenance.** The offline arms (#186–#191) ran at tree `b5fba5e`. 24 GB M3 MacBook Pro, macOS 27.0.1,
   Xcode 27 / Swift 6.4 toolchain present but not exercised — the probes are C, built `cc -O2 -Wall -Wextra`,
   and `swift build` was not re-run. No model loaded, no GPU work, no numerics touched. #192 is the exception
   on every one of those points: it loaded the installed 4-bit Qwen3.8-Flash-Next through the release binary
   built at 04:29 from the last source commit `9c858cd`, with `b5fba5e` (a benchmark-only change) as `HEAD`
   and this document the only dirty file. Its numbers are therefore attributable to a committed engine, but
   the tree they were measured in was not clean, and the standing rule is not to publish from one — so the
   #192 rows are a measurement of `9c858cd`, recorded here as such, not a shipped-artifact benchmark.
   Everything in §5af is a throwaway under `/tmp`; nothing shipped.
7. **The #192 driver averaged `iostat` over the wrong denominator.** It divided by every row past `NR>2`,
   including the trailing blank and zero rows, so each arm's device mean was understated by roughly 9–10 %.
   Every device figure in the #192 table was recomputed from the raw `.iostat` files with `NR>3 && $3+0>0`.
   The wall clocks, hit rates and byte counters came from the driver and are unaffected.
8. **`iostat`'s first data row is the since-boot average.** The driver's per-arm MiB total included it, adding
   ~47 MiB to ~121 206. The recomputed totals the table uses exclude it.
9. **The device window is the whole arm, not the decode loop.** `iostat` runs from before model load to after
   the last token, so the device MiB/token column is session-total bytes ÷ generated tokens and carries the
   load phase. The paired arm-to-arm deltas are still the right comparison; the absolute 197–241 MiB/token is
   an upper bound. The `engine MiB/tok` column is the engine's own counter and has no such inflation.
10. **The feeder's own achieved rate was not captured** — its stdout went to `/dev/null` — so its bytes are
    derived by difference between the recomputed device total and the engine counter, not read from the probe.
    Its process liveness and wall were captured.
11. **The driver's `load1` field is malformed** (`tr -d '{} '` collapses `vm.loadavg`'s spaces into
    `3.693.293.25`). Load was read off the raw sysctl for the record; no conclusion in this section rests on it.
12. **Swap moved during the live batch**: 7.35–7.85 GiB used across the ten arms against 6.19 GiB before it.
    §5v's warning is that swap movement, not the device, is what distorts a live arm. Here the regression is
    monotone in feeder intensity (−17.1 %, then −53.3 %) and the closing control returned to 5.132, which is
    what contention predicts and not what a swap cliff looks like — but the swap was there, and it is the
    strongest reason to re-run this arm on a machine that has not been fed before adopting anything from it.

### Verdict

**The drive wall is 3.3–3.4 GB/s and the shipped reader is already at it.** In-flight bytes set throughput
and 10.56 MiB is the knee (#186); sustained feeding is neither slower nor faster than a burst (#187); a
co-tenant's bytes come straight out of the demand burst, by 0.62–1.64 GB/s (#188); and no process count
(#189), burst split (#190) or address set (#191) moves the wall by more than the control's own noise. The
constant-feeder concept therefore dies in the offline limit for the same reason it died in the cache: the
machine can already be kept busy, and keeping it busy buys bytes the engine has nowhere to put. #192 then
tested the one version the offline probes could not see — the same feeder scheduled into a *live session's*
gaps instead of into a synthetic burst — and the session pays for it: −17.1 % decode with one worker, −53.3 %
with four, nine negative pairs out of nine, identical output hash. The idle clock really is fillable (44 % →
81 % → 91 % of the wall) and filling it is **worse than the fixed pie #188 measured**, because a speculative
read that arrives late is not used at all: the same expert crosses the bus again on the path the token waits
for. Refused as shipped behaviour. What survives of the operator's idea is the internal half of it — the
engine's own ring discards ~21 reads per token, and §5ag prices that and keeps the ring.

## 5ag. #193 prices the engine's own feeder by switching it off: it earns +5.6 % while spending 54 MiB per token on reads it throws away (2026-10-16)

§5af refused the *external* constant feeder and left the internal one unpriced. The shipped decode path already
does what the operator described — it keeps the drive reading while the GPU computes — through a one-slot staging
ring ([ExpertPrefetchRing.swift:154](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L154),
built with `slotCount: ringDepth` at
[RealForwardRunner+BuildCore.swift:99](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L99)).
`TINYTITAN_PREDICTIVE_PREFETCH=0` turns it off and has never been run at this configuration —
`docs/qwen38-remaining-levers-plan.md:59` lists it as a check under hypothesis 1 and never ran it, and the
September depth study that retired two- and three-ahead ran at `--ram-budget 8G` with no depth-0 arm.

Three paired rounds of ring-on (shipped) against ring-off, plus a closing ring-on, greedy 384 tokens, `--ram 12`,
96 slots, `--max-context 262144`, mains power, one process at a time, `iostat -d -w 1 disk0` across every arm:

| arm | tok/s | paired Δ vs same-round ring-on | hit | `io_ms` | `wait_ms` | engine MiB/tok | device MiB/tok | issued/pt | adopted/pt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ring on | 5.079 | — | 0.8459 | 68.3 | 92.6 | 197.1 | 309.1 | 39.45 | 18.86 |
| ring off | 4.793 | **−0.286 (−5.6 %)** | 0.8070 | 85.3 | 88.1 | 246.9 | 254.7 | 0.00 | 0.00 |

Per-round deltas −0.204, −0.492, −0.286: three pairs, all negative. Control spread 1.9 % (5.027–5.124) and the
closing ring-on returned to 5.043, inside it. All seven arms emitted the same hash (`95aa4531b4ea2304…`).

**The ring is worth 5.6 % of decode, and it is worth it while moving 18 % fewer bytes than it saves.** That
sentence is the finding, and the counters close on it exactly:

- Turning the ring off removes **20.59 issued reads per token** (39.45 − 18.86 adopted) and the device total
  falls by **54.4 MiB per token** — `20.59 × 2.641 MiB = 54.4`, to the read.
- Turning it off adds **+7 242 demand misses** over the session, which is **18.86 per token** — the adopted
  count, exactly. `18.86 × 2.641 = 49.8 MiB` per token returns to the demand path, and the engine's own
  `expert_read_mib` rises 197.1 → 246.9.
- So the ring's trade is: spend 54.4 MiB/token of device traffic to move 49.8 MiB/token off the critical path.
  `io_ms` 85.3 → 68.3 is the payment landing, and it is worth 5.6 % of the clock.

**Refused: do not throttle or remove the speculative ring.** The half of it that is discarded is the price of
the half that arrives ahead of the router, and §5af just measured what happens to a token when something else
competes for those bytes instead.

**The two shapes this arm might have suggested are already closed, and saying so is not redundant — #193 is
their first re-confirmation at the shipped tier.** `docs/qwen38-prefetch-predictor-study.md` scored the staged
*choice* against ground truth from the prefetch trace, at the September `--ram 8` meaning (64 cache slots), and
found the shipped probe the best ordering available: P@1 0.552, P@2 0.450, P@3 0.379, while frequency
re-ranking, the previous token's route and the previous token's misses are neutral or worse — the temporal
predictors stage *nothing*, because the cache has already absorbed that reuse. Its monotone decay by rank also
settles what this section would otherwise have had to open: the list the ring truncates with `prefix(count)`
([ExpertPrefetchRing.swift:88-98](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L88))
is already in router-weight order, so the staged expert is the highest-weight non-resident prediction rather
than an arbitrary one, and the caller's residency walk decides where it stops. The same study priced the
*volume* question and lost every arm: 2 slots −3.3 %, 3 slots −10.7 %, mixed horizon −1.4 %, L+2 horizon −3.7 %,
all byte-identical.

What #193 adds is the two things that study did not have: a depth-0 arm at all, and the current 96-slot tier at
5.08 tok/s against its 4.56. The conclusion survives the move — the ring earns +5.6 % here — and one number
moves. Adoption per issued read is 18.86 / 39.45 = **0.478** at 96 slots against 23.3 / 42.8 = **0.544** at 64,
and issued reads fall 42.8 → 39.45. Both halves of that move are what a bigger cache predicts: fewer
predictions are worth staging, and more of the ones that are get loaded by the demand path before the staged
read is wanted. That is an inference from two counters in two batches, not a measurement, and it is the kind
that must not be carried into a decision — the study's own note that its `--ram 8` rows are not this tier is
exactly why. #194 scores it from a trace at the shipped configuration.

The ring's own footer also says which failure mode a deferral would have to avoid: over 18 001 `begin` calls the
single slot is observed free 0.99 of the time and in flight 0.01, with a mean speculative load of 1.21 ms
against a 3.75 ms layer window (`body_ms 179.9 / 48`). A staged read nearly always *finishes* in time; what it
fails to be is the right expert. So lateness is not the loss channel at depth 1, and any deferral is spending
the one thing the ring actually has — its headroom.

The shape neither batch tested is **ordering at fixed volume**: keep the 39.45 speculative reads a token, and
issue each one after that layer's demand burst has taken service, so the same bytes land in the idle window
instead of at the head of the queue. It can add no device traffic and remove none, which makes it the only
remaining version of the operator's "read constantly" premise that §5af's contention result does not already
kill — and the only version the engine itself owns. Its risk is named above: it trades headroom for queue
position, and adoption depends on the headroom. That is #194.

### Deviations

1. **Seven arms, not eight.** The plan was three rounds of the pair plus a closing control; the closing control is
   ring-on only, so it is a drift check and carries no pair. Reported as measured.
2. **Provenance.** `HEAD 8e82098` (§5af, docs-only); the binary is the release build from `9c858cd` at 04:29, the
   same one §5af's #192 ran, and the tree was clean at launch. 24 GB M3 MacBook Pro, macOS 27.0.1, Swift 6.4.
   Nothing was changed in engine code to run this arm — the toggle is the existing env switch — so both arms are
   the shipped engine, and the only unshipped thing here is the record.
3. **The device window includes model load**, as in #192, so `device MiB/tok` is session-total ÷ 384 and the
   309.1 / 254.7 absolutes carry the load phase. The *difference* between the two arms is what the arithmetic
   above uses, and the ring-off arm's 7.8 MiB/token residual is the load and non-expert traffic, which is
   consistent with #192's control.
4. **`engine MiB/tok` is demand-path bytes only.** `recordSuccessfulLoadsUnlocked` charges
   `experts.count × expertStride` on the demand load path and the adoption path only bumps the load count
   ([PreadExpertStreamer+Residency.swift:379-399](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Residency.swift#L379)),
   so speculative reads never appear in `expert_read_mib`. That is why the engine/device gap is the ring's
   traffic and not a discrepancy — the gap in the ring-on arm (112.0 MiB/token) is `39.45 × 2.641 = 104.2`
   plus the 7.8 MiB residual, which closes to 0.4 %.
5. **Swap** 7.66–7.92 GiB (ring on) and 7.69–7.82 GiB (ring off): both arms ran under the same pressure, and the
   closing control sat inside the round range, so no arm is explained by a swap move. Load and pressure at launch
   were as recorded for §5af; no model process was running before or after the batch.
6. **`adopted_pt` medians are 18.86 where §5af quoted 19.01** for the same shipped configuration. Different batch,
   same arm, 0.8 % apart — inside the control spread of both batches, and the identity claims above use each
   batch's own numbers rather than a cross-batch mix. Never compare a qwen38 clock across batches.
7. **`ExpertPrefetchRing` has no unit test.** A repo-wide search finds the type only in `sources/` and `docs/`.
   What *is* covered is the config seam — `tests/TinyTitan/Runtime/Configuration/ModelProfileTests.swift:116` and
   `ExpertCacheBudgetTests.swift:211-240` assert `prefetchDepth` 1 and 0, which is the arm #193 ran — but nothing
   pins the ring's `.completed` gate or its `prefix(count)` truncation. Any change #194 makes has to start by
   testing the adoption contract, not by changing it.

## 5ah. #194 reads the ring's shortfall off a trace: 93 % of the discarded reads were the wrong expert, 7 % were right but late, and the late pool is too small to measure (2026-10-16)

§5ag left one number unexplained and one idea alive. The number: the ring issues 39.45-39.59 reads a token and
adopts 47-48 % of them, so 20.5 reads a token are paid for and thrown away. The idea: if a meaningful share of
those 20.5 were *correct* reads that merely missed their layer, then re-ordering them — issuing after the layer's
demand burst has taken service, or awaiting them briefly at plan time — would convert them at zero extra bytes.
§5ag called that the only untested shape. This section tests it, and it does not survive the test.

Two greedy 384-token decodes of the same prompt at the shipped default (`--ram 12`, 96 slots,
`--max-context 262144`, `TINYTITAN_RUNNER_STATS=1`), each writing `TINYTITAN_PREFETCH_TRACE`: one with the ring
on, one with `TINYTITAN_PREDICTIVE_PREFETCH=0`. 18 384 records each (383 positions × 48 layers), and the two
responses are byte-identical (`md5 c172890df88cdbbfa0627bd320d16614`).

### The offline replica, and why its output can be trusted

`ExpertPrefetchRing.begin` is re-implemented offline exactly: at layer L the staged expert is the **first
element of `next_layer_prediction` that is not in the target layer's `resident` list**, taken from the record at
(`position`, L+1) — the same set the caller passes in
([RealForwardRunner+DecodeMoE.swift:322-349](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L322)),
`model.routedExpertResidentIDs(layer: L + 1)`. Each staged read is then classified against that layer's own
record: *correct* if it is in `experts` (the exact route), *adopted* if it is in `experts` and **not** in
`misses` (the trace's `misses` are post-adoption, so a read the plan took from the ring is not a miss), *late* if
it is in `experts` and still in `misses`.

| check | replica | engine's own counter |
| --- | ---: | ---: |
| staged reads per token | 39.73 | `prefetch_issued_per_token` 39.59 |
| adopted reads per token | **19.25** | `prefetch_adopted_per_token` **19.25** |
| demand misses removed per token | **19.25** (33 709 − 26 316 between the two arms) | `expert_misses` 35 901 − 28 508 = **7 393** = 19.25/token |
| speculative MiB/token implied | 104.9 | §5ag's engine/device gap, 112.0 − 7.8 residual = 104.2 |

Three counters agree to the read. The replica over-counts *issues* by 54 reads (0.35 %) because `active`
deduplication and the rare busy slot are not modelled; every one of those 54 is a wrong read, so the adopted and
miss identities are exact while the issued figure is a 0.35 % upper bound.

### The decomposition: 92.7 % precision, 7.3 % timing

| per token | reads | MiB | share of the 20.48-read shortfall |
| --- | ---: | ---: | ---: |
| issued | 39.73 | 104.9 | — |
| correct **and** adopted | 19.25 | 50.8 | — |
| correct but not adopted (**late**) | **1.50** | 3.95 | **7.3 %** |
| not in the route at all (**wrong**) | **18.98** | 50.1 | **92.7 %** |

The identity closes on the counts: 15 257 issued = 7 393 adopted + 575 late + 7 289 wrong, and 7 968 correct =
7 393 adopted + 575 late. Nothing in the ring is unaccounted for.

**The lateness channel is spread over the layer stack, not concentrated at a boundary**: 45 of the 47 target
layers have at least one late read (only layers 1 and 31 have none), the largest count is 33 at layer 38 (9.6 %
of that layer's 345 issues), and layers 45-47 together carry 19 of 575. So there is no structural discard to
fix — not the token-boundary
`beginToken()` reclaim, not the last layer's slot. The ring footer agrees: over 18 001 begins the single slot is
free 1.00 of the time and submitted, in flight and held are all 0.00, with `spec_load_ms` 1.22 against a 3.91 ms
layer window (`body_ms 187.974 / 48`). Reads land inside the window; they are simply not the expert the router
asks for.

### What that prices, and why #195 is not run

The pool a perfect await could convert is 1.50 reads/token = 3.95 MiB/token of duplicate crossings. Three
routes to a price, all from this campaign's own measurements:

1. **Linear in the byte savings #193 just measured.** §5ag: removing the ring's 49.8 MiB/token of *demand*
   savings costs 5.6 %. 3.95 / 49.8 × 5.6 % = **+0.44 %**.
2. **The exchange rate.** 2.641 MiB at the §5af wall (3.3-3.4 GB/s) is 0.78-0.80 ms, so 1.50 reads = 1.20 ms of
   a 196.8 ms token = **+0.61 %**.
3. **A two-anchor per-MiB price model.** #193 gives `49.8d − 104.2c = 5.6` and #192's 1-worker feeder gives
   `322.9c + 17.1(d − c) = 17.1`, where `c` is the cost of a speculative MiB and `d` the value of an avoided
   demand MiB. Solving: `c = 0.042 %/MiB`, `d = 0.201 %/MiB`, so awaiting a late read is worth 1.50 × 2.641 ×
   0.201 = **+0.80 %** before its own cost. That model is fitted to those two arms, so it reproduces them by
   construction; its useful output is the break-even below, not the 0.80 %.

Route 3's upside is the ceiling, not the expectation: the await puts the read's remaining load time on the
critical path, and 0.3-0.5 % of token time is a plausible cost for it, so the honest range is **+0.3 % to
+0.8 %**. This campaign's control spread in the same configuration is 1.9 % (§5ag) and 4.6 % (§5af). **No arm
run here can resolve the effect it predicts.** #195 — ordering at fixed volume, and its bounded-await cousin —
is screened out before being built, and the ring's adoption contract therefore needs no test for a change that
will not be made. That is the answer §5ag's last deviation line was holding open.

### Two things the trace says that no prior record does

**Rank order is the right staging rule, measured.** Precision by the rank of the first non-resident prediction
falls monotonically: rank 1 0.754 (2 151 issues), 2 0.687 (1 999), 3 0.611 (1 916), 4 0.560 (1 835), 5 0.501
(1 647), 6 0.415 (1 478), 7 0.403 (1 320), 8 0.325 (1 143). The ring already stages the best-ranked
non-resident candidate, so within one read a layer there is no better *choice* available in its own list.
Overall precision here is 0.522 — the September study's 0.552 at the 64-slot tier, same shape, slightly worse.

**Per-layer precision varies 4×, and it is not actionable.** Layer 6's staged reads are 0.773 correct; layer
42's are 0.310 and **layer 47's 0.181** (66 correct of 364 issues). Under the route-3 prices a read is worth
issuing at all when its precision exceeds `c/d` = 0.21, so only layer 47 falls below the line — and dropping its
staging is worth **+0.02 %** (it saves 2.50 MiB/token of speculative bytes and gives up 0.44 MiB/token of demand
savings). The reason layer 47 routes so much less predictably is *not* established here; the probe reads layer
46's residual and misses 82 % of the time at the layer above it, and no candidate explanation was tested. It is
recorded as a finding and not as a lever, because the screen above says it cannot be cashed at depth 1.

### The size of what is actually left

Scored against the ring-off arm's true demand need, **34.65 of the 48 layer-slots a token have at least one
demand miss** (histogram: 5 080 layers need nothing, 4 573 need one, 3 370 need two, 2 272 need three, and 20
need all ten). At depth 1 the ring can adopt at most one read per layer, so that is the ceiling on adoption:
**19.25 achieved against 34.65 available**. A perfect next-token predictor would issue *fewer* reads than the
shipped ring does (34.65 against 39.73) and adopt every one of them, worth roughly +4.6 % by route 1's
arithmetic — and it would remove 13 MiB/token of traffic rather than adding it.

That is the campaign's answer to the feeder premise, and it is the same answer the September study gave from
the other direction. `docs/qwen38-prefetch-predictor-study.md` measured live per-layer frequency, the previous
token's route and its misses, an EMA of probe scores, their unions, a second rank, a second horizon and a second
layer of lookahead; none beat the shipped probe, and every wider arm lost 1.4-10.7 %. What remains after §5af,
§5ag and §5ah is therefore not a scheduling question at all: **the bytes the ring throws away are the price of
the 47 % it keeps, the price is set by prediction quality, and prediction quality is a model-side problem this
engine cannot manufacture by re-ordering reads.** The decode line for qwen38 4-bit on this machine is closed at
the shipped tier by three independent measurements — remove the ring and lose 5.6 %, feed the device from
outside and lose 17-53 %, aim the ring better and find that 93 % of what is left is already known to be wrong.

### Deviations

1. **These two arms are traced, so their clocks are not the shipped ones.** 4.865 (ring on) against 4.496 (ring
   off) is −7.6 %, where §5ag's untraced pair measured −5.6 %; `TINYTITAN_PREFETCH_TRACE` writes 18 384 JSON
   lines in both arms and is not part of the shipped path. The pair is used for *structure* (which is what #194
   asks) and never for magnitude. The classification counters agree with §5ag to a tenth of a point: hit rate
   0.8468 / 0.8070 here against 0.8459 / 0.8070 there.
2. **383 positions, not 384.** The trace covers positions 10-392; the last decode step is not recorded, so the
   replica denominators are 383 × 48 records while the per-token figures divide 384. That is a ≤0.26 % skew on
   any count that ends in the untraced step.
3. **`late` means "in the route and not adopted", which is the union of three causes.** A still-in-flight buffer
   at `readyBuffers`, a failed operation and a slot reclaimed before its plan all look the same from the trace.
   The footer's `inflight = 0.00` is sampled at the *next* `begin`, which runs after that layer's plan, so it
   cannot exclude in-flight-at-`readyBuffers`; what it does exclude is a read still running at the end of its
   own layer. So 1.50/token is an **upper bound** on genuine lateness, which makes the screen above conservative
   in the direction that matters.
4. **The residency filter is read one instant late.** `resident` in the trace is captured before that layer's
   plan; the ring queried the same set at the previous layer. For depth 1 the two are identical — nothing writes
   layer L+1's cache between them, because adoption happens inside layer L+1's plan, which the snapshot precedes.
   At depth 2 or 3 they would differ, so this replica is only valid for the shipped geometry.
5. **The ceiling is scored against the ring-off arm on purpose.** The ring's own adoptions shrink the demand need
   visible in its own trace (68.53 misses/token against 87.78), so measuring headroom on the ring-on records
   would score it against a need it has already eaten.
6. **The scorer is a throwaway.** `/tmp/score194.py` is not in the tree; the algorithm is specified above
   completely enough to rebuild it, and the trace facility it reads is shipped. Nothing in `sources/` changed to
   produce this section, so no test was added and the golden baseline is **not checked** (docs-only record, as in
   §5af and §5ag).
7. **Single round per arm.** #194 is an offline scoring exercise over two traces, not a paired timing batch, so
   the interleave-and-control rule does not apply to it; the only clock it touches is deviation 1's, and that is
   reported as contaminated. The tree was clean at `HEAD cd229f6`, the binary is the 04:29 release build, AC
   power at 100 %, `memory_pressure -Q` 86 % free, swap 6.65 GiB used at launch and 6.63 GiB after, no other
   model process before or after either arm.

## 5ai. #195 separates MTP's byte term from its time term at the shipped tier: a width-2 verify pass moves 0.79x the device traffic the shipped path moves per token, so the recorded loss is a time story — and the number a rewrite would have to beat is 1.582x (2026-10-16)

§5ah ends by naming prediction quality as the only lever left, and the one trained next-token predictor in this
tree is the MTP draft head. §8 carries MTP as closed on three records — 2026-09-18 (a pair pass at **2.238x** a
scalar token, 86.9 % acceptance), 2026-09-21 at `--ram 8` / 64 slots on a machine Chrome was using (−32.3 % and
−46.9 % paired, and the project's own B3 gate at **−43.2 %** with 85.7 % acceptance and byte-identical output),
and the archived 8-bit case (1.926 emitted per pass, −2 %). Two things about those records are checkable without
a model run: whether the *byte* half of the price was ever separated from the *time* half, and whether either
half moved when the tier went 64 → 96 slots and B1 lifted the hit rate 0.7577 → 0.8459.

**The trace answers the byte half.** A width-W verify plan loads the union of its W rows' selections as one
cache plan (`RealForwardRunner+MTPVerify.swift:64-68`), so its expert traffic is `|union(experts) \ resident|`.
Consecutive trace positions are consecutive decode tokens and `experts` is the target router's exact selection,
so that union is readable from #194's ring-off trace without running MTP. The replica is validated the way §5ah
validated its own: `misses == experts \ resident` holds on **18 384 of 18 384 records**, and the demand it
reconstructs is 88.013 reads/token against the footer's 87.78.

**Route locality is the reason the union is cheap, and it is measured here for the first time.** Of the second
row's 10 selections at a layer, over **18 336** sliding consecutive-pair layer-slots (both rows present, residency
read at the first row as a verify plan would read it; the three columns partition the 10 by construction and sum
to 10.000):

| where the next token's expert is | per layer | note |
| --- | ---: | --- |
| also selected by the previous token | **3.820** | independence over 512 experts would give 0.195 |
| not in it, but already resident | **4.439** | |
| needs a fresh read | **1.741** | the disjoint-pair tiling the width table uses puts the same marginal at 1.750 |
| control: 10 tokens apart (17 904 slots) | 1.878 / 5.652 / 2.471 | locality decays but does not vanish |

That closes a question the September study left as an assertion. It found the previous token's route scores
P@1 0.000 as a prefetch ranking and wrote "the previous token's experts are always already resident, which is
why a temporal predictor stages nothing at all". Now both halves have a number: the reuse is real (38 % of a
token's selections), and the cache is what absorbs it, so there is nothing left for a temporal predictor to
stage. The same locality is why MTP's second row costs 1.74 reads rather than 10.

**Width table** — `E` is the emitted tokens per pass at the recorded acceptance 0.869 (`E = Σ pᵏ`, capped at
1/(1−p) = 7.63), and the two ratio columns are the shipped path's own counters from §5ag: 309.1 MiB/token of
device traffic, of which 197.1 MiB is demand-path:

| W | passes | reads/layer/pass | MiB/pass | E | MiB per emitted token | vs 309.1 (device) | vs 197.1 (demand) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 383 | 1.834 | 232.4 | 1.000 | 232.41 | 0.752 | 1.179 |
| 2 | 191 | 3.584 | 454.3 | 1.869 | 243.07 | **0.786** | 1.233 |
| 3 | 127 | 5.254 | 666.0 | 2.624 | 253.79 | 0.821 | 1.288 |
| 4 | 95 | 6.879 | 872.0 | 3.280 | 265.81 | 0.860 | 1.349 |
| 5 | 76 | 8.447 | 1070.7 | 3.851 | 278.06 | 0.900 | 1.411 |
| 6 | 63 | 10.044 | 1273.0 | 4.346 | 292.91 | 0.948 | 1.486 |
| 7 | 54 | 11.621 | 1472.9 | 4.777 | 308.35 | 0.998 | 1.564 |
| 8 | 47 | 13.069 | 1656.5 | 5.151 | 321.58 | 1.040 | 1.632 |
| 13 | 29 | 20.358 | 2580.4 | 6.403 | 402.98 | 1.304 | 2.045 |
| 42 | 9 | 52.414 | 6643.5 | 7.613 | 872.70 | 2.823 | 4.428 |

**What that says about the closure.** The union does keep growing per emitted token — 232.4 → 872.7 MiB from
W=1 to W=42 — so `RealForwardRunner+MTPVerify.swift:31-34`'s stated reason is directionally right, and this
table is the first quantification of it on qwen38 (their 5.18x at W=13 and 11.25x at W=42 are Qwen3.6-35B-A3B,
256 experts, top-K 8, a different cache). But against the baseline a shipped token actually pays, the growth is
not what fails MTP: **a pair pass costs 0.786x the shipped path's device traffic per emitted token**, and the
byte race stays favourable to W=7 (308.35 against 309.1 MiB) before W=8 turns it. The reason is that the two
mechanisms are substitutes for the same locality: the ring spends 54.4 MiB/token on reads it throws away
(§5ag), and the draft's second row needs only 1.74 fresh reads because the cache already holds 8.26 of its
10 selections.

So the −17 % of 2026-09-18 and the −43 % of 2026-09-21 cannot be a byte story, and the record already knew
that — "the union is not where it goes", with `verify_routed_pair` at 1.45x the scalar routed GPU time. The cost
lives in the two **time** terms the same header names: two rows through the 32-token prefill kernels at
1.6-1.7x where the decode path is 1.0x, and ~200 ms/pass of host and commit (0.93x a token), where the
sequential fetch awaits nothing at width 2. Neither term depends on the cache tier, which is why 64 → 96 slots
and +8.9 points of hit rate could not have flipped them. **MTP stays closed.**

**The number to beat, if anyone revisits it.** Starting from 5.079 tok/s (§5ag, shipped), break-even needs a
pass costing ≤ E = 1.869x a token and 6.0 tok/s needs **≤ 1.582x** (1.869 ÷ 6.0/5.079). The best width-2 pass
ever measured anywhere in this tree is 1.585x — on Qwen3.6, and as a union model, not a qwen38 decode-native
kernel. So the design target is a 29 % cut in the pass's time from 2.238x, and since the byte term is already
favourable, the whole cut has to come out of the prefill-path inefficiency and the per-pass host work. That is
a much better-specified question than the "large rewrite against a gate that fails by 43 points" the 2026-09-21
verdict recorded, and it is the one thing this section leaves open.

**Not run: the live arm, refused by its own guard.** The arm that would price the two time terms at the shipped
tier — `benchmark/tinytitan_mtp_phases.py --target models/qwen3.8-flash-next_125B_A6B_4Bit --sidecar
models/qwen3.8-flash-next_125B_A6B_MTP_4Bit --ram-budget 12G --verify-arm pair --presence-penalty 0 --pairs 1
--warmups 1` — reached `gate0.preflight(15)` and stopped: "GPU is already 51 % busy before TinyTitan launches".
`idle_gpu_utilization()` (`ioreg -c AGXAccelerator`, "Device Utilization %") read 51 % then 45 % on a second
sample, no model process was live, and **`--allow-busy-gpu` was not passed**, so nothing was measured and no
number in this section comes from a contended GPU. The sidecar is present and its manifest guard passed:
`models/qwen3.8-flash-next_125B_A6B_MTP_4Bit/{manifest.json, model_weights.bin 51 681 440 B, packed_experts/}`,
which the driver mounts at `--mtp-memory-mib 384`. It is still the right next run — it also settles the one
correctness finding the tree has never explained, the prose-scenario divergence at character 553 that §8's own
verdict says must be explained "before quoting any speed number".

Deviations.

1. **The byte denominators are three, and the table shows two of the comparisons.** 232.41 MiB/token is this
   trace's own demand (ring off, 96 slots), and is the denominator the union is measured against in the same
   records; 309.1 is §5ag's device traffic including 104.2 MiB of speculation and a 7.8 MiB load/non-expert
   residual; 197.1 is the shipped demand path. So the only strictly internal figure is 243.07 ÷ 232.41 =
   **1.046x** — a pair costs 4.6 % more demand bytes than a scalar token — while the 0.786x headline is the
   comparison against the shipped alternative the reader is choosing between, and it inherits §5ag's deviation
   that the device counter includes model load, which makes it *harder* for MTP, not easier.
2. **The first pass at this scored the wrong ratio.** Dividing the union by W (rows) instead of E (emitted
   tokens) produced 0.977x at W=2 and, at W=42, 0.681x — a flattering full-acceptance bound that would have
   read as "widening is byte-free". The correction is why the table divides by E and reports W=42 at 2.823x.
3. **The tables were re-derived from one loop each before this commit, and four cells changed.** The first
   draft mixed outputs: the locality table's resident-only and fresh columns (4.352, 1.828) came from a
   disjoint-pair loop while its shared column and slot count came from the sliding one, and the W=13 / W=42
   `MiB/pass` cells (2493.6, 6399.9) do not reproduce from the width loop, which gives 2580.4 and 6643.5. The
   republished numbers come from `/tmp/score195c.py` (widths, disjoint tiling, row-0 residency) and
   `/tmp/locality195.py` (locality, sliding, same snapshot). Nothing the argument rests on moved: the
   MiB-per-emitted-token column and both ratio columns are what the single method already gave, and the
   sliding and disjoint marginals agree to 0.009 reads (1.741 against 1.750). **W=13 is 29 disjoint passes and
   W=42 is 9**, so those two rows are sample-thin; W ≤ 8 has 47 passes or more.
4. **E assumes independent acceptance.** Real qwen38 acceptance is measured per pass (85.7-92.6 %) and streaks
   are autocorrelated; the pass also pays all W rows' bytes up front, so a draft rejected at rank 2 of 6 has
   already staged rows 3-6. That waste is invisible to this accounting, and it grows with W — one more reason
   the short drafts are the only byte-favourable ones.
5. **Residency comes from a ring-off demand trace.** A live MTP session loads two tokens' experts per pass, so
   its cache holds more of the next union and its misses are fewer: the reads in this table are an **upper
   bound** on the byte bill.
6. **The draft's own traffic is not here.** The 384 MiB sidecar has its own `packed_experts` and runs a proposal
   pass per verification pass; its bytes, and every non-byte term (KV, PLE, head, commit, rollback), are outside
   this accounting. `tinytitan_mtp_phases.py` reads them as `proposal=` milliseconds, not as bytes.
7. **One session, one prose prompt, positions 10-392.** Same trace as #194, so the same ≤0.26 % denominator
   skew and the same prompt-shape limit; the code-shaped decode in the September study is not represented.
8. **`io_ms` is not converted to a rate anywhere in this section.** §5ag's marginal 17 ms for 54.4 MiB is 3.20
   GB/s and matches the saturated device, but its absolute io_ms/bytes ratio does not, so claims here are byte
   claims only.
9. **My own launch masked the driver's exit status** by piping it through `tee`, so the background task reported
   exit 0 while the driver raised `SystemExit`. The refusal text in the log is the evidence; the status was
   lost. That is the AUD-230 shape, in my command line rather than in the driver.
10. **Nothing in `sources/` changed.** `/tmp/score195.py`, `/tmp/score195b.py`, `/tmp/score195c.py` and
    `/tmp/locality195.py` are throwaways, so no test was added and the golden baseline is **not checked**
    (docs-only, as in §5af-§5ah). Provenance: tree clean at `HEAD 0ef95b7`, 04:29 release binary with no
    `sources/` commit since (`git log -1 -- sources/` = 04:28:32), AC 100 %, `memory_pressure -Q` 85 % free,
    swap 6.41 GiB, load average 3.42, no model process before or after.

## 5aj. #196 makes the leakage rule structural in the shipped screen: a bounded strictly-past recency ranking names experts that really are selected 65.4 % of the time and still finds only 16 worth-staging picks in 18 001 links, because the cache already holds 17 985 of them (2026-10-16)

§5ah ends by naming prediction quality as the only lever left, and §5ai closes the last predictor-shaped question
that could be answered without a GPU (MTP's byte half). Everything that campaign used to rank candidates was a
throwaway in `/tmp` — including the script that scored the previous window's temporal arms. So this task's
question was not "is there a better ordering", asked a fourth time, but the one thing that would make the answer
reusable: **can the shipped screen be made incapable of answering it dishonestly.** The standing instruction
here is "do not rely just on old tests", and the old tests could not have caught what this section reports —
the falsifier paragraph below runs them against a deliberately leaky build and they pass.

**The rule, stated once.** When the ring issues a plan for layer *L+1* at position *p*, the engine knows the
earlier positions' selections at layer *L+1* and position *p*'s layers ≤ *L*. It does not know *(p, L+1)* — that
is what it is trying to guess. Any feature that reads the target row (its selection, its misses, its residency)
is not a predictor, and on this trace reading it inflates a 152-issue, 13-useful arm into 1 994 issues at
precision 0.9975, which is why the rule had to become code rather than a comment.

**What landed** (`894ecfa`, all of it in `benchmark/`):

| piece | where | what it forbids or enables |
| --- | --- | --- |
| `Link` | `tinytitan_prefetch_trace_analysis.py:40` | one *L → L+1* plan, carrying only `source`, `target`, `top_m`, `counts`, `history` — a ranker cannot reach anything else |
| three uniform rankers | `:97`, `:102`, `:110` | `recorded`, `transition`, `recent`, all `Link -> list[int]` |
| `RANKERS` / `PREDICTORS` | `:120-121` | one derived name list feeds `--predictor`'s `choices` *and* `analyze`'s refusal, so an arm cannot exist in the CLI and not in the guard |
| position-boundary publish | `:164-165` | history is appended **after** every link at that position is scored — the leak is structurally unavailable |
| `--window` | `:184-194` | depth of strictly-past history, default 8, refuses ≤ 0 |
| name refusal | `:131-132` | `unknown predictor 'frequency'` raised, not silently scored as `transition` |

Five tests now pin it (`test_tinytitan_prefetch_trace_analysis.py:17, 33, 42, 55, 67`): the pre-existing
transition-prior recall test, the unknown-name refusal, a positive test that `recent` credits a past selection
the probe misses, the no-target-row test, and the window-bound test. All five pass; the python gate is clean
(`ruff 0.16.7`, check and format, 201 scripts parse under 3.13).

**The falsifier, run rather than asserted.** I copied module and tests to `/tmp/leakcheck` and moved the publish
above the scoring loop — a one-line change that makes the target position's own selection visible to the ranker.
Three tests fail, with the exact messages `3 != 1`, `1 != 0` and `4 != 0`, and **the two pre-existing tests stay
`ok`**. That is the "old tests" finding in concrete form: the suite that shipped before this commit certifies a
leaky screen, because it only ever scored arms that cannot read the future.

### The shipped screen's own numbers, on #194's ring-off trace

18 384 records, 383 positions, 48 layers, identity `misses == experts \ resident` holding on 18 384 of 18 384,
18 001 scored links (layer 47 has no *L+1*), 32 046 actual misses:

| predictor | top_m | window | issues | useful | recall | precision | selections named (of 10) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recorded (shipped probe) | 1 | — | 2 151 | 1 622 | 0.0506 | 0.7541 | **0.919** |
| recorded | 2 | — | 4 899 | 3 486 | 0.1088 | 0.7116 | 1.798 |
| recorded | 4 | — | 12 064 | 7 587 | 0.2368 | 0.6289 | 3.402 |
| recent | 1 | 4 | 21 | 1 | 0.00003 | 0.0476 | **0.654** |
| recent | 1 | 8 | **16** | 1 | 0.00003 | 0.0625 | 0.654 |
| recent | 1 | 16 | 9 | 0 | 0.0 | 0.0 | 0.650 |
| recent | 2 | 8 | 44 | 1 | 0.00003 | 0.0227 | 1.192 |
| recent | 4 | 8 | 152 | 13 | 0.0004 | 0.0855 | 2.085 |
| transition | 1 | — | 36 | 30 | 0.0009 | 0.8333 | 0.722 |
| transition | 2 | — | 88 | 71 | 0.0022 | 0.8068 | 1.324 |

The last column is the screen's `raw_topk_agreement` × 10 — the mean number of the ten actually-selected experts
that the plan names. At top_m = 1 it is also a probability, and the honest way to read those three rows is as
probabilities against a chance level of 10/512 = **1.95 %** (one pick from 512 landing among the 10): the probe
hits 91.9 % of the time, the transition prior 72.2 %, and a bounded strictly-past recency ranking **65.4 %, 33×
chance**.

**The finding is the issue column, not the precision column.** `recent` is not a bad guesser: 65.4 % of its
single picks are experts the router really does select at *L+1*, 33× chance, and the transition prior reaches
72.2 %. It is a bad *staging* rule, because 17 985 of its 18 001 picks are already resident when the plan is
issued — so only **16 links out of 18 001** offer it anything to read, and one of those 16 turns out to be
needed. Deeper history makes that worse, not better (window 16 → 9 issues, 0 useful), and widening does too: at
depth 4 `recent` reaches 0.0004 recall where the probe reaches 0.2368, 592× the demand.

That is §5ai's locality table read from the other side. It measured that the previous token already holds 3.820
of the next token's 10 selections and the cache covers 4.439 more, leaving 1.741 fresh reads; this section
measures the consequence for a predictor — **recurrence and cache residency are the same mechanism**, so a
temporal ranking spends its entire vocabulary on experts the 96-slot ring already has. That is why the September
study's `previous token` arms scored P@1 0.000 and why the three temporal shapes in §5w staged nothing live: not
a scoring artifact, an empty order.

### What the leak would have bought

Same trace, leaky module, `recent` only (`recorded` and `transition` are byte-identical between the two modules,
so the leak is specific to the arm that reads history):

| arm | clean: issues / useful / precision | leaky: issues / useful / precision |
| --- | --- | --- |
| `recent` top_m=1 w=4 | 21 / 1 / 0.048 | 213 / 213 / **1.0000** |
| `recent` top_m=1 w=8 | 16 / 1 / 0.063 | 137 / 137 / **1.0000** |
| `recent` top_m=1 w=16 | 9 / 0 / 0.000 | 119 / 119 / 1.0000 |
| `recent` top_m=2 w=8 | 44 / 1 / 0.023 | 466 / 464 / 0.9957 |
| `recent` top_m=4 w=8 | 152 / 13 / 0.0855 | **1 994 / 1 989 / 0.9975** |

A leaky recency arm reads as precision 1.000 because it names the target row's own selections and then discards
the resident ones — the definition of "predicting" a miss. Published as-is that table says *a bounded recency
ring matches the probe's recall at 8× fewer reads*, and it would have been the basis for building an engine arm
that stages nothing. The honest figure for the same arm is 152 issues, 13 useful.

### Three precisions that are not three measurements of one thing

§5ah's rank table and the September study both quote precision, and this screen adds a third figure; they do not
line up and the reason is definitional, not a disagreement:

| figure | what it is | tier |
| --- | --- | --- |
| 0.552 | study probe P@1 — precision of the first non-resident pick of the whole list, scored against a residency simulation calibrated to the **ring-on** footer (`issued 43.0` against measured `42.74`) | 64 slots |
| 0.522 | §5ah's *pooled* precision over its per-rank table (ranks 1-8 alone carry 13 489 issues) | 96 slots |
| 0.7541 | this screen at `--top-m 1` — the probe list truncated to 1 **before** the residency filter | 96 slots |

The checkable cross-validation is the count, not the ratio: the screen's top_m=1 issues **2 151** and §5ah's
rank-1 row issues **2 151** on the same trace, so the shipped tool reproduces the throwaway that produced §5ah.
No truncation depth reproduces 0.522 — the curve runs 0.7541 (1), 0.7116 (2), 0.6679 (3), 0.6289 (4), 0.5851
(5), 0.5424 (6), 0.4687 (8), 0.3995 (10) — because that figure pools ranks rather than deepening a plan, and
its per-rank denominators (2 151, 1 999, 1 916, …) are a different partition of the same issues. Two consequences
for how §5ah line 3396 should be read: its "same shape, slightly worse" sentence compares the study's *rank-1*
figure with this campaign's *pooled* one, so it is not a tier comparison, and the 0.522-vs-0.552 direction it
asserts is not supported by anything in that table. Both numbers it names are reproduced; the comparison between
them is the defect, and it is recorded here rather than edited into §5ah, which is a published record.

### Where this leaves the feeder question

§5ah established the size of the prize (a perfect next-token predictor issues 34.65 reads/token, fewer than the
ring's 39.73, and adopts all of them — ≈ +4.6 %). This section establishes, on the shipped tool rather than a
scratch script, that **nothing issue-time feasible is a candidate**: the two engine-buildable strictly-past
rankings offer 16 and 36 worth-staging picks in 18 001 links, and the only ranking with volume is the probe's own
list, which the ring already stages in that order. Combined with §5ah's "rank order is the right staging rule,
measured" — the first non-resident pick is the best pick available in the list — the predictor lever is closed on
precision, not on timing, and it is closed with a guard in the tree so that the next arm proposed against it has
to pass the same three tests.

**The GPU guard, measured rather than assumed — and one prior reading corrected.** §5w's deviation (2) recorded
that the `ioreg` AGX "Device Utilization %" reader "showed 61 % with the machine idle … so it is not an occupancy
instrument on this host". Three samples taken for this section, with the sub-counters, say otherwise: 52 / 71 /
74 %, with `Tiler`, `Renderer` and `Device` agreeing to ≤ 1 point every time, while `WindowServer` holds 43.8 %
CPU and `Qoder Helper (Renderer)` 42.3 %. A latched gauge does not move with three correlated counters and a live
compositor. So the reading §5w took as a broken instrument was real co-tenant GPU work, gate0's 15 % refusal is
correct, and **every remaining timing arm in this campaign — the live MTP pair, #163's wiki row — stays blocked
while this app is running**, which is also why the operator's "the macbook is idle, stop checking" holds for
*my* processes (the `pgrep` guard is empty, nothing of mine runs) but not for the desktop's. `--allow-busy-gpu`
was not passed and nothing in this section comes from a contended GPU.

Deviations.

1. **The recall denominator is 18 001 scored links, not 18 384 layer-slots.** Layer 47 has no *L+1* target, so
   it is absent from every recall figure here, and it is the hungriest layer in the trace: 33 709 misses over all
   48 layers against 32 046 over scored links, i.e. **1 663 misses (4.9 % of all demand) sit on the unscored
   layer**, at 4.34 per position against the scored average of 1.78. Demand here is therefore 1.780 misses per
   link, which is *not* §5ai's 1.834 per layer (that figure is 33 709 ÷ 18 384); the two differ by exactly the
   missing layer.
2. **`recent`'s precision is not measurable at its own issue count.** 1 useful of 16 issues, and 0 of 9 at
   window 16: the quoted 0.0625 and 0.0000 have no error bar worth printing. What is solid is the *count* — 16
   and 9 against the probe's 2 151, a 134× gap that no sampling noise reaches.
3. **The ranking measured is "recency among everything", filtered to nonresident — not a recency ranking of the
   nonresident set.** That alternative (an LRU-style "stage the oldest-seen thing") was not scored, and it would
   carry no next-token information at all; it is a cache-policy idea, not a predictor, and #196's subject is
   predictors.
4. **No clocks and no bytes appear in this section.** The screen is offline and non-mutating by design, so
   nothing here is a tok/s claim, and no §6 or §8 closure is reopened or overturned by it.
5. **One session, one prose prompt, positions 10-392.** Same trace as #194 and #195, so the same ≤ 0.26 %
   denominator skew and the same prompt-shape limit; the code-shaped decode the September study scored is not
   represented, and its 0.554 row is not comparable to this table for the reasons above.
6. **The leaky module is a throwaway, not a shipped arm.** It exists only at `/tmp/leakcheck/` and was deleted
   after this section; nothing in `benchmark/` can score an arm that reads its target row.
7. **The transition prior's own arm is not the engine's.** `rank_transition` accumulates *L → L+1* counts from
   strictly-past positions, which is what a runtime table could learn, but the shipped probe is a trained
   next-token head, so this row is a lower bound on what a learned prior could do — not a measurement of the
   head's alternatives.
8. **Nothing in `sources/` changed.** `894ecfa` touches two files under `benchmark/`, so no engine behaviour
   moves and the golden baseline is **not checked** (docs-and-python only, as in §5af-§5ai). Test inventory: 3
   tests added, 2 pre-existing tests unmodified, both green. Provenance: tree clean at `HEAD 894ecfa`, AC 100 %,
   `memory_pressure -Q` 83 % free, swap 5.49 GiB used, load average 4.45, Swift 6.4 / macOS 27.0.1, `pgrep` guard
   empty before and after, and no model process at any point in this task.
9. **Every `top_m` row in this section was scored under the withdrawn v4.3 issue policy, and §5ak corrects what
   the axis means.** Before `bc4c648` the screen capped the ranking at `top_m` *then* dropped the resident names,
   which is what v4.3 did and not what ships; the shipped ring is handed the whole list and caps the *reads*. So
   the 0.7541-at-`top_m`=1 row is a marginal count of §5ah's rank table, not the engine's issue total, and the
   "2 151 = 2 151" cross-validation in this section proves less than it appears to. §5ak adds the policy axis,
   reproduces the shipped total exactly, and states which figures here survive the correction.

## 5ak. #197 puts the issue policy inside the screen: the shipped filter-then-cap rule reproduces §5ah's live ring to the count, and the depth ladder that reproduction exposes is decided by the engine's own series rather than by the screen (2026-10-16)

§5aj's deviation (9) is this section's premise, and it was checked rather than assumed. The engine hands the ring
the **whole** ranked prediction and lets the ring's own size cap the reads
([RealForwardRunner+DecodeMoE.swift:334-348](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L334)),
and `ExpertPrefetchRing.begin` then computes `wanted = experts.filter { !resident.contains($0) && … }`,
`count = min(wanted.count, free.count)` and stages `wanted.prefix(count)`
([ExpertPrefetchRing.swift:88-98](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L88)).
That is **filter first, cap second**. The screen had been doing the opposite — cap the ranking at `top_m`, then
drop the resident names — which is exactly the v4.3 behaviour the same comment records as the reason v4.3 lost.
One flag name was sweeping two different axes, and every `top_m` figure in §5aj belongs to the withdrawn one.

### What landed

| piece | where | what it does |
| --- | --- | --- |
| `Link.candidate_limit` | [benchmark/tinytitan_prefetch_trace_analysis.py:50](../benchmark/tinytitan_prefetch_trace_analysis.py#L50) | every ranker may expose `max(top_m, len(prediction))` names, so the two policies differ **only** in the order of cap and filter |
| `stage_filter_first` | [:125](../benchmark/tinytitan_prefetch_trace_analysis.py#L125) | the shipped rule — drop resident names, then take the first `top_m` |
| `stage_truncate_first` | [:135](../benchmark/tinytitan_prefetch_trace_analysis.py#L135) | the withdrawn v4.3 rule — take the first `top_m` names, then drop resident |
| `analyze(…, policy="filter")` | [:151-157](../benchmark/tinytitan_prefetch_trace_analysis.py#L151) | unknown names raise; the default is the shipped arm, so a reader gets the engine's behaviour unless they ask for the other one |
| `--policy`, `report["policy"]` | [:196](../benchmark/tinytitan_prefetch_trace_analysis.py#L196), [:218](../benchmark/tinytitan_prefetch_trace_analysis.py#L218) | the axis is named in the artefact, not implied by a flag's default |
| the pinning test | [benchmark/test_tinytitan_prefetch_trace_analysis.py:82](../benchmark/test_tinytitan_prefetch_trace_analysis.py#L82) | one fixture, both policies: the pool's rank-1 name is resident and rank-2 is a real miss, so shipped stages 1 read and 1 useful while v4.3 stages 0 and 0 |

Test-first, in full: the new test was written before the implementation and failed with
`TypeError: analyze() got an unexpected keyword argument 'policy'` — the expected RED, since the axis did not
exist. It is now green along with the five pre-existing tests (6/6, `python3 -m unittest`, serial). The five are
unmodified; their fixtures leave every target's `resident` empty, so the two policies coincide on them and the
shipped default moves none of their assertions. `ruff format` reflowed two lines of *my own new test* (not a
pre-existing file) and the suite was re-run to green after it; `tools/lint.sh python` reports
`ok (ruff 0.16.7, check and format clean, 201 scripts parse under 3.13)`. Commit `bc4c648`.

### Both policies on the #194 ring-off trace, `recorded` ranking, 18 001 scored links

| `top_m` | filter issues | filter useful | filter precision | truncate issues | truncate useful | truncate precision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **15 257** | **7 968** | **0.5223** | **2 151** | **1 622** | **0.7541** |
| 2 | 26 594 | 12 660 | 0.4760 | 4 899 | 3 486 | 0.7116 |
| 3 | 34 356 | 15 300 | 0.4453 | 8 212 | 5 485 | 0.6679 |
| 4 | 39 274 | 16 768 | 0.4269 | 12 064 | 7 587 | 0.6289 |
| 5 | 42 130 | 17 509 | 0.4156 | 16 437 | 9 617 | 0.5851 |
| 6 | 43 737 | 17 863 | 0.4084 | 21 239 | 11 521 | 0.5424 |
| 7 | 44 604 | 18 011 | 0.4038 | 26 548 | 13 411 | 0.5052 |
| 8 | 45 063 | 18 079 | 0.4012 | 32 283 | 15 130 | 0.4687 |
| 9 | 45 277 | 18 106 | 0.3999 | 38 550 | 16 693 | 0.4330 |
| 10 | 45 339 | 18 113 | 0.3995 | 45 339 | 18 113 | 0.3995 |

**The shipped arm is the one the engine already produced.** `filter` at `top_m` = 1 reads 15 257 times and finds
7 968 correct names, and §5ah's offline replica of `ExpertPrefetchRing.begin` — built from the live run, validated
against three engine counters — reports **15 257 issued** and **7 968 correct** on the same trace
(§5ah line 3356), with overall precision 0.522 against this table's 0.5223. Identical to the read. `truncate` at
`top_m` = 1 reads 2 151 times at 0.7541 — which is §5ah's **rank-1 marginal row** (2 151 issues, 0.754 precision,
line 3393), a slice of a partition, not an issue total. That is what §5aj's "2 151 = 2 151" cross-validation
actually matched — one marginal row of the rank table, while the engine's own total, which the shipped tool now
reproduces exactly, went unscored by it for a whole section. The partition reconciles completely: the per-rank
marginals sum to 13 489 over ranks 1-8 (the figure §5aj quotes) and to 15 257 over all ten, which is the shipped
total. The two policies meet at `top_m` = 10 because the recorded pool is exactly ten names wide, so beyond that
the cap cannot bind and only the filter order differs.

### Under the shipped rule `top_m` is ring depth, not pool width

The caller never truncates, so the pool is always the full prediction; the only thing `top_m` can be is
`slotCount`, the number of reads in flight — the knob `ringDepth = max(1, profile.prefetchDepth)` sets in the
build path. So the two rows at `top_m` = 1 are the two arms of the v4.3 withdrawal, and the gap between them is
its size: **7.09× more reads** (15 257 against 2 151) and 4.91× more correct names. §5aj's precision curve
0.7541 → 0.3995 is the withdrawn arm's *precision against pool width*; the shipped arm's *precision against ring
depth* is 0.5223 → 0.3995, and the two are not interchangeable at any depth below ten.

### The depth ladder, priced with this campaign's own money

Counts from the `filter` column, divided by 384 tokens as §5ah does, `2.641 MiB` per read. `adopted-equiv`
multiplies `useful` by the one measured conversion in the record — 19.25 adopted of 20.75 correct at depth 1 =
**0.9277** — because the screen counts a correct-but-late read as adopted and §5ah measured that 1.50 reads/token
never were. `ceiling` is `Σ min(depth, |misses|)`, the reads a *perfect* predictor could use at that depth.

| depth | reads/tok | spec MiB/tok | useful/tok | adopted-equiv | demand MiB avoided | marginal precision of slot *n* | ceiling reads/tok | cum % at route-3 prices |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.73 | 104.9 | 20.75 | 19.25 | 50.8 | 0.5223 (pooled) | 33.67 | **+5.81** |
| 2 | 69.26 | 182.9 | 32.97 | 30.58 | 80.8 | 0.4139 | 55.53 | +8.55 |
| 3 | 89.47 | 236.3 | 39.84 | 36.96 | 97.6 | 0.3402 | 68.78 | +9.70 |
| 4 | 102.28 | 270.1 | 43.67 | 40.51 | 107.0 | 0.2985 | 76.26 | +10.16 |
| 5 | 109.71 | 289.8 | 45.60 | 42.30 | 111.7 | 0.2594 | 80.12 | **+10.29** |
| 6 | 113.90 | 300.8 | 46.52 | 43.16 | 114.0 | 0.2203 | 82.03 | +10.27 |
| 7 | 116.16 | 306.8 | 46.90 | 43.51 | 114.9 | 0.1707 | 82.90 | +10.21 |
| 8 | 117.35 | 309.9 | 47.08 | 43.68 | 115.4 | 0.1482 | 83.26 | +10.17 |
| 9 | 117.91 | 311.4 | 47.15 | 43.75 | 115.5 | 0.1262 | 83.42 | +10.14 |
| 10 | 118.07 | 311.8 | 47.17 | 43.77 | 115.6 | 0.1129 | 83.45 | +10.13 |

The screen's depth-1 row validates; its extrapolation to depth > 1 is **falsified by the engine**, and the reason
is structural rather than a calibration question. Price a saved demand MiB `d` and a spent speculative MiB `c`
from the two arms that measure them:

- **#193's ring-off↔on arm** gives `49.8d − 104.2c = 5.6`. Read at `c = 0` it gives `d = 0.1124 %/MiB`, and then
  the ladder is monotone: depth 10 is the best arm at **+13.0 %**. Take #192's `c = 0.0530` as a flat cost instead
  and the same equation gives `d = 0.2233`. §5ah's route-3 pair splits the difference by modelling the feeder as
  partly *substituting* demand reads (`322.9c + 17.1(d − c) = 17.1`), which solves to `c = 0.042, d = 0.201` —
  and that is the pair in the last column: **cumulative maximum +10.29 % at depth 5**, depth 2 worth +2.74 % more
  than depth 1.
- Both readings are wrong about depth, and wrong in the same direction: **no price pair this campaign can produce
  puts the peak at depth 1.** The route-3 pair peaks at **+10.29 % (depth 5)**, #192's flat cost gives a lower
  curve that still peaks at **+9.6 % (depth 4)**, and the `c = 0` corner is monotone to **+13.0 % (depth 10)**. The
  live ladder in the build path measured **M=1 +12.2 % (hit 78.7 %), M=2 +6.0 % (77.9 %), M=4 −9.8 % (77.1 %)**
  against the same prefetch-off control in the same batch: depth 2 is 6.2 points *worse* than depth 1 and depth 4
  is 22 points worse, while all three priced ladders say both are better — depth 2 by 2.6-3.4 points and depth 4
  by 3.8-6.3.

What the screen cannot see is what the build-path comment names: the drive is saturated while it reads, so a
second speculative read steals service from the demand burst it was meant to anticipate and lands too late to be
adopted — which is why the live *hit* column falls with depth while this table's byte totals rise. The `useful`
counter has no arrival time in it, so depth beyond one is outside what the screen measures. **The predictor lever
is screened by this tool; the depth lever is closed by the engine's own paired series, and this section adds
nothing to it.**

### The one figure that does carry, and one §5aj magnitude that moves

At depth 1 the shipped ring reads 15 257 times where 12 930 links have any demand miss at all: **2 327 reads
(6.06/token, 16.0 MiB/token) are traffic a perfect depth-1 predictor would never generate**, and of the 12 930
links that needed something the single pick is right on 7 968 — **61.6 % of what depth 1 can reach, 24.9 % of all
demand**. That is the prize §5aj named, now on the shipped denominator.

§5aj's other headline needed the same correction and one number in it was off by an order of magnitude. Under the
shipped policy the two engine-buildable strictly-past rankings do not offer "16 and 36 worth-staging picks":
`recent` issues **607** reads (92 useful, precision 0.1516) and `transition` **805** (514 useful, precision
0.6385) at depth 1, and both saturate at 740 and 1 705 reads by depth 10. The qualitative conclusion is untouched
— 805 against the 12 930 links that need a read is 6.2 %, and the probe's own list is the only ranking with
volume — but the quoted magnitudes belonged to the withdrawn arm. §5aj's deviation (9) points here.

Deviations.

1. **Per-token figures divide by 384, per-link figures by 18 001.** Same convention as §5ah (its 39.73 reads/token
   is 15 257 ÷ 384 while the trace holds 383 positions), and it makes every per-token number here ~0.26 % high
   against a per-position reading.
2. **The 0.9277 adopted conversion is measured at depth 1 only.** Applying it to depths 2-10 is *generous* to them:
   lateness must grow with depth for the reason this section is about, so the true cum curve is flatter and lower
   than the last column, and its peak is earlier. The live ladder already says where the peak is; nothing here
   moves it.
3. **Three ring behaviours stay unmodelled**, all named in §5ah: `active` deduplication, the rare busy slot, and
   arrival time. The first two make the depth-1 issue count a 0.35 % upper bound; the third is the whole
   depth argument.
4. **The live M=1/M=2/M=4 series is quoted from the tree, not re-measured here**, and it comes from its own batch:
   per the campaign rule its +12.2 % is not compared with #193's +5.6 %, only with its own M=2 and M=4 arms. The
   "hit" column is quoted as the comment labels it; which counter it names was not established.
5. **`raw_topk_agreement` is a prefix property of the ranking, not of the issued set**, so at equal `top_m` it is
   identical under both policies and none of §5aj's selection-named figures needed re-running.
6. **Only `recorded` is the engine's ranking.** The `filter` rows for `transition` and `recent` are screen-only
   arms of synthetic rankers; the shipped ring never sees them.
7. **No clocks, and nothing in `sources/` moved**: `bc4c648` touches two files under `benchmark/`, so no engine
   behaviour changes and the golden baseline is **not checked** (docs-and-python, as in §5af-§5aj). Provenance:
   code at `HEAD bc4c648` with only this doc file modified when the environment was read, AC 100 %,
   `memory_pressure -Q` 83 % free, swap 5 458 MiB used, load average 2.57, Swift 6.4 / macOS 27.0.1, `pgrep` guard
   empty before and after, no model process at any point, and the trace is the #194 artefact pair under
   `/tmp/trace194/` (both arms' responses byte-identical, `md5` recorded in §5ah).
8. **The link form in this repository's prose is repo-relative, and `file:///` absolute links break CI.** The
   markdown-link check (`.github/workflows/ci.yml:984-1018`) resolves a link relative to the document's own
   directory and skips only `http(s)://`, `#` and `mailto:` — so a `file:///Users/…` target is read as a path
   under `docs/` and reported broken. This section's 9 references and **7 more that already sat on `main`**
   (2 in §5af, 4 in §5ag, 1 in §5ah) were in that form and are now `../sources/…` or `../benchmark/…`; the
   checker reports "Checked 66 local link(s); 0 broken". The pre-existing rows are the AUD-180 class — a doc
   reference no local gate caught — found again, and `tools/lint.sh docs` does not cover link targets.

## 5al. #198 scores the one axis the live depth series never varied: slot 2's marginal precision spans 8.1x across the 47 links and is stable within a session (r = 0.922), so uniform-depth closure does not close it — but the campaign's two price models put a layer-selective arm on opposite sides of zero, and only a live paired run can settle it (2026-10-16)

§5ak closed **depth as a uniform knob**: one number, applied to every layer, decided by the engine's own paired
M=1/M=2/M=4 series. That series never varied the other axis in the same line of the build path — depth is *one
scalar for all 47 links* ([RealForwardRunner+BuildCore.swift:87](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L87),
handed to the ring at `slotCount: ringDepth` line 99) — and the §5z result that survived the whole campaign says
why a uniform number is suspect: **which layer is busy is a property of the model; which expert is busy is a
property of the conversation**, and six cross-trace per-layer budget transfers all beat the shipped uniform split.
So the question #198 asks is narrow and unasked before: is the second read worth staging *at some layers*?

### First finding: a counterfactual depth arm must be scored on the ring-off trace

The screen ran on `on.jsonl` first and returned a depth-1 precision of **0.0377** — not the 0.5223 §5ak
established for the shipped arm. That is not a bug in either section; it is what the two traces record.
In `on.jsonl` the engine's `misses` field is written **after** the ring has already staged its read, so a
counterfactual arm evaluated there is scored against a demand set the shipped prefetcher already consumed:

| scored on | depth | issues | useful | precision | recall | `actual_misses` |
|---|---|---|---|---|---|---|
| `off.jsonl` (ring off) | 1 | 15 257 | 7 968 | **0.5223** | 0.2486 | 32 046 |
| `off.jsonl` | 2 | 26 594 | 12 660 | 0.4760 | 0.3951 | 32 046 |
| `on.jsonl` (ring on) | 1 | 15 257 | **575** | 0.0377 | 0.0233 | 24 653 |
| `on.jsonl` | 2 | 26 594 | 5 267 | 0.1981 | 0.2136 | 24 653 |

The issue counts agree to the count across the two substrates (15 257 and 26 594 — they are properties of the
prediction list, not of what ran), while the useful column collapses. Every number below is on `off.jsonl`, the
same 18 384 observations and 18 001 scored links as §5ak. The on-trace stays the right substrate for *adoption
and lateness* (§5ah/§5ai) and the wrong one for *counterfactual value*; nothing in the repo says which is which,
so it is recorded here.

### Method

No production code changed and nothing new was added to `benchmark/`. The shipped screen
([../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)) was run
at `--top-m` 1, 2, 3, 4, 6, 10 with `--predictor recorded --policy filter` on `off.jsonl`, and the per-link
counters differenced between adjacent depths:

    slot-k marginal precision = (useful@k − useful@k−1) / (predicted_nonresident@k − predicted_nonresident@k−1)

That is the value of the k-th read *alone*, on the shipped issue policy — filter-then-cap means `--top-m` is ring
depth, per §5ak. The differencing, the two price models and the gate tables are arithmetic, done in throwaway
scripts under `/tmp/depthsel/` (`slot2.py`, `gates.py`, `split.py`, `layeraxis.py`, `verify.py`); they are not
proposed as a shipped tool.

### The spread is real and it is 8-10x, across all three slots

| slot | min | p25 | median | p75 | max | ratio |
|---|---|---|---|---|---|---|
| 1 — the shipped read (`prec1`) | 0.085 | 0.410 | 0.567 | 0.667 | 0.836 | **9.8x** |
| 2 — one deeper read (`prec2`) | 0.093 | 0.284 | 0.454 | 0.533 | 0.752 | **8.1x** |
| 3 — a third (`prec3`) | 0.064 | 0.248 | 0.372 | 0.478 | 0.660 | **10.4x** |

All 47 L→L+1 links have a slot-2 read available and none saturates at depth 3 either — the smallest slot-2
increment over the whole trace is 69 issues (slot 3: 19, slot 4: 10), so the deeper ring is never offered a layer
it cannot fill. Slot 2 pooled over the trace is 4 692 useful against 11 337 extra reads = **0.4139** — below the
median of the per-link distribution, which is exactly what a spread of that size means: the uniform arm averages a
good half and a bad half into one number that is neither. The eight most valuable slot-2 links:

| link | `prec2` | extra useful/tok | extra reads/tok | its `prec1` |
|---|---|---|---|---|
| L 6 | 0.752 | +0.55 | +0.73 | 0.836 |
| L21 | 0.708 | +0.59 | +0.83 | 0.809 |
| L37 | 0.704 | +0.49 | +0.70 | 0.820 |
| L 5 | 0.634 | +0.51 | +0.80 | 0.773 |
| L28 | 0.600 | +0.32 | +0.53 | 0.769 |
| L40 | 0.596 | +0.17 | +0.28 | 0.757 |
| L 7 | 0.565 | +0.37 | +0.66 | 0.705 |
| L44 | 0.563 | +0.15 | +0.27 | 0.531 |

Two structural facts come with it. **One ordering, not two:** `prec1` and `prec2` correlate at Pearson **r = 0.851**
across the 47 links (`prec2` vs `prec3` at **0.899**), so a layer that is a good use of a first read is a good use
of a second — one per-layer quality list serves both gates. **Concentration:** ranked by slot-2 useful reads, the
top 11 links carry 1 865 of 4 692 = **39.7 %** and the top 23 carry 3 187 = **67.9 %**, for 2.26-14.83 extra
reads/token instead of uniform depth 2's 29.52.

### And it is stable within the session, which is the part §5z could not show

The obvious objection is that a per-layer statistic from one 384-token capture is that conversation's noise. A
position split tests it: the trace's positions 10-201 and 202-392 are scored separately (this is a clean
out-of-sample test for a *layer-selection* rule, because `recorded` reads only the trace's own prediction list and
the gate is fitted on one half and applied unchanged to the other):

| per-link statistic | half-1 mean | half-2 mean | Pearson r across the 47 links |
|---|---|---|---|
| `prec2` (slot-2 marginal) | 0.428 | 0.418 | **0.922** |
| `prec1` (slot-1) | 0.514 | 0.538 | **0.939** |

A gate fitted on half 1 and never shown half 2 selects 24 / 17 / 9 / 6 links at `prec2` ≥ 0.45 / 0.50 / 0.55 /
0.60 and realises **0.540 / 0.568 / 0.620 / 0.672** on half 2. All four clear the optimistic break-even derived
below (0.4579) and none reaches the pessimistic one (0.8403): what survives the split is the *ranking* — the
layers that aimed well in the first half aim well in the second — while whether that earns anything stays the
price dispute below.

### Two price models for one extra read, and they disagree in sign

The screen cannot say whether 0.546 realised precision is good, because the campaign has two defensible answers.
Both are anchored on the same live pair quoted in §5ak — **M=1 +12.2 %, M=2 +6.0 %**, one batch, one control, so
neither crosses batches.

1. **Optimistic.** Solve a benefit `B` per adopted-equivalent read/token and a cost `K` per speculative
   read/token from the two ladder arms and this trace's own counts: `B = +5.142`, `K = −2.184`. Break-even
   marginal precision `= −K / (B × 0.9277)` = **0.4579**. It reproduces both anchors by construction, and
   **22 of the 47 links clear it**, carrying 2 739 of the 4 692 slot-2 useful reads (58.4 %).
2. **Pessimistic.** Hold the benefit at §5ak's route-3 price (`d = 0.201 %` per demand MiB avoided, times
   stride, times the 0.9277 conversion → 0.4925 % per adopted-equiv read/token) and charge the whole measured
   6.2-point depth-1→2 loss to the 11 337 extra reads → **0.4138 % per extra read/token**. Break-even
   **0.8403** — and **no link reaches it**; the best is 0.752.

The optimistic fit is the one to distrust, and the ladder itself proves it. Its depth-4 falsifier: predicted
`u4 × 0.9277 × B + r4 × K` = **−15.1 %** against the **measured −9.8 %** (and +12.2 / +6.0 reproduced exactly).
The error is specific — a *constant* per-read cost under-prices the third and fourth reads, because each one
contends longer with the demand burst — which is precisely the mechanism §5ak named. It says nothing about
whether a *per-layer* constant is right, but it is the same extrapolation class the selective arm is asking for.

### Arm B — depth 2 only at the links that aim best

Using the whole trace to fit the gate (the live arm would have to ship a list; see the seam below):

| gate on `prec2` | links kept | extra reads/tok | extra MiB/tok | realised precision | optimistic net | pessimistic net |
|---|---|---|---|---|---|---|
| 0.00 — uniform depth 2 | 47 | 29.52 | 78.0 | 0.414 | **−6.20 %** (the ladder, by construction) | −6.20 % |
| 0.30 | 35 | 21.13 | 55.8 | 0.499 | +4.16 % | −3.55 % |
| 0.40 | 30 | 17.67 | 46.7 | 0.527 | +5.82 % | −2.73 % |
| **0.45** | **25** | **14.83** | **39.2** | **0.546** | **+6.21 %** | **−2.15 %** |
| 0.50 | 15 | 8.41 | 22.2 | 0.602 | +5.76 % | −0.99 % |
| 0.55 | 9 | 5.21 | 13.8 | 0.648 | +4.73 % | −0.49 % |
| 0.60 | 5 | 3.60 | 9.5 | 0.684 | +3.88 % | −0.28 % |
| 0.65 | 3 | 2.26 | 6.0 | 0.721 | +2.84 % | −0.13 % |

The gate's shape is the interesting part: at `prec2 ≥ 0.45` it keeps **66.2 % of slot-2's useful reads for 50.2 %
of its bytes**, so the pessimistic model's objection — speculative traffic — is the one thing it demonstrably
reduces, while its own break-even claim (0.8403) is a bar no arm of *any* depth can clear. A third reading,
route-3's own prices, is worth naming because it disagrees with both: under `c = 0.042 / d = 0.201` the selective
arm is +0.55 to +2.85 % against the shipped depth-1 arm, but so is *uniform* depth 2 (+2.74 %), i.e. route-3 says
the layer gate buys 0.11 points and depth itself is free — the model §5ak already rejected on live evidence.

### Arm A — depth 0, i.e. skipping the shipped read at the bad layers

The mirror question (the ring is already spending there; is the *first* read worth it?) was priced the same way and
answers no. Under route-3's prices, keeping only links with `prec1` ≥ threshold:

| gate on `prec1` | links kept | reads/tok | useful/tok | vs shipped |
|---|---|---|---|---|
| 0.20 | 42 | 35.34 | 20.09 | **+0.16 %** |
| 0.30 | 41 | 34.38 | 19.88 | +0.16 % |
| 0.40 | 36 | 30.23 | 18.49 | −0.06 % |
| 0.50 | 30 | 24.66 | 16.06 | −0.64 % |
| 0.60 | 18 | 15.03 | 10.60 | −2.26 % |

Nothing above +0.16 % is worth a code change, and the reason is arithmetic rather than subtle: route-3 prices a
demand MiB avoided about **4.8x** above a speculative MiB spent, so the reads that are lost are worth more than the
traffic that is saved. Under the optimistic pair the same gate models +6.4 to +10.5 % — which is the same
sign-disagreement as arm B, and is a *larger* extrapolation (arm A removes 4-25 reads/token where the ladder only
ever moved between 39.73 and 69.25), so neither model is being believed here either.

### What would settle it, and the seam it needs

One live paired arm: **selective depth 2 on a fixed link list versus the shipped uniform depth 1**, interleaved
A/B/A within rounds, same batch, same control, greedy responses hash-checked for invariance. Its predicted band
is −2.2 % to +6.2 % against a known uniform result of −6.2 % — i.e. the arm is only worth running because the
screen says the layer axis could be worth ~12 points relative to the uniform arm that was measured and rejected.

It is not implementable by flipping a knob, and this is the concrete finding for whoever runs it:

- `ExpertPrefetchRing.begin(model:layer:experts:resident:currentLayer:)`
  ([ExpertPrefetchRing.swift:65-114](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L65)) takes
  **no depth argument** — the depth arrives once, as the ring's `slotCount`, and `wanted.prefix(count)` caps every
  link identically.
- The ring carries **global counters only** — `issuedReads` (:51), `begins` (:58), `reclaimedOps` (:61) and the
  `observed*` gauges (:54-57). There is no per-link state, so an *online* gate that learns which layers aim badly
  has nothing to key on; it would be new bookkeeping, not a filter.
- What is already reachable is the **offline-fitted list**, which is how §5z's per-layer budgets ship: one value
  per layer in the profile, read at build time, handed to `begin` as a per-layer cap. That is the shape the live
  arm needs, and it costs a `depth:` argument (or a profile array) plus the guard that a per-layer depth outside
  `1...topKExperts` is refused the way line 89's scalar check already refuses it.

### Verdict

**The layer axis is genuinely untested, the effect is large (8.1x), stable within a session (r = 0.922, and a
half-trace gate transfers to the other half at 0.540-0.672), and concentration is real (top 23 links = 67.9 % of
slot 2's value). §5ak's closure was of uniform depth and does not automatically extend to it.** It stays a
*question* rather than a lever, because the campaign's two price models put the same arm at +6.21 % and −2.15 %,
the optimistic one fails an out-of-sample check at depth 4, and the pessimistic one's break-even (0.8403) is above
anything any link achieves — a bar that would also condemn the shipped depth-1 arm, which earns +12.2 % live.
No engine change is proposed, no default moved, nothing shipped.

Deviations.

1. **Every figure is from one 384-token capture, one session, one prompt shape.** The half-split test is
   out-of-sample *within* a session; it is not a cross-session portability claim. §5z's +0.957/+0.960/+0.976
   per-layer shape correlation across three captures is the evidence that the layer axis travels, and it is
   quoted, not re-measured here.
2. **`--top-m` differencing assumes the shipped filter-then-cap policy at each depth.** Depth k−1's issue set is a
   prefix of depth k's because both use the same ranking and the same `resident`/`active` filters, and the
   candidate side is checked rather than assumed: no link runs out of names at any depth used here (smallest
   per-link increments 69 / 19 / 10 reads for slots 2 / 3 / 4). What is **not** checked is the runtime free-slot
   pool — the rare busy slot §5ah names — which could make a depth-k read fail to stage where depth k−1's did;
   §5ak's 0.35 % bound is carried here rather than re-derived.
3. **The two price models are two readings of one three-point ladder, not two measurements.** Both are anchored
   on +12.2 % / +6.0 % / −9.8 % quoted from the tree; §5ak deviation 4 still applies — the "hit" column's counter
   was never established. The optimistic fit is used for its break-even and its falsifier, not as a forecast.
4. **`split.py`'s gate-transfer nets (+2.49 to +2.99 %) are not comparable to the arm-B table's nets.** That script
   prices its two anchors as depth-1 = +5.6 % (#193's batch) and depth-2 = −0.6 % (that batch's +12.2 % minus this
   ladder's 6.2 points), which composes across batches against the campaign rule, and it prints its break-even as
   `−K/B` (0.4441) while its own net formula multiplies benefit by 0.9277, so its stated bar is 0.4788 under its
   convention. **Only the price-free parts of that script are carried above**: the two half-to-half correlations
   and the realised precisions 0.540/0.568/0.620/0.672. Its nets are recorded as a direction and are not quoted
   in the verdict or the tables.
5. **The per-link `prec2` used for the gates is a whole-trace statistic, so the arm-B table is in-sample fitted.**
   The half-split rows are the honest estimate of what a shipped list would achieve; the table is the menu of what
   the axis could reach, and the two are deliberately separated.
6. **Adoption is still not in any of this.** `useful` has no arrival time (§5ak's closing sentence), so slot 2's
   extra reads are credited as if they always land in time. The live ladder's falling hit column with depth is
   the counter-evidence, and it is the reason the recommendation is a live arm rather than a default change.
7. **No clocks, and nothing in `sources/` or `benchmark/` moved.** The golden baseline is **not checked**
   (docs-only, as in §5af-§5ak); no model process ran at any point, `pgrep` guard empty before and after.
   Provenance: analysis run against `HEAD 760b62a` with this doc file as the only modification, the trace being
   the #194 artefact pair under `/tmp/trace194/` (`off.jsonl`, mtime 2026-10-16 07:47, unchanged since capture),
   AC power. Environment at the time the screen ran: battery 100 %, `memory_pressure -Q` 83 % free, swap
   5 434 MiB used, load average 2.35, Swift 6.4 / macOS 27.0.1. Re-verified at 13:05 with the machine busier
   (97 % charging, 72 % free, swap 3 968 MiB, load 4.50): **every number above reproduced identically**, because
   the analysis is a deterministic read of a frozen file — which is the check that makes the load readings
   irrelevant here rather than a protocol deviation. The determinism claim is itself tested, not assumed: the
   shipped screen was re-invoked at 13:07 on `off.jsonl` at `--top-m` 1 and 2 and its JSON output is
   **byte-identical** to the 09:55 artefacts every table above was built from (`md5` of the trace
   `67d221e2546baca4b26d1e66aaa360ae`).
8. **The per-token denominator is 384, inherited with all its error from §5ak deviation 1** — every `reads/tok`,
   `MiB/tok` and `useful/tok` figure here is ~0.26 % high against a per-position reading, and the same convention
   makes the two price models' `B`/`K` internally consistent with each other but 0.26 % off a strict per-token
   truth. It cancels in every comparison inside this section, since all arms share the denominator.

## 5am. #200 tests the constant-feeder premise in its other dimension and closes it on arithmetic rather than price: of the 70 043 expert names that survive one token, **70 043 are already resident and 0 are missed**, so a token-ahead feeder has literally nothing to fetch (2026-10-16)

The operator's brief for this campaign is a *shape* of reading, not a knob: "the SSD can read constantly while a
user has an active session and can keep on reading weights even if compute is doing stuff… instead of a
chunk-by-request reader". Two sections have now priced one dimension of that — **how many reads per link**
(§5ak closed uniform depth, §5al found the layer axis undecided). This section tests the other dimension, which
no section has measured: **how far ahead in time a read can be issued at all.** A feeder that runs ahead in time
is the only version of "constant" that could fill the idle windows, so the question is whether the future is
predictable far enough out to be worth reading for.

It is not, and the reason is stronger than a bad precision number. On the same #194 ring-off trace
(18 384 observations, 383 decode positions 10-392, 48 layers), for each horizon `p` the same-layer top-10 route
was compared between position `t` and `t+p`, and both a persistence ranker (this token's score-ordered route
names the read for `t+p`) and a strictly-past recency ranker were scored depth-1 under the shipped filter-then-cap
policy — the same rule §5ak put in the screen, and the same §5al substrate rule applies: ring-off, or the
counterfactual is scored against demand the shipped prefetcher already ate.

### Route churn between tokens is real — and entirely invisible to the cache

| horizon `p` | links | mean top-10 overlap | per-layer mean overlap min/p25/med/p75/max |
|---|---|---|---|
| 1 | 18 336 | **0.382** | 0.075 / 0.276 / 0.399 / 0.466 / 0.654 |
| 2 | 18 288 | 0.299 | 0.072 / 0.204 / 0.302 / 0.365 / 0.566 |
| 3 | 18 240 | 0.274 | 0.072 / 0.189 / 0.277 / 0.323 / 0.521 |
| 4 | 18 192 | 0.257 | 0.075 / 0.178 / 0.254 / 0.306 / 0.489 |

Only 38 % of the ten names a layer demands at `t+1` were also routed at `t` (the overlap is scored against
`t+1`'s set, which is the one a feeder would have to hit), and the layer axis is there again (some layers repeat
65 % of their picks, one repeats 7.5 %). Now the part that decides the lever — of the names that *do* carry over,
what does the cache still hold at the moment they are demanded:

| horizon `p` | carried names | already resident at `t+p` | in `t+p`'s miss set |
|---|---|---|---|
| 1 | 70 043 | **70 043 (1.000)** | **0 (0.000)** |
| 2 | 54 682 | 53 716 (0.982) | 966 (0.018) |

Not a rate near one — exactly one, with no exception in 70 043 observations, on the *pre-plan* residency capture
([RealForwardRunner+Diagnostics.swift:151-153](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift#L151):
"`resident` is captured before cache planning so a later miss reservation cannot make the trace falsely report an
expert as absent"). The complement carries the lesson: **every demand miss at `t+1` is a name that was not routed
at `t`.** So a one-token-ahead feeder would issue reads only for experts that are already hits, and would need to
name the misses out of a set that does not contain them. There is no price model, no adoption discount and no
noise floor that turns that into traffic worth paying for — it is the rare case in this campaign where the screen
settles a lever by itself.

### What the two past-visible rankers get, and what the shipped link gets instead

| horizon `p` | persistence slot-1 issues / useful / precision | persistence slot-2 marginal | recency slot-1 precision |
|---|---|---|---|
| 1 | **0 / 0 / —** (nothing non-resident to stage) | — | 0.0267 |
| 2 | 7 355 / 564 / **0.0767** | 0.0638 | 0.0268 |
| 3 | 8 855 / 607 / 0.0685 | 0.0591 | 0.0308 |
| 4 | 9 391 / 536 / 0.0571 | 0.0521 | 0.0382 |

Against the shipped link — **same token, one layer ahead, `recorded` ranker: 15 257 issues, 7 968 useful,
precision 0.5223, recall 0.2486** (§5ak) — the best cross-token arm is 6.8x worse at depth 1 and reaches 564
useful reads out of 33 709 demand misses in the trace (**1.7 % recall**) while the shipped ring reaches 24.9 %.
The asymmetry explains itself: the pipeline inside one token gives the ring a *causal* input (layer L's route is
known while layer L computes, and layer L+1's demand is one kernel away), whereas a token-ahead feeder's input is
a statistical guess about a route that has not been drawn yet. Depth-1 lookahead in **layers** is the only place
where this model's future is readable.

### The axis this trace cannot answer, named so the next capture fixes it

"Constant while compute runs" has a third reading that is *not* closed here: not more reads per link (§5ak/§5al)
and not future tokens (closed above), but **more slack within the token** — issuing for layer L+2 while layer L
computes, so a read has two layers of compute to land in instead of one. The trace has a field for exactly that,
`next2_layer_prediction`, and in this capture it is **empty in all 18 384 rows**. So the slack axis is unscreened,
and the cheapest untried experiment in the campaign is the one that would screen it: a re-capture with that field
populated (the MTP work in §5ai reads it, so it is writable), which costs one model run and no code.

Deviations.

1. **The overlap and precision figures are route-set arithmetic on one capture** — one prompt shape, 383 decode
   positions, `--ram 12`, 96 slots, no clocks and no model process. The 1.000 residency result is a joint property
   of *this* slot budget and *this* eviction order, not of the route. Measured on the same file: each layer's own
   resident view averages 91.9-95.9 of the 96 slots while admitting only **1.78 demand misses per scored link**
   (32 046 over 18 001; 1.83 per observation over all 18 384), so a token's arrivals are about 2 % of what the
   layer already holds, and the order deciding who is dropped is LFU by accumulated use count
   ([ExpertCacheTypes.swift:176](../sources/TinyTitan/Infrastructure/Streaming/ExpertCacheTypes.swift#L176)) —
   LRU is the configured alternative and was not run, so neither order has been varied here. A tighter budget,
   where those two arrivals are a larger share, is what would break the 1.000; §5r-§5s's budget work is where that
   trade lives. The smallest resident set anywhere in the capture is 28, at positions 10-12 — the cold-start ramp,
   not steady state.
2. **"Every miss at `t+1` was not routed at `t`" is a statement about the route, not about predictors.** A
   transition-table ranker can still name some of those misses — that is §5aj's `transition` arm, which does it at
   horizon 0 (805 issues, 0.6385 precision) and is what #196's leakage guard exists to keep honest. This section
   closes the *persistence* route to them, not every conceivable ranking.
3. **Recency and persistence are ranked here from trace-visible past state only** (`t`'s route, and the strictly
   past window of 8 for recency), matching the shipped screen's rule. Both are scored against `t+p`'s own
   `resident` and `misses`, which are the ring-off cache's — a token-ahead feeder would have changed that state by
   now. The first-order effect is nil at `p=1` because that arm issues zero reads; at `p≥2` it is unmodelled, and
   none of those arms is anywhere near worth modelling.
4. **Two different miss denominators appear in this section and both are right**: 33 709 is the sum over all
   18 384 observations, while §5ak's 32 046 counts only the 18 001 scored `L -> L+1` links (layer 47's demand is
   never a scored target). The recall figures here use the first, §5ak's uses the second.
5. **The `next2_layer_prediction` emptiness is a property of the capture, not of the engine.** Nothing was
   re-run to fill it, because that is a model run and none was available; the claim checked is the count (0 of
   18 384 rows carry a name).
6. **Nothing in `sources/` or `benchmark/` moved.** Docs-only, as in §5af-§5al, so the golden baseline is
   **not checked**. Provenance: `HEAD 3bd3314` with this file the only modification, read at 13:21 on AC power
   (99 % finishing charge), `memory_pressure -Q` 70 % free, swap 3 904 MiB used, load average 3.51, Swift 6.4 /
   macOS 27.0.1, `pgrep` guard empty before and after, no model process at any point. The machine state is
   recorded for completeness, not because it can move a number: every figure here is a deterministic read of the
   same frozen file §5al was built on (`/tmp/trace194/off.jsonl`, md5 `67d221e2546baca4b26d1e66aaa360ae`). The
   arithmetic lives in throwaways under
   `/tmp/horizon/` (`decay.py` for the arms and the mechanism table, `shape.py` and `reshape.py` for the churn
   and resident-view counts in deviation 1), deliberately not added to `benchmark/`: they reuse
   [../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)'s
   `load_trace` and `analyze` rather than duplicating a second screen, and nothing in them is a shipped arm.

## 5an. #201 asks whether the two-layer-slack axis §5am left unscreened is blocked by predictability: a learned hop-2 ranking holds hop-1 precision (ratio 1.019-1.061), and the capture it called for then shows the engine's real probe does not (2026-10-16)

§5am closed feeding *across tokens* and ended on a sentence that needs qualifying here: "Depth-1 lookahead in
**layers** is the only place where this model's future is readable." What §5am actually measured is that a name
carried from one token to the next is already resident by the time it is demanded. It did not measure whether
layer L+2 is readable *from* layer L — it reported that the trace field which would say so, `next2_layer_prediction`,
is empty in all 18 384 rows of the #194 capture. That is the third reading of the operator's "keep reading weights
even if compute is doing stuff": not more reads per link (§5ak), not future tokens (§5am), but **more slack within
the token** — issue for L+2 while layer L computes, and a read gets two layers of compute to land in instead of one.

The `L -> L+2` link turns out not to need the engine's field at all. It is just the pair of observations at
(`t`, `L`) and (`t`, `L+2`), and the frozen ring-off capture has 17 618 of them against 18 001 hop-1 links — the
same 383 positions, 46 layers instead of 47. So the predictability question is screenable offline, on the
substrate §5al requires (ring off) and the policy §5ak put in the screen (filter-then-cap).

### Three arms, and one of them exists to prove the script leaks nothing

All three rank with the shipped `rank_transition`, on tables keyed by layer, and every link is scored before the
position's pairs are folded in — the shipped screen's anti-leakage order
([../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py#L189)),
reproduced rather than trusted:

- **C** hop-1: `L`'s route ranks names for `L+1` through an `L -> L+1` co-occurrence table. This is the decay
  denominator, and it is also the control — the shipped screen scores the same arm independently.
- **A** hop-2, weakest: `L`'s route ranks names for `L+2` through an `L -> L+2` table. No prediction at all, just
  the same learned co-occurrence one layer further out.
- **B** hop-2 through the probe: the engine's own recorded `L+1` prediction (`source.next_layer_prediction`, the
  list the shipped ring actually reads) is composed with the learned `L+1 -> L+2` table, so each name the real
  probe offers is carried one layer further. Because the layer sweep at a position runs ascending, the table B
  reads has only strictly-past positions in it — the current position's `L+1` pairs are folded in after B scores.

| depth | arm C hop-1 | arm A hop-2 | arm B hop-2 | A/C | B/C | shipped `recorded` | A extrapolated | B extrapolated |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6385 (805 / 514) | **0.6506** (601 / 391) | 0.5727 (557 / 319) | **1.019** | 0.897 | 0.5223 (15 257 / 7 968) | **0.5321** | 0.4684 |
| 2 | 0.6160 (1 190 / 733) | **0.6534** (854 / 558) | 0.5649 (793 / 448) | **1.061** | 0.917 | 0.4760 (26 594 / 12 660) | **0.5050** | 0.4366 |
| 4 | 0.6078 (1 545 / 939) | **0.6384** (1 048 / 669) | 0.5567 (997 / 555) | **1.050** | 0.916 | 0.4269 (39 274 / 16 768) | 0.4484 | 0.3911 |

Precision is issues / useful under the shipped policy; the last two columns are the ratio applied to the shipped
`recorded` arm, not a measurement of it. The control passes exactly: arm C reproduces the shipped screen's
`transition` row to the count at all three depths (805 / 514 / 0.6385, 1 190 / 733 / 0.6160, 1 545 / 939 / 0.6078),
so this script's table order and its `candidate_limit` behave like the shipped one's.

**One hop out, precision does not decay — it rises slightly and holds.** The ratio is above 1 at every depth, and
the warm subset (positions ≥ 60, away from the cold-start ramp) says the same thing: 0.6535 against C's 0.6425 at
depth 1, 0.6562 against 0.6185 at depth 2, 0.6414 against 0.6110 at depth 4. That is the §5am result in reverse:
**across tokens the future is unreadable, across layers it is nearly as readable as the next layer is.** What does
decay is recall (0.0128 against 0.0160) and raw top-k agreement (0.0684 against 0.0722), which is expected — the
set two layers out is a different set, so a ranker that is precise about it is precise about fewer of the misses.
**What this does not license is the inference one would most want from it** — that the engine's own probe decays the
same way one hop out. That is a claim about a different ranker, and §5ao measures it at 0.808 rather than 1.019.

### The demand pool is not the binding constraint either

| horizon | scored links | target-side demand misses | per link |
|---|---|---|---|
| 1 (`L -> L+1`) | 18 001 | 32 046 | 1.78 |
| 2 (`L -> L+2`) | 17 618 | 30 649 | 1.74 |

And on this substrate every one of those is genuinely available to an early feeder, by structure rather than by
counting: with `TINYTITAN_PREDICTIVE_PREFETCH=0` nothing else can have staged them, so a name in `L+2`'s miss set
is absent from the moment `L` computes until `L+2` demands it. Two honest limits sit next to that. The misses are
counted against the *target's own* pre-plan resident view, so a live arm running *beside* the shipped 1-layer ring
would see its room shrink by whatever the ring already staged — §5ah measured the ring converting 19.25 of ~39.5
reads a token, and that overlap is not modelled here. One statistic that would have looked good in this section was
measured and then dropped: the share of `L+2`'s misses whose expert id also appears in `L`'s resident view, or in
`L+1`'s route. Both are cross-layer comparisons over an id space, and the cache is keyed by (layer, expert), so
"resident at `L`" says nothing about slot `(L+2, id)`; the 0.190 / 0.182 figures that came out of it are properties
of the id union, not of the feeder's room, and they are recorded here only so the next session does not reuse them.

### What the two price models make of it, and what only a capture can settle

§5al left this campaign with two break-even precisions for one extra read per link: **0.4579** on the optimistic
constant (`B = +5.142`, `K = −2.184` per read, CONV 0.9277) and **0.8403** on the pessimistic one. The hop-2
extrapolation sits above the first at depths 1 and 2 (0.5321, 0.5050) and at depth 4 below both (0.4484); arm B
clears the optimistic line only marginally (0.4684). So the screen said what §5al's arm B said: **the
predictability objection is void, and the price question is still open.** The asymmetry that decided §5am was
absent — a feeder issued at layer L has a causal input either way, because layer L+2's router can be run on L's
residual rather than on a route that has not been drawn yet.

What the screen could not supply was the *arm a shipped engine would actually run*: not a learned co-occurrence
table but the second probe's own ranking, and whether that ranking still aims one layer further out. The flag for
it already exists and needs no code —
[RealForwardRunner.swift:156-160](../sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift#L156)
reads `TINYTITAN_PROBE2_TRACE=1`,
[RealForwardRunner+DecodeLayers.swift:179](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift#L179)
encodes layer L+2's router on `hidden: routedX` (L's residual) into a dedicated buffer, and
[RealForwardRunner+DecodeMoE.swift:80](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L80)
is its **only** consumer: the value reaches the trace line and nowhere else. The #194 capture simply did not set
the flag.

**That run was made in the same session, and it refutes the extrapolation.** The engine's real hop-2 probe scores
0.4223 against the hop-1 probe's 0.5223 — a ratio of 0.808, not 1.019 — and its non-duplicate reads are worth
0.240, below both break-evens. §5an's learned-table arms therefore describe how *a table* behaves one hop out, not
how *the probe* does, and the two disagree in sign. The measurement is §5ao; this section stands as the record of
the screen that came to the wrong conclusion and of the check that caught it.

Deviations.

1. **Both hop-2 arms are learned-table arms on one capture** — one essay prompt, 383 decode positions, `--ram 12`,
   96 slots, ring off, no clocks. Arm A issues 601 reads where the shipped ring issues 15 257, so its precision
   rests on a sample 25x smaller than the number it is being used to discount, and a table arm is only as
   conservative as its sparsity: 601 issues in 17 618 links means it declines to name almost everything.
2. **The extrapolated columns are arithmetic, not measurement, and they were wrong.** They transfer a ratio
   measured between two learned-table arms onto a recorded-probe arm whose ranker is different, which assumes the
   two kinds of ranker decay at the same rate one layer out. §5ao ran the capture that tests the assumption and it
   failed: the measured hop-2 recorded precision is **0.4223**, a ratio of 0.808 to the same probe at hop 1, not
   the 1.019 predicted. The columns above are kept because the mistake is the finding — a learned co-occurrence
   table is not a proxy for the router probe, and the campaign's screens should not treat one as the other.
3. **The control is the reason to believe the order.** Arm C equals the shipped screen's independent `transition`
   row at all three depths to the read, which is what rules out the leakage this script could most easily have
   introduced (a hop-2 table folded in before its own links were scored). No second check on that exists.
4. **Precision is not adoption and not speed.** Nothing in this section measures whether a read issued at `L`
   lands before `L+2` demands it, how much queue it competes with, or what it evicts; §5ah's late pool (7 % of
   discarded reads) is the shape of that question and it was already too small to measure there.
5. **Warm subset uses `WARM_START = 60` positions**, chosen as comfortably past the ramp §5am located at
   positions 10-12; the all/warm pair is reported side by side rather than one being the headline, because the
   difference between them never exceeded 0.006 precision.
6. **Nothing in `sources/` or `benchmark/` moved.** Docs-only, as in §5af-§5am, so the golden baseline is **not
   checked**. Provenance: `HEAD d4189f6` with this file the only modification, `memory_pressure -Q` 72 % free,
   AC power 100 % finishing charge, load average and swap as read at the capture's start below, Swift 6.4 /
   macOS 27.0.1, `pgrep` guard empty before the offline work. Every figure in the tables above is a deterministic
   read of the same frozen file §5al and §5am were built on (`/tmp/trace194/off.jsonl`, md5
   `67d221e2546baca4b26d1e66aaa360ae`). The arithmetic lives in throwaways under `/tmp/slack/` (`hop2.py` for the
   three arms and the extrapolation, `room.py` for the miss-pool counts; its cross-layer "already resident" column
   is the statistic deviation above rejects), deliberately not added to `benchmark/`: they reuse
   [../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)'s
   `load_trace`, `Link`, `rank_transition`, `stage_filter_first` and `analyze` rather than duplicating a second
   screen, and nothing in them is a shipped arm. The capture this section calls for was run in the same session and
   its figures are §5ao; the machine readings above were taken before it, and none of §5an's numbers is a clock.

## 5ao. #202 runs the capture §5an called for and prices the slack-in-layers axis: the engine's real two-layer-ahead probe keeps 0.808 of the one-layer probe's precision (0.4223 against 0.5223), and once its duplicated reads are removed it is at 0.240 — below both of §5al's break-evens (2026-10-16)

§5an ended on an inference it could not test and told the next reader exactly what would test it. This section is
that run. One model run, no code change: the flag
[RealForwardRunner.swift:156-160](../sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift#L156)
already exists, the L+2 router encode at
[RealForwardRunner+DecodeLayers.swift:179](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift#L179)
already runs on layer L's residual, and
[RealForwardRunner+DecodeMoE.swift:80](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L80)
is still the only consumer — the value reaches the trace line and nowhere else, so nothing here is a shipped arm
and nothing here needed a rebuild.

```
TINYTITAN_RUNNER_STATS=1 TINYTITAN_PREFETCH_TRACE=/tmp/trace200/probe2.jsonl \
TINYTITAN_PROBE2_TRACE=1 TINYTITAN_PREDICTIVE_PREFETCH=0 \
  .build/release/TinyTitanCLI --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0 --max-context 262144
# exit 0; 18 384 trace rows; md5 caf3128e9ddc7fdd1e0394bc63b20bc9 (14 701 537 bytes)
```

### Two controls, both of them exact

The field is populated where #194's was empty: **17 618 of 18 384 rows carry a full ten-name
`next2_layer_prediction`**, and the 766 that do not are exactly layers 46 and 47 of each of the 383 positions — the
rows with no `L+2` to predict. Two things had to be checked before any of that could be scored.

1. **Does the field name the layer it claims to?** Mean overlap with the actual top-10 route of the layer it should
   describe versus its neighbour: **5.952 of 10 against layer `L+2`'s route, 0.191 of 10 against `L+1`'s.** The
   alignment is the documented one, not an off-by-one artefact of where the readback lands.
2. **Did turning the probe on change the engine the numbers are measured on?** It did not, and this is the check
   that matters most, because every precision figure below is a claim about cache state. Scored per observation
   against the frozen `/tmp/trace194/off.jsonl` (md5 `67d221e2546baca4b26d1e66aaa360ae`), all four traced fields —
   `experts`, `misses`, `resident`, `next_layer_prediction` — are **identical on all 18 384 rows**, and the run's
   own counters repeat the #194 ring-off arm to the read: `expert_hits=150131`, `expert_misses=35901`,
   `expert_evictions=31293`, `expert_reloads=22986`, `expert_read_mib=94801.1`, `hit_fixup_layers=13304`,
   `io_host_waits=13304`, `rdadvise_calls=33518`, `prefetch_issued_per_token=0.00`. Route identity across 18 384
   observations is also the response-identity evidence §5ah got from an md5 of the text; the greedy decode
   followed the same tokens, so the two captures are the same engine run with one more field written.

### The real probe, one hop out

Same file, same `recorded` ranker, same shipped filter-then-cap policy, `analyze` on the hop-1 link and the hand
score below on the hop-2 link — and the hop-1 arm re-scored **on the new file** reproduces §5ak's figures exactly
(15 257 / 7 968 / 0.5223 / 0.2486), which is what lets the two columns be read as one comparison.

| measure | hop 1 (`L -> L+1`) | hop 2 (`L -> L+2`) | ratio |
|---|---|---|---|
| names of the real top-10 route hit, of 10 | 6.709 | 5.952 | 0.887 |
| slot-1 name present in the route | 0.9193 | 0.8677 | 0.944 |
| rank-ordered (positionwise) matches | 0.1575 | 0.1272 | 0.808 |
| issues / useful under the policy, depth 1 | 15 257 / 7 968 | 15 455 / 6 527 | 1.013 / 0.819 |
| **precision** | **0.5223** | **0.4223** | **0.808** |
| recall of the target's demand misses | 0.2486 | 0.2130 | 0.857 |

Depth 2 and 4 behave the same way they do on hop 1, only lower — 27 444 / 10 240 / 0.3731 (recall 0.3341) and
41 610 / 13 560 / 0.3259 (recall 0.4424), against hop-1's 0.4760 and 0.4269; the warm subset (positions ≥ 60) is
0.4096 / 0.3611 / 0.3142. **§5an's extrapolation was wrong in the direction that mattered.** Its learned-table arms
predicted a ratio of 1.019 and a shipped-arm landing at 0.5321; the probe measured 0.808 and 0.4223. The set
accuracy only falls 11 % (6.709 → 5.952) and slot-1 accuracy only 6 % (0.9193 → 0.8677), so the loss is not mainly
in naming the route — it is in naming route members that are *still absent* when demanded, which is the thing a
feeder has to hit. That is where a co-occurrence table and a router probe part company, and it is why the screen's
answer could not be trusted.

### 42 % of the early reads are reads the engine was already going to make

The two probes are not alternatives — a feeder running the hop-2 probe would run *beside* the shipped ring, and the
shipped ring stages for layer `L+2` one layer later. Both arms therefore name experts in the same (layer, expert)
key space, so the overlap is a real duplication count rather than the cross-layer id union §5an had to reject:

| depth | early issues | also issued by the ring | marginal issues | marginal useful | **marginal precision** | ring's own, same target |
|---|---|---|---|---|---|---|
| 1 | 15 455 | 6 443 (0.417) | 9 012 | 2 160 | **0.2397** | 14 874 issues / 7 811 useful |
| 2 | 27 444 | 12 137 (0.442) | 15 307 | 2 522 | 0.1648 | 25 828 / 12 402 |
| 4 | 41 610 | 18 716 (0.450) | 22 894 | 2 546 | 0.1112 | 37 748 / 16 376 |

Read the depth-1 row as the arm a live engine would run: adding the second probe to the shipped ring costs **9 012
extra reads to remove 2 160 more demand misses — precision 0.240** — while raising total coverage from 7 811 to
9 971 useful reads (+27.7 %) on 14 874 to 23 886 issues (+60.6 %). Against §5al's two price models for one extra
read per link, the break-evens are **0.4579** (optimistic `B = +5.142`, `K = −2.184`, CONV 0.9277) and **0.8403**
(pessimistic). **0.240 is below both, and so is the unadjusted 0.4223** — for the first time in this stretch of the
campaign the two models agree on a sign, because the arm is not near either line.

### The other reading of "constant", and what it does not buy

The 6 443 duplicated reads are the version of the operator's idea that this section cannot price as a cost: those
bytes are spent by the shipped engine anyway, and knowing them one layer sooner moves them out of the demand burst
without adding a byte. That is genuinely the "keep reading while compute works" shape. It is also the one the
campaign already priced from the other side: §5ah found **7 %** of the ring's discarded reads were correct-but-late
and called the pool too small to measure, and this run's own `io_hidden_pct` is 99.89 — the demand reads are almost
entirely hidden already, so a read that arrives earlier than an already-hidden read hides nothing. The slack arm
therefore has no traffic argument and no latency argument left in this configuration. What would reopen either is a
**tighter slot budget** (§5r-§5s's axis, where arrivals are a larger share of what a layer holds and reads stop
being hidden) or the **layer-selective depth** §5al left undecided — not a longer horizon.

*Amended by the §5ax correction:* the `io_hidden_pct` = 99.89 clause is the move §5w forbids and §5ap declines to
make — the counter measures host blocking, not device idle, so "almost entirely hidden already" was never evidence
that arriving earlier buys nothing. The verdict survives without it. The traffic half of the argument never needed
the counter (a slack arm adds no byte, it moves one), and the latency half rests on §5ah's 7 % correct-but-late
share and on the two break-even models agreeing below both lines for the adjusted and unadjusted precision alike.
Against the device ledger the correct basis for the same conclusion is the one §5ax now publishes: the wait is
bytes ÷ a ceiling the burst already runs at 97 % of, so earlier arrival relocates the wait rather than removing it.

Deviations.

1. **One capture, one prompt, one configuration** — 383 decode positions, `--ram 12`, 96 slots, ring off, LFU order,
   greedy. The duplication counts and precisions are deterministic reads of that file, and §5al's r = 0.922
   session-stability result is what makes them worth reading as a shape rather than a one-off; they are still one
   prompt's shape.
2. **Marginal precision is a counterfactual against a ring that was not running.** The substrate rule (§5al) forces
   ring off, so the "also issued by the ring" column is what the shipped ring *would* have staged from the same
   pre-plan `resident` snapshot, not what a live co-running pair would each find still absent. A live arm would see
   the earlier read land, and the ring would then skip it — which moves bytes between arms rather than adding them,
   and is exactly why the marginal column, not the total, is the one compared to the break-evens.
3. **The run's clock is not a magnitude and is not used as one.** It read `decode_tok_s=2.636` against the #194
   ring-off arm's 4.496, with `body_ms` 354.7 against 204.2, `wait_ms` 229.7 against 96.1 and the engine's own
   `router_readback_ms` 0.0401 against 0.0194. That is a cross-batch comparison — the paired-benchmarking rule this
   campaign follows forbids reading it as an effect — and the second router dispatch plus a ten-index host readback
   per layer per token is tracing overhead a shipped feeder would not pay, since the prediction would stay on
   device. What can be said from it is only that the capture is not a speed run, which is what §5an predicted
   before making it. Nothing here prices the probe itself; if a future arm needs that, it is a paired A/B.
4. **The two 0.808s in the probe table are different quantities that happen to agree.** Rank-ordered match rate and
   policy precision both fall by that factor; nothing makes one follow from the other, and the coincidence is not
   evidence.
5. **`next2_layer_prediction` is still trace-only.** This section measures what a feeder *would have aimed at*; it
   does not add a feeder. `sources/` did not move, so the golden baseline is **not checked**.
6. **Nothing in `sources/` or `benchmark/` moved.** Docs-only, as in §5af-§5an, so the golden baseline is **not
   checked**. Provenance: `HEAD d4189f6` with this file the only modification, and the tree's sources unchanged
   since `9c858cd` (04:28:32) against a CLI built at 04:29, so the binary is HEAD's sources. Preconditions read
   once immediately before the run: `pgrep` guard empty, `memory_pressure -Q` 72 % free, AC power 100 % finishing
   charge, 283 GiB free on the data volume, Swift 6.4 / macOS 27.0.1 on this campaign's 24 GB M3 MacBook Pro, one
   model process start to finish. The screen arithmetic lives in `/tmp/slack/probe2.py` (controls, both horizons,
   the duplication count) and `/tmp/slack/hop2.py` (§5an's arms), deliberately not added to `benchmark/`: they reuse
   [../benchmark/tinytitan_prefetch_trace_analysis.py](../benchmark/tinytitan_prefetch_trace_analysis.py)'s
   `load_trace`, `Link`, `stage_filter_first`, `analyze` and `Counters` rather than duplicating a second screen, and
   nothing in them is a shipped arm. The capture is `/tmp/trace200/` and, like every other trace in this campaign,
   is not committed.

## 5ap. #202 runs the paired live arm §5al asked for: the gate-0.45 layer list measures −8.4 % over four interleaved rounds, and the counters name the term the screen had no way to price — the 33 extra reads a token do not add adopted reads, they remove 10.5 of them, exactly one for each demand hit the arm loses (2026-10-16)

§5al's closing requirement was quoted by §5ao and is worth restating exactly, because this section is either its
fulfilment or its refusal:

> One live paired arm: **selective depth 2 on a fixed link list versus the shipped uniform depth 1**, interleaved
> A/B/A within rounds, same batch, same control, greedy responses hash-checked for invariance.

Its predicted band was −2.15 % to +6.21 % against a measured uniform-depth result of −6.2 %. Four rounds ran. Every
one of them is below the band's floor.

### The instrument needed one change first

The seam was already in the tree from `dcebe75` — `TINYTITAN_PREFETCH_LAYER_DEPTH`
([../sources/TinyTitan/Runtime/Configuration/PrefetchDepthProfile.swift:17](../sources/TinyTitan/Runtime/Configuration/PrefetchDepthProfile.swift#L17),
read at [../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift:104](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L104),
applied as `min(freeSlots, depthByTargetLayer[layer] ?? slots.count)` at
[../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift:116](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L116))
— and it is default-off: with the variable unset the parse gets `nil` and every link keeps the shipped depth.

What the sweep could not do was answer the hash half of that sentence. It compared rates without proving the arms
generated the same text, so a rate could have been a rate at different work. `2a06f7b` adds the check: the streamed
completion is reassembled from `delta.content` only — `reasoning_content` is a separate channel and is deliberately
not hashed — at
[../benchmark/tinytitan_knob_sweep.py:347](../benchmark/tinytitan_knob_sweep.py#L347), hashed per arm at `:375`,
and `verdict()` now refuses a page whose arms disagree (`CONTESTED`, status 1) or where an arm streamed nothing
(status 2). 47 tests in [../benchmark/test_knob_sweep_verdict.py](../benchmark/test_knob_sweep_verdict.py),
ruff 0.16.7 clean.

```
SWEEP_TOKENS=384 TINYTITAN_BENCH_RAM_BUDGET=12G \
  python3 benchmark/tinytitan_knob_sweep.py --arms base,layer_depth25,base_again
# arm env: TINYTITAN_PREFETCH_LAYER_DEPTH = the 25-link gate-0.45 list, 119 chars,
#   built at benchmark/tinytitan_knob_sweep.py:70 from §5al's table
# rounds 3-4: exit 0, "response text identical across 3 arms (md5 3c0c3391)"
```

The ring evidence line confirms the arm configured what it claims: `slots=2 links_deep=25 free=1.90` against the
baseline's `slots=1 links_deep=0 free=0.99`, in every round.

### Four interleaved rounds

| round | base tok/s | depth-2-on-25 tok/s | paired delta | drift control | text hash |
|---|---|---|---|---|---|
| 1 | 4.403 | 4.072 | **−7.5 %** | −1.1 % | not taken (pre-`2a06f7b`) |
| 2 | 4.370 | 3.973 | **−9.1 %** | +3.9 % | not taken (pre-`2a06f7b`) |
| 3 | 4.451 | 4.086 | **−8.2 %** | +0.2 % | `3c0c3391`, all three arms |
| 4 | 4.464 | 4.075 | **−8.7 %** | +0.5 % | `3c0c3391`, all three arms |

Mean −8.4 %, spread 1.6 points, four of four negative, and the two hash-verified rounds alone land at −8.2 % and
−8.7 %. The drift control does not explain the ordering: round 2's control got *faster* across the batch (+3.9 %),
which if anything flatters the middle arm, and that round is the most negative. The greedy text is also identical
*between* rounds, so the four rounds are four clocks on one answer rather than four answers.

**§5al's Arm B is refused: the layer gate does not turn depth 2 from a loss into a win.**

### Where the screen was wrong is the useful part

Round 4's per-token counters, the hash-verified one (round 1 agrees to two decimals on every line):

| per token | base | depth 2 on 25 links | delta |
|---|---|---|---|
| prefetch issued | 42.76 | 76.12 | **+33.36** |
| prefetch adopted | 22.80 | 12.35 | **−10.45** |
| adoption ratio | 0.533 | 0.162 | |
| demand hits (384 tok) | 145 313 | 141 298 | −4 015 |
| demand misses (384 tok) | 42 330 | 46 345 | +4 015 |
| demand total | 187 643 | 187 643 | 0 |
| `expert_read_mib` | 111 777.7 | 122 379.8 | +10 602.1 = 27.61/token |
| `io_ms` / `body_ms` | 100.406 / 205.540 | 116.461 / 225.691 | +16.06 / +20.15 |

Read the two bold lines together. 4 015 ÷ 384 = **10.456**, and the adopted column fell by **10.45**. The identity
holds in every round, to the printed precision: round 1 11.5625 against 11.56, round 3 11.1745 against 11.17, and
round 2 7.8411 against 7.84 (scored against its control, for the reason in deviation 2). The demand total is
187 643 in both arms of every round.

So the extra 33 issues per token do not add adopted reads. **They remove them, one for one with the demand hits the
arm loses.** That is §5al deviation 6 answered on the machine it was written about — *"Adoption is still not in any
of this. `useful` has no arrival time … the live ladder's falling hit column with depth is the counter-evidence"* —
and it is why the screen's 0.546 realised precision could never be spent: the screen credits a slot-2 read when its
name is in the route, and charges nothing for the slot-1 read that no longer gets adopted.

### The screen missed in both directions, and the cheap one favours the arm anyway

| quantity | §5al's gate-0.45 screen | measured (rounds 1/3/4) |
|---|---|---|
| extra issues/token | 14.83 | 33.06 / 33.25 / 33.36 |
| extra MiB/token | 39.2 | 30.53 / 29.51 / 27.61 |
| extra useful reads/token | +8.1 | **−11.56 / −11.17 / −10.45** |
| net | −2.15 % … +6.21 % | −7.5 % … −9.1 % |

The arm spends ~30 % *fewer* bytes than forecast while issuing 2.25× the ops: 0.83 MiB moved per extra issue against
the 2.64 MiB a demand miss costs, so most of the extra issues carried little or no read. The likeliest reading is
that a live ring sees its own earlier arrivals in `resident` and drops them, which is exactly what a ring-off trace
cannot show; that sentence is a hypothesis, and the ratio is the measurement. It does not matter for the verdict —
the cheaper-than-predicted traffic still cost 8.4 %, because the loss is not in the bytes, it is in adoption.
Against §5al's two price models, whose break-evens were 0.4579 (optimistic) and 0.8403 (pessimistic), this arm's
whole-config precision is 0.162 and its *marginal* precision is negative. Both models agree on the sign for the
first time in the campaign, and the arm is not near either line.

### What the counters cannot say

`expert_evictions` reads **51 086** and `expert_reloads` **43 662** in every arm of every round, while the hit and
miss columns move by thousands. Reading the increment site explains the shape of the problem: line 75 of
[../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Plan.swift:75](../sources/TinyTitan/Infrastructure/Streaming/PreadExpertStreamer+Plan.swift#L75)
ticks an eviction per demand-miss slot the plan reassigns, so the column describes the demand plan rather than the
cache's total pressure — and a speculative arrival that pushed an adopted expert out would not necessarily appear
in it. The consequence for this page is narrow and it is a refusal, not a mechanism: the displacement channel —
slot pressure or arrival lateness — is **not decided by these counters**, and §5ap does not claim it. What is
established is the one-for-one adoption↔hit identity and the 8.4 % that comes with it.

`io_hidden_pct` is deliberately not quoted. §5w established the counter is unfit for an overlap claim, and the
99.7–99.8 readings in these logs would otherwise read as "the extra traffic was free" — which round 4's `body_ms`
(+20.15 ms against a rate-implied +21.4 ms, agreeing to 6 %) contradicts.

### Verdict

**The layer axis is closed by measurement.** §5ak closed uniform depth; this closes the selective reading of it,
so §5al's finding that the *spread* is real (8.1×, r = 0.922, transfers half-to-half at 0.540-0.672) stands as a
description of the routes and fails as a lever: concentrating a bad read does not make it a good read. Nothing
shipped — the seam stays default-off and no shipped configuration sets it, no runtime default moved, and no
numerics changed.

One reopen is left. §5ao named two conditions that could still make an earlier read pay: the layer-selective depth
(now refused) and a **slot budget tight enough that a staged arrival is a larger share of what a layer holds**. The
second is an interaction and neither single-knob arm answers it, so the cell that turns both knobs at once — 64
slots with the same 25 links deep — is added in `5502ccb` as `slots64_depth25` and is measured next.

Deviations.

1. **Rounds 1 and 2 predate the hash instrument.** They are paired, controlled and same-batch, but they do not
   satisfy §5al's invariance clause. Rounds 3 and 4 do, and they carry the conclusion on their own (−8.2 %,
   −8.7 %); the four-round mean is reported with the two unhashed rounds labelled as such.
2. **Round 2's baseline footer is gone.** The harness writes one log path per arm name
   (`.build/benchmark-logs/sweep_<arm>.log`), so round 3 overwrote it before it was copied. Its rate, hit, io and
   waits cells survive in `/tmp/seam202/pass2.log`; its full counters do not, which is why round 2's adoption
   identity is scored against that round's control arm instead of its baseline. Procedural fix adopted and used
   from round 3 on: copy each round's three logs to `/tmp/seam202/pass<N>/` before starting the next.
3. **A partial log was caught before it was read.** The salvaged `pass2/sweep_base.log` holds one footer at 32
   tokens (`decode_tok_s=4.581`, `hit=0.6406`, `misses=6 715`) — a warm-up, not a measurement — because the
   extractor took `rates[-1]` from a file still being written. It is excluded from every table here. The extractor
   bug (silent `rates[-1]` over a log with one footer) is in `/tmp`, not in `benchmark/`, and is not shipped.
4. **One prompt, one session, one 384-token measured request per arm per round**, greedy, `--ram 12`, 96 slots,
   48 layers, top-k 10 of 512, the shipped 96-slot default as the baseline. Cross-round agreement measures this
   machine's noise floor, not prompt portability; §5al deviation 1 still applies to the gate list itself, which was
   fitted on a whole trace.
5. **The 2.25× issue discrepancy is reported, not explained.** The byte/issue ratio that accompanies it is a
   measurement; the residency-filter reading attached to it is a hypothesis and is labelled as one.
6. **`expert_evictions` and `expert_reloads` carry no information for this claim**, for the reason above. They are
   still printed by the sweep's table; a reader should not treat their invariance as evidence that the cache
   behaved identically.
7. **Provenance.** Rounds 1-2 ran at HEAD `b8bb1fa`, rounds 3-4 at HEAD `2a06f7b`; both differ from current `main`
   only by benchmark python rows. `sources/` was unchanged across all four rounds and across the golden run: every
   seam file was last written by 14:23:53, [../sources/TinyTitanLib/ServerModelSession+Diagnostics.swift](../sources/TinyTitanLib/ServerModelSession+Diagnostics.swift)
   and the rest of the tree were untouched after that, the release binaries on disk are 14:37:28 (CLI) and
   14:37:36 (server), and `find sources -newer` against them is empty — so whatever binary each round used, it
   carries those sources. `tools/golden-baseline.sh --check qwen38-4` at 14:31:39 read
   `ok — output identical to baseline` against that same source state. This section moves no sources; the two
   python commits it cites are ruff-clean with 47 tests, so the golden baseline is **not re-run** for them and is
   **not** needed by them. Preconditions read immediately before the rounds: `pgrep` guard empty,
   `memory_pressure -Q` 73-74 % free, AC power 100 % charged, 281-283 GiB free on the data volume, Swift 6.4 /
   macOS 27.0.1 on this campaign's 24 GB M3 MacBook Pro, one model process at a time, `--ram 12` throughout.
   Logs live in `/tmp/seam202/pass<N>/` and, like every other capture in this campaign, are not committed.


## 5aq. #203 turns both knobs at once, as §5ao asked: at 64 slots depth 2 costs −8.1 % against −12.1 % at 96, so the interaction has the predicted sign and no corner of the 2×2 beats the shipped baseline — one clean round, and two more lost to an occupant (2026-10-16)

§5ap's closing reopen named the one cell that could still make an earlier read pay: *"a slot budget tight
enough that a staged arrival is a larger share of what a layer holds"*. It is an interaction, so neither the
depth series (§5ak, uniform depth at 96 slots) nor the budget series answers it. `5502ccb` adds
`slots64_depth25` — `--expert-cache-slots 64` **and** the same 25-link gate-0.45 depth list — so one batch can
span both knobs:

```
SWEEP_TOKENS=384 TINYTITAN_BENCH_RAM_BUDGET=12G \
  python3 benchmark/tinytitan_knob_sweep.py \
    --arms base,slots_64,layer_depth25,slots64_depth25,base_again
# arm env: benchmark/tinytitan_knob_sweep.py:222-227; every arm's argv carries --ram-budget 12G
```

### Round 1 — the one usable round

| arm | pos | tok/s | vs base | hit | `io_ms` | `body_ms` | issued/pt | adopted/pt | `expert_misses` | hash |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 4.55 | — | 0.7745 | 99.4 | 201.129 | 42.75 | 22.85 | 42 313 | `3c0c3391` |
| slots_64 | 2 | 4.47 | −1.7 % | 0.7741 | 99.1 | 204.825 | 42.72 | 22.64 | 42 392 | `3c0c3391` |
| layer_depth25 | 3 | 4.00 | **−12.1 %** | 0.7552 | 117.0 | 230.720 | 76.03 | 13.43 | 45 927 | `3c0c3391` |
| slots64_depth25 | 4 | 4.11 | **−8.1 %** | 0.7576 | 112.7 | 224.694 | 76.22 | 14.58 | 45 489 | `3c0c3391` |
| base_again | 5 | 4.33 | −4.9 % | 0.7708 | 101.6 | 212.825 | 42.57 | 21.05 | 43 004 | `3c0c3391` |

Harness exit 0, its own drift line reading `4.9% … swallows nothing: no arm claims a win`, and the text hash
identical across all five arms. The ring evidence line again confirms the configuration: `slots=2 links_deep=25
free=1.90` for both depth arms against `slots=1 links_deep=0 free=0.99` for the other three.

Read the cell rather than the four levels:

| pair | arms | paired δ | `body_ms` δ | adopted δ |
|---|---|---|---|---|
| depth 2 at 96 slots | base → layer_depth25 | **−12.1 %** | **+29.59** | −9.42 |
| depth 2 at 64 slots | slots_64 → slots64_depth25 | **−8.1 %** | **+19.87** | −8.06 |
| interaction | — | **+4.0 points** | −9.72 ms | +1.36 |
| 64 slots at depth 1 | base → slots_64 | −1.7 % | +3.70 | −0.21 |
| 64 slots at depth 2 | layer_depth25 → slots64_depth25 | +2.7 % | −6.03 | +1.15 |

**The interaction has §5ao's predicted sign, in the clock and in two counters at once.** Deepening costs
9.7 ms/token less at a tight budget, and the reason is visible: at 96 slots depth 2 destroys 9.42 adopted reads
a token, at 64 slots only 8.06. **And it is still a loss at both budgets** — the shipped corner (96 slots,
uniform depth 1) is the fastest arm in the round, so the cell the reopen asked for is answered as a refusal.

§5ap's adoption↔demand-hit identity re-checks clean at a second budget in a second batch: 45 927 − 42 313 =
3 614 = 9.436 per token against 9.42 adopted lost, and 45 489 − 42 392 = 3 097 = 8.086 against 8.06. The
identity is not an artefact of one arm or one round.

Position is not doing this. The arms run in `ARMS` order
([../benchmark/tinytitan_knob_sweep.py:316](../benchmark/tinytitan_knob_sweep.py#L316)), so the two depth pairs
span exactly two positions each (1→3 and 2→4) and a linear drift term cancels in the *difference of the
differences*. Applying the control's −4.9 % as a linear ramp (−1.23 % per position) moves both levels by about
−2.5 % and leaves the interaction where it is: roughly −9.6 % at 96 slots, −5.6 % at 64.

### Rounds 2 and 3 are withdrawn, and they were withdrawn for an occupant, not for drift

| | round 2 | round 3 |
|---|---|---|
| depth-2 arms | 0.56 and 1.23 tok/s | 3.53 and 3.90 tok/s |
| control `base_again` | 0.08 tok/s (**−98.3 %**) | 4.61 tok/s (**+22.1 %**) |
| harness exit | 1, `INCONCLUSIVE` | 1, `INCONCLUSIVE` |
| `io_ms` range | 99.1-109.1 | 102.2-128.1 |

Those are not arm effects: a 0.08 tok/s control on a configuration that ran 4.55 minutes earlier is the machine
being taken. `ps aux -r` at that moment showed **RobloxPlayerPhysics at 124.9 % CPU**, Time Machine's `backupd`,
and `spotlightknowledged.updater`; load averages 4.74 / 6.54 / 7.08 with 7.6-9.2 GB of swap used. Live timing
stopped there and nothing belonging to anyone else was terminated. Round 3's *ordering* happens to agree with
round 1's (+3.3 % at 64 slots against −6.4 % at 96), but its control rose 22 % across the batch, so its levels
are position, and it is not used for anything.

### Verdict

**The cell §5ao named is refused, with the interaction's direction confirmed at n = 1 clean round.** A tight
budget softens deepening's harm by ~4 points and ~10 ms/token but does not reverse it, so nothing on the
depth axis — uniform (§5ak), layer-selective (§5ap), or interaction (§5aq) — pays at this configuration. No
runtime default moved; the seam stays default-off and no shipped configuration sets it.

What is left in the prefetch family is not deepening at all. §5ap showed the ring issues 42.8 reads a token and
adopts 22.8, so the question with a positive side is the mirror image: **can the 20 discarded reads be picked out
at issue time and not issued?** That is §5ar, and it is priced here rather than run, because the price band
turns out to be the binding uncertainty.

Deviations.

1. **n = 1.** One clean round is a direction, not a magnitude. The 4.0-point interaction would need at least
   three more clean rounds to price, and every one of them needs the machine quiet; §5ap's four-round spread
   (1.6 points) is the best available estimate of what "clean" costs in noise.
2. **Rounds 2-3's logs are gone.** Only round 1's were copied to `/tmp/seam203/logs1/` before the next round
   started; the withdrawn rounds' logs were overwritten in place by the harness, so their arm-level counters
   survive only as far as the printed tables in `/tmp/seam203/round{2,3}.out`. The procedural fix from §5ap
   deviation 2 was applied for round 1 and skipped for the rounds that were rejected on sight of their rates —
   which is defensible only because nothing from those rounds is used.
3. **Round 1's pressure and swap were not recorded** in the form this page usually prints. Its provenance is
   therefore: `pgrep` guard empty at launch, mains power, exit 0, control drift 4.9 %, and the two withdrawn
   rounds' `ps`/swap readings taken ~20 minutes later. The 4.9 % control is the evidence round 1 was usable,
   not a precondition line.
4. **`--expert-cache-slots 64` is a weak lever in this batch**: −1.7 % wall while `expert_hit_rate` moved 0.0004
   and `expert_evictions` read 51 086 in all five arms. That is §5ap's counter finding ("the eviction column
   describes the demand plan, not the cache's pressure") re-confirmed, and it means the 2×2's budget axis is
   doing less than the arm name suggests.
5. **Provenance.** HEAD `5502ccb` (the arm's own commit); `sources/` was last touched by `dcebe75` (14:34:32),
   the release binaries on disk are 14:37:28 (CLI) and 14:37:36 (server), and `find sources -newer
   .build/release/TinyTitanCLI` is empty — so the arms ran the current sources, the same pair §5ap ran. Tree
   clean at the time of writing this section. One model process at a time, `--ram 12` throughout, one prompt,
   greedy, 32 warm tokens then 384 measured per arm.

## 5ar. #204 prices the mirror image of the feeder — withholding the reads the ring throws away — and finds the campaign's first positive-side lever: a six-link depth-0 list that transfers out-of-sample at S/L = 6.44, worth +0.05 % to +0.32 %, and a price band too wide to settle it (2026-10-16)

The operator's premise is that the drive should read constantly rather than burst-by-burst. §5ag found the
engine already does that, §5am found a token-ahead feeder has nothing to fetch, and §5aq closed making it read
*more*. This section takes the remaining direction — read *less*, by not issuing the half of the ring's work
that is discarded — and it answers it before writing any engine code, because the answer is arithmetic.

### The price pair, and why it is a band

Let `x` be the wall a single issued speculative read costs and `y` the wall an adopted read saves. Three paired
arms from three batches give three equations, each in **its own batch's** counters:

| id | arms (same batch) | equation |
|---|---|---|
| E1 | §5ag ring on → off: 39.45→0 issued, 18.86→0 adopted | `18.86 y − 39.45 x = 11.7 ms` |
| E2 | §5ap round 4 base → depth-2-on-25: +33.36 issued, −10.45 adopted | `33.36 x + 10.45 y = 20.15 ms` |
| E3 | §5aq round 1 base → depth-2-on-25: +33.28 issued, −9.42 adopted | `33.28 x + 9.42 y = 29.59 ms` |

E2 and E3 are the **same configuration in two batches** and they disagree by 9.44 ms per token, so no `(x, y)`
satisfies all three: E2 is fit at `y = 1.138`, E3 only at `y = 1.559`, which is outside the physically
admissible range. That range is what anchors the band — an adopted read removes exactly one 2.641 MiB expert
read from the critical path, so `y` is that read's service time, and §5af measured this drive's wall at
3.3-3.4 GB/s:

| `y` (ms/adopted) | source | `x` from E1 | `p* = x/y` | E2 residual | E3 residual | suppression ceiling |
|---|---|---|---|---|---|---|
| 0.788 | one read at 3.35 GB/s, the measured wall | 0.080 | 0.102 | −9.23 | −19.49 | 0.80 % |
| 0.981 | 2.69 GB/s | 0.172 | 0.176 | −4.16 | −14.62 | 1.73 % |
| 1.109 | 2.38 GB/s | 0.233 | 0.211 | −0.77 | −11.38 | 2.34 % |
| 1.301 | 2.03 GB/s, pessimistic service | 0.325 | 0.250 | +4.30 | −6.51 | 3.26 % |

**So the campaign's own arms prove an issued read is not free** — `x > 0`, 0.08-0.33 ms of it, because E1 does
not close otherwise (if `x = 0`, `y = 0.62 ms` and E2's 20.15 ms would have to come from 6.5). And the whole
subtractive family's ceiling — withhold every read that will be discarded, with perfect knowledge — is
`20.59 × x` = **0.8 % to 3.3 % of a 205.5 ms token**. The byte cross-check holds: 19.03-20.59 discarded reads ×
2.641 MiB = 50-54 MiB/token against §5ag's measured 54.4.

A gate is worth running iff it removes more reads than it removes adoptions, in the price pair's ratio:
`S/L > y/x = 1/p*`, i.e. **pays iff `p* > L/S`**. That single inequality is the yardstick for everything below,
and it never needs a point estimate of `x`.

### Screen 1 — the layer index does separate, and it survives split-half

`/tmp/seam204/slot1.py` runs the shipped screen's `analyze()` at `top_m=1` over the frozen ring-off capture
(§5al's input, `/tmp/trace194/off.jsonl`; **not** the `benchmark/expert-cache/route-trace-*.txt.gz` files, which
are a different space-separated format for the replay sims). Calibration: 39.84 issued and 20.80 needed reads
per token, against the capture's own batch at 39.45 issued / 19.25 adopted (§5ah's replica on the same trace).
Per-link slot-1 precision spans **0.085 to 0.836 — 9.8×, median 0.567**.

Six links fall below the whole band's top (`p* ≤ 0.288`): **14, 22, 30, 31, 38, 46** (precisions 0.186, 0.217,
0.085, 0.179, 0.130, 0.181). The next link up is 15 at 0.309, so the list is stable for any `p*` in
[0.22, 0.30] — most of the band. The env string is 26 characters:

```
TINYTITAN_PREFETCH_LAYER_DEPTH=14=0,22=0,30=0,31=0,38=0,46=0
```

Fitting the gate on the first 191 positions and scoring it on the last 192 (`/tmp/seam204/transfer.py`) is the
test §5al ran for depth 2 and this rule passes it better:

| quantity | in-sample (whole trace) | fitted on half A, scored on half B |
|---|---|---|
| per-link precision correlation | — | **r = 0.937** (§5al's depth-2 transfer: 0.540-0.672) |
| gate list | 14, 22, 30, 31, 38, 46 | **the same six links** |
| suppressed reads / adopted lost per token | 5.37 / 0.87 = **S/L 6.14** | 5.30 / 0.82 = **S/L 6.44** |
| pays iff | `p* > 0.163` | `p* > 0.155` |

At the band's rows that is **+0.053 % (p\*=0.176) to +0.318 % (p\*=0.250)** of decode, and a small loss
(−0.11 %) only at the very bottom row, where the drive serves a read at its measured wall. First positive-side
prediction of the campaign — and its magnitude is smaller than §5ap's four-round control spread.

### Screen 2 — no read-level gate pays, and that is the closure

`next_weights` is the field the trace already carries and every earlier section ignored: the probe's own weight
vector for the L+1 candidate list, index-aligned with `next_layer_prediction` at
[../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift:287](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift#L287).
The engine is computing layer L while the ring stages for L+1, so these numbers are in hand at issue time — a
gate on them carries no future information. `/tmp/seam204/conf.py` reconstructs the shipped depth-1 issue set
exactly (the first non-resident name in prediction order, filter-then-cap), labels each read *needed*, and
screens six features:

| feature | precision, worst decile → best | best threshold (either direction) | `S/L` | pays iff |
|---|---|---|---|---|
| `own` — the issued name's weight | 0.298 → 0.851 (monotone) | drop below 0.049 | 3.46 | `p* > 0.289` |
| `rel` — own ÷ top-1 | 0.357 → 0.857 | drop below 0.195 | 2.87 | `p* > 0.348` |
| `rank` — position in the prediction | 0.245 → 0.754 | drop at rank ≥ 8 | 4.53 | `p* > 0.221` |
| `top1` — head of the list | 0.403 → 0.489 | drop below 0.135 | 2.72 | `p* > 0.368` |
| `norm_entropy` | 0.351 → 0.592 | drop at the top decile | 2.87 | `p* > 0.349` |
| `spread` — w[0] − w[9] | 0.367 → 0.488 | drop below 0.059 | 2.83 | `p* > 0.353` |
| **`link` — the layer index** | 0.085 → 0.836 | six-link fitted table | **6.44** | **`p* > 0.155`** |

Read the two columns that matter together. Routing confidence **is** a real signal — the issued name's own weight
drives precision from 0.298 to 0.851, a 2.9× spread the layer index cannot match — and it is still not enough,
because **even the worst decile of the best feature sits above every break-even in the band**. The ring's mean
precision (0.485 adopted, 0.522 needed) is roughly 2× break-even, so there is no pool of reads cheap enough to
be worth dropping: the best confidence gate reaches `S/L = 3.46`, which needs `p* = 0.289`, above the band's
ceiling. Only `rank` (needs 0.221) enters the top fifth of the band at all, and it does so by suppressing 2.11
reads a token to lose 0.47 — a fifth of what the layer table removes, at a worse ratio.

**So the wasted reads are not pickable at read granularity, and the layer table beats everything.** That is the
closure: the subtraction lever exists, is positive, and is capped at the +0.05…+0.32 % that screen 1 names.

### Why the seam is not built

Expressing the six-link list needs `depth == 0`, which today is a refusal at
[../sources/TinyTitan/Runtime/Configuration/PrefetchDepthProfile.swift:43](../sources/TinyTitan/Runtime/Configuration/PrefetchDepthProfile.swift#L43)
and an existing test assertion at
[../tests/TinyTitan/Runtime/ExpertPrefetchDepthTests.swift:40](../tests/TinyTitan/Runtime/ExpertPrefetchDepthTests.swift#L40)
(`"3=0"` is in the refused list). Everything downstream already handles it: `staged()` returns an empty list
and `begin` returns before issuing
[../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift:114-120](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L114-L120),
and `slotCount: max(ringDepth, depthByTargetLayer.values.max() ?? ringDepth)` at
[../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift:114](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+BuildCore.swift#L114)
leaves a pure-suppression list at the shipped one-slot ring, so the arm would allocate exactly what the baseline
allocates. It is a one-line change plus inverting that assertion plus a summary line that must not call a
suppression list `links_deep`
([../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift:169](../sources/TinyTitan/Runtime/Inference/ExpertPrefetchRing.swift#L169)).

It is not built because the arm it exists to measure cannot be decided. At +0.05…+0.32 % against §5ap's observed
paired spread (1.6 points over four rounds, sd ≈ 0.7), resolving the sign needs on the order of **30-60 clean
rounds**, and the machine is not clean — §5aq lost two rounds in twenty minutes to an occupant. Spending an hour
of live time to measure a number that the price band already brackets below the noise floor would be the same
mistake as the arms this campaign refused for being too small.

### Verdict

**The subtractive feeder is closed at read granularity by measurement, and at layer granularity it is closed as
unresolvable rather than as a loss.** The campaign's answer to the constant-feeder premise is now complete in
all three directions it can take: read ahead of demand (§5am — nothing to fetch), read more per link (§5ak,
§5ap, §5aq — a loss at every budget and every selectivity), read less (§5ar — at most +0.3 %, unmeasurable).
What remains is the premise itself, and it is worth restating with the numbers attached: the drive **is**
reading constantly during decode, the ring earns 5.6 % by doing it (§5ag), and each of those continuous reads
still costs 0.08-0.33 ms of wall, because the price is service time on a queue the demand path shares, not idle
waiting. Nothing shipped, no default moved, no numerics touched; `next_weights` remains unread by any shipped
policy.

The one reopen this section earns is an instrument, not a lever: **two depth arms on one link list in one batch**
(depth 2 and depth 3 on the 25 links, against base) give two equations that share a drift term, so `x` and `y`
become identifiable *within* a batch and the band collapses to a point. If the collapsed `p*` lands above 0.155,
the six-link depth-0 arm is worth building and running; if it lands below, §5ar's table above is confirmed and
the family is closed for good. That costs a few rounds, not thirty.

Deviations.

1. **`x` and `y` are assumed batch-invariant.** E1, E2 and E3 come from three batches, and E2 versus E3 shows
   the same arm disagreeing by 9.44 ms/token between two of them. The band is the honest form of that fact; a
   point estimate would not be. Every conclusion above is checked against the whole band, and the read-level
   closure holds across all of it.
2. **The screen labels reads *needed*, not *adopted*.** On a ring-off capture, `misses` is the set a read would
   have served; §5ah's replica on this same trace finds 20.80 needed against 19.25 adopted, so the label
   over-counts value by ~7 %. That makes every suppression gain quoted here a slight *under*-estimate (about
   0.006 ms/token at the layer gate) and, in screen 2, makes each feature's `S/L` a slight under-estimate — it
   does not move any row across the threshold, and the `own`-weight gate would need `p* > 0.270` even with the
   label corrected.
3. **The first run of this screen was wrong and is discarded.** It indexed `rank` by the number of reads already
   issued rather than by position in the prediction list, which made `rank ≡ 0`, `own ≡ top1` and `rel ≡ 1`, and
   its cut scan tested only "suppress low values" — the wrong direction for `rank` and `entropy`. Both fixed;
   the tables above come from the corrected scan (`/tmp/seam204/conf_out5.txt`). The earlier
   "every decile is above break-even, no cut pays anywhere" reading was produced by that broken run and the
   corrected one is reported instead — it separates real signal from a threshold artefact and leaves the
   conclusion narrower, not broader.
4. **The six-link gate was fitted on this trace.** Split-half says the list and its ratio transfer (deviation
   1 of §5al's discipline), but both halves are one prompt, one session, one model, and links 30/31/38/46 sit at
   0.085-0.181 with `actual_miss_recall` 0.12-0.21, so the choice is a property of these routes, not of a
   principle. A live arm must re-fit on a fresh capture or carry the transfer test as its own premise.
5. **The ceiling row uses 20.59 discarded reads/token** (E1's batch) rather than the trace's 19.03 or §5ap's
   19.90; the three differ by ≤ 8 % and the ceiling inherits that spread.
6. **Nothing was measured live in this section.** Every number is arithmetic on arms already published in §5ag,
   §5ap and §5aq, plus the frozen trace. The throwaways are `/tmp/seam204/{slot1,conf,price,transfer}.py`, are
   not proposed as tools, and are not committed; `next_weights` is still read by no shipped code.

## 5as. #205 retires the instrument §5ar named before spending a live round on it — two depth arms sit 1.7-4.5° apart in the (issued, −adopted) plane and price nothing, while the pair that does identify `x` and `y` (ring-off × depth-2, 41.4° apart) has been in the sweep registry since the first pass and has never shared a batch (2026-10-16)

§5ar's reopen was an instrument, not a lever:

> **two depth arms on one link list in one batch** (depth 2 and depth 3 on the 25 links, against base) give two
> equations that share a drift term, so `x` and `y` become identifiable *within* a batch and the band collapses
> to a point.

`2327dfd` builds exactly that: `LAYER_DEPTH325` on the byte-identical 25-link list (119 characters both, so the
lists are the one §5ap ran), pinned by two new assertions in `benchmark/test_knob_sweep_verdict.py` (47 → 49
tests, ruff clean). This section does not run it. It shows the instrument cannot answer the question it was
built for, using arms the campaign has already measured, and names the one that can.

### Why two depth arms price nothing

Write each paired arm as a row `(Δissued, −Δadopted)` acting on the price vector `(x, y)`, so
`Δms = x·Δissued − y·Δadopted`. Two arms identify `(x, y)` iff their rows are not nearly parallel: the condition
number is `1/sin θ`, where θ is the angle between them. The admissible `y` is 0.788-1.301 ms (§5ar's service
time of one 2.641 MiB read). The campaign owns a **three-depth, single-batch series with issued and adopted
counters on every row** — `docs/qwen38-prefetch-predictor-study.md:72-77`, the September per-expert ladder — so
the depth pair can be *solved* here rather than guessed at:

| design | rows `(Δissued, −Δadopted)` | θ | det | `1/sin θ` | solve | ±1 ms in each `Δms` moves `y` | usable? |
|---|---|---:|---:|---:|---|---|---|
| September 1→2 vs 1→3 slots | (+33.5, −3.2), (+60.2, −4.0) | **1.7°** | 58.6 | 34.6 | `x` 0.991, `y` **9.467** | 7.87-11.06 ms | no — `y` is 7.3× the band top |
| September 1→2 vs 2→3 slots | (+33.5, −3.2), (+26.7, −0.8) | 3.7° | 58.6 | 15.3 | the same three points, so the same answer | 8.44-10.49 ms | no |
| live depth 2 vs depth 3 (synthesised) | (+33.28, +9.42), (+55.28, +13.02) | 2.6° | −87.4 | 22.5 | returns `x` 0.270, `y` 1.185 only at exact `Δms` | 0.17-2.20 ms | no — `p*` spans 0.006-3.082 |
| **ring-off × depth-2, one batch** | (+39.45, −18.86), (+33.28, +9.42) | **41.4°** | **999.3** | **1.5** | `x` 0.270-0.448, `y` 1.185-1.558 | ±0.07 ms | **yes** |

The mechanism is structural, not a calibration accident: **one knob cannot move two counters in two different
directions.** Deepening the ring raises issued reads and moves adoption by whatever the marginal slot's
precision happens to be — in the September series +3.2 then +0.8 reads, in §5ap/§5aq's live arms *minus* 9.4 —
so every depth arm lands on nearly the same ray, and the ill-determined direction of the design is the price
*level*. The September solve proves it: residuals −0.000/+0.000 ms, so the two equations are satisfied exactly,
and the answer is `y` = 9.467 ms — 7.3× the physical maximum, because an adopted read removes one 2.641 MiB
service and nothing else. A design that returns an impossible level while fitting its data perfectly does not
have a calibration problem; it has no information.

The third row is the live pair §5ar asked for, so the retirement is not tied to September's operating point. It
is a **synthesis**: depth 3's row is generated *from prices inside the band* (`x` = 0.270, `y` = 1.185, the
mid-band pair below) and then read back with the batch's own noise. Recovery at exact `Δms` is trivially right;
at ±1 ms it returns `y` anywhere in 0.17-2.20 ms and `p*` anywhere in **0.006-3.082** — i.e. the same instrument
says the suppression family is worthless or worth 3× its own ceiling, depending on a millisecond of wall. Three
projected ladder shapes give θ = 2.1°, 2.6° and 2.8° (condition 20-28), so no plausible depth-3 outcome fixes
that. And the noise is real, not assumed: that batch's own round-to-round spreads are 2.82, 10.24 and 0.79
ms/token per arm, against a `Δms` of 2.91 and 21.80 ms to resolve.

One nuance, because it is the kind of trap this campaign has walked into before. The September solve's `p*`
(*ratio*) is tight — 0.101-0.110 across the ±1 ms corners — and `p*` is all the six-link verdict needs, so the
depth pair looks salvaged. It is not: that tightness is the shadow of an inadmissible level. The ambiguity
direction for near-parallel rows moves `x` and `y` together, so the ratio survives only because the solve landed
at `y` ≈ 9.5 ms; the same computation on live-regime rows, where the adopted coefficient flips sign, gives the
0.006-3.082 swing above. **A ratio read off an ill-conditioned level is not evidence**, and §5ar's verdict must
not be closed by it.

### The instrument that works is already in the registry

`prefetch_off` — `TINYTITAN_PREDICTIVE_PREFETCH=0` at
[../benchmark/tinytitan_knob_sweep.py:139](../benchmark/tinytitan_knob_sweep.py#L139) — has been in the
registry since the first pass and was run by §5ag, but never in the same batch as a depth arm. It is the arm
that moves adopted the *other* way: switching the ring off removes 39.45 issued reads **and** 18.86 adopted ones
while costing 11.7 ms/token, whereas deepening adds issued reads while destroying adoptions. Opposite-signed
`Δadopted` at 41.4° is what makes the level identifiable: condition 1.5, and ±1 ms/token of wall — the
resolution a 384-token paired round can reach — moves `y` by 0.07 ms, one-seventh of the band.

```
SWEEP_TOKENS=384 TINYTITAN_BENCH_RAM_BUDGET=12G \
  python3 benchmark/tinytitan_knob_sweep.py \
    --arms base,prefetch_off,layer_depth25,layer_depth325,base_again
```

Four paired arms plus the drift control, in `ARMS` order, so `base` at position 1 and `base_again` at 5 bound
the ramp the way §5aq's did. The two equations are `prefetch_off → base` and `base → layer_depth25`, and
`layer_depth325` is no longer an equation — it is the **out-of-sample test** (below).

That arm is the whole instrument, so two ways it silently stops being one are now pinned by tests (49 → 51):
`prefetch_off`'s value must stay `"0"`, and no arm may set a key that no `.swift` source reads any more — the
failure mode this file's own comment on the removed knobs names, and the one that would show up as a clean
base-vs-base pair fitting any equation it is asked. Both are pins on existing behaviour, so they pass the moment
they were written; proven non-hollow by mutation (flipping the value and truncating a key each fail, and each
fails with a message that names the offender rather than the 3.5 MB corpus it searched).

Composing the same two rows from the two batches that already measured them is not a measurement, but it shows
what an in-batch answer would look like and where it would land:

| price pair | how obtained | `x` ms/issued | `y` ms/adopted | `p*` | §5ar's six-link gate | subtractive ceiling |
|---|---|---:|---:|---:|---:|---:|
| solved from a depth pair | September 1→2 vs 1→3 | 0.991 | 9.467 *(7.3× band)* | 0.105 | −1.22 % | 9.9 % |
| `y` anchored, `x` from the 1→2 step | `y` = 0.788 | 0.162 | 0.788 | 0.206 | +0.104 % | 1.6 % |
| `y` anchored, `x` from the 1→2 step | `y` = 1.301 | 0.211 | 1.301 | 0.162 | +0.025 % | 2.1 % |
| rows from §5ag × §5ap (two batches) | composed | 0.270 | 1.185 | 0.228 | +0.224 % | 2.7 % |
| rows from §5ag × §5aq (two batches) | composed | 0.448 | 1.558 *(above band)* | 0.288 | +0.535 % | 4.5 % |

*Added by §5at, which re-derived these rows: the last two solve the depth step's Δms from `body_ms` (29.591 ms).
On the wall-clock Δms (30.220 ms) the same two rows give `x` 0.460, `y` 1.584, `p*` 0.290 — still above the band,
so the conclusion this section draws does not depend on the choice, but the two prices do, and the table did not
say which column it had read.*

Gate column is `x·5.30 − y·0.82` over a 205.5 ms token, the out-of-sample S/L = 6.44 of §5ar's screen 1, which
pays iff `p* > 0.155`. **The instrument's own falsifier is the interesting part:** if the in-batch trio returns
`y` inside 0.788-1.301, the two-price linear model is confirmed and `p*` is known to ±0.02, which settles the
six-link gate's sign for good; if it returns `y` outside the band — as the §5ag×§5aq composition does — then the
model itself is refuted, because an issued read's cost is not additive in the number issued (the queue-
saturation channel the predictor study named: 1.2 ms cold per read idle against 3.4-6.5 ms with four or more
outstanding). Refuting the price model is the larger finding, and it is the one a depth pair can never produce.

### What the ladder economics say without any identification

The series cannot price `x` and `y`, and it does close depth on the one quantity that needs no price model — the
wall spent per adopted read gained, against a value that is physically capped at 1.30 ms:

| step | Δissued | Δadopted | marginal precision | Δms | ms per extra read | **ms per adopted gained** |
|---|---:|---:|---:|---:|---:|---:|
| 1 → 2 slots | +33.5 | +3.2 | 0.096 | +2.91 | 0.087 | **0.91** |
| 1 → 3 slots | +60.2 | +4.0 | 0.066 | +21.80 | 0.362 | **5.45** |
| 2 → 3 slots | +26.7 | +0.8 | 0.030 | +18.89 | 0.707 | **23.61** |

The last column is the ladder's shape expressed as a price: the first extra read is bought at about a service
time (which is why depth 2 is only −3.3 %), and the third costs 5-18× what an adoption is worth. `wait_ms`
rises 83.0 → 86.8 → 85.4 → 91.1 along the same series, so the demand queue is the channel doing the charging.
Depth is closed by §5ak, §5ap and §5aq on the clock, by this table on the price mechanics, and by the condition
number on any further pricing.

### Why nothing ran live

At 18:39 the machine carried **RobloxPlayer at 143.7 % CPU** (up since 17:52), a Chrome renderer at 92.8 %,
WindowServer at 41.8 %, load 6.13 / 4.79 / 4.82 and 5.99 of 7.17 GB of swap used, on mains power at 100 % with
no model process running. The trio's entire requirement is ±1 ms/token resolution, and §5aq's rounds 2-3 are
this hour's own evidence of what an occupant does to it: a 0.08 tok/s control against a 4.55 tok/s one measured
minutes earlier. Running the batch now would spend ~40 minutes to produce a number the condition table already
condemns. Nothing belonging to anyone else was terminated.

### Verdict

**§5ar's named instrument is retired by its geometry, and the reopen passes to an arm selection that needs no
new code**: `base,prefetch_off,layer_depth25` in one batch on a quiet machine identifies `x` and `y` at
condition 1.5, decides the six-link gate's sign, and falsifies the two-price model if it cannot hold. No runtime
default moved, no engine code changed, no shipped configuration sets either knob.

The depth-3 arm stays in the registry, and its committed comment is corrected in this section's commit: it was
added as "the second equation" and it is not one. Its real job is the out-of-sample check — predict
`Δms(depth 3) = x·Δi₃ − y·Δa₃` from the trio's solved prices and compare it with the measured arm. One
prediction, one measurement, same batch: if the linear model holds, the third point falls where the first two
put it; if it does not, the non-additivity is the finding. That is a use the depth pair cannot make of itself.

Deviations.

1. **The solved rows are September's, and September is not this tier.** `docs/qwen38-prefetch-predictor-study.md`
   ran at `--ram-budget 8G` (the old meaning: 64 slots), 256-token prose, two interleaved rounds per arm, and its
   per-expert-operation arms no longer exist — `TINYTITAN_PREFETCH_PER_EXPERT` was added by `45ea73b` and removed
   by `3eb11cf`, so that series cannot be re-run. Its rates are means of two rounds and its table carries no
   `body_ms`. What transfers is the **condition number**, which is a property of row angles, not of levels; what
   does not transfer is any `x`, `y` or `p*` taken from it.
2. **The third and fourth rows of the geometry table are syntheses.** The projected depth-3 row is generated from
   mid-band prices and read back through the batch's noise, so it demonstrates the design's sensitivity, not the
   engine's behaviour. The composed `x`/`y` pairs join §5ag's batch to §5ap's or §5aq's, which is exactly the
   cross-batch move §5ar deviation 1 forbids: note that the two compositions disagree with each other (0.228 vs
   0.288) and one lands outside the band. They are bracketed candidates for what the in-batch trio will find,
   nothing more.
3. **The campaign's own baseline moves by 10 % between batches** — §5ag's ring-on ran 5.079 tok/s, §5aq's `base`
   ran 4.55 at the same nominal configuration. So a composition across those batches carries a drift term far
   larger than the 1 ms it is trying to resolve, which is a second, independent reason the trio must be one batch.
4. **`p* = 0.105 from the September solve would "close" the six-link gate as a −1.22 % loss, and this section
   refuses to accept that** on the grounds it states above (a ratio read off an inadmissible level). The refusal
   is a judgement, not a measurement, and the trio is what adjudicates it.
5. **The gate re-score inherits §5ar's *needed*-not-*adopted* label** (its deviation 2), so each gain column is a
   slight under-estimate, ~0.006 ms/token at the layer gate.
6. **The depth-3 arm was built test-first on a premise this section then refuted, and the test file carries the
   seam of that.** Full inventory: in `benchmark/test_knob_sweep_verdict.py` the two `DepthArmPair` *assertions*
   are untouched — what changed is their docstring, which claimed two depth arms "identify both prices" and now
   says they cannot and states what the pair is for; a new `PricePair` class adds the two registry pins (49 → 51
   tests, both passing on arrival and mutation-checked). In `benchmark/tinytitan_knob_sweep.py` nothing but
   comments and the `layer_depth325` purpose string changed. No test was edited to reach green, and no run this
   section reports was withdrawn, because no section-#205 measurement exists yet. The throwaways are `/tmp/seam205/{identify,trio}.py`, not proposed as tools and not
   committed.
7. **Provenance.** §5aq/§5ar landed at `31b8f61`, and the depth-3 arm at `2327dfd`; both are on `main` before
   this section, and every number above comes from commits already on `main`, so nothing is published from a
   dirty tree. This section, the sweep's rewritten comments and the two `PricePair` pins are the commit that
   carries them. No model process was started here.
8. **Three `tinytitan_knob_sweep.py` line pointers in earlier sections were stale and are repointed here** —
   `:314`→`:337` and `:342`→`:365` (§5ap, the stream reassembly and its hash), `:283`→`:306` (§5aq, the
   `ARMS`-order return). Six lines of comment were all that moved them; the constructs are unchanged, and the
   two markdown links are what CI's link check reads, so nothing else re-points them when the file shifts again.
   **That last clause was false within one commit**: §5at added ten lines above all three and they moved again
   (`:337`→`:347`, `:365`→`:375`, `:306`→`:316`, `:139` unchanged). Nothing guards a `#LNN` fragment — the link
   check strips it — so each shift is found by hand or not at all.

## 5at. #206's instrument could not print its own inputs — `COUNTERS` carried neither number §5as's price solve reads, so the named batch would have run 40 minutes and handed the identification back to a person with a log file; the seam closed, and §5as's geometry re-derived from published counters rather than remembered (2026-10-16)

§5as's reopen is one command and two equations, and both equations are read from `prefetch_issued_per_token` and
`prefetch_adopted_per_token`. Reading the driver instead of trusting the design found the gap: `run_arm()` keeps
from the footer only the names listed in `COUNTERS`, and neither of those two was listed; `PRINTED_CELLS` and the
table's header carried five counters, not seven. The page §5as said it would produce could not have been produced.
The ring row the whole instrument turns on — 39.45 issued against 18.86 adopted — is in this file because a person
read it out of the server log in §5ag, and after §5as the same hand step would have been required of a 40-minute
batch whose own output was silent on the two columns the solve consumes.

### The engine already emits both, unconditionally

`prefetch_issued_per_token=%.2f prefetch_adopted_per_token=%.2f` sits in the same `TinyTitan runner …` diagnostic
line as `io_ms` and the hit rate, at
[../sources/TinyTitanLib/ServerModelSession+Diagnostics.swift:348](../sources/TinyTitanLib/ServerModelSession+Diagnostics.swift#L348),
with no `if` in front of it, so a ring-off arm reports `0.00` rather than omitting the key. Nothing engine-side
changed here; the gap was purely in the reader. That distinction is what makes the refusal below worth having: a
missing cell can only be a driver that dropped a name or a footer that moved, never an arm that legitimately
issued no reads.

### The seam, test-first

51 → 53 tests in [../benchmark/test_knob_sweep_verdict.py](../benchmark/test_knob_sweep_verdict.py). Both new
`PricePair` tests were seen failing before the production change, and each says which seam it pins:

```
AssertionError: 'prefetch_issued_per_token' not found in ('expert_hit_rate', …, 'expert_load_p50_ms')
AssertionError: Lists differ: ['4', '26'] != ['44.10', '12.35']
```

Then `COUNTERS` 9 → 11 names, `PRINTED_CELLS` 5 → 7, `table()`'s header and each row gained the two columns, and
`verdict()`'s `NOT MEASURED: the {arm} arm logged no {key}` refusal now covers seven cells — so a page missing
either counter exits 2 instead of printing `--` under a heading that promises a number.

Both pins kill. The mutants were applied to the working file, run, and reverted; `git diff --stat` after the
restore is byte-for-byte the diff before them:

| mutant | test | outcome |
|---|---|---|
| the two `prefetch_*` names dropped from `COUNTERS` | `test_the_batch_parses_the_two_counters_the_price_solve_consumes` | FAIL, and the message names the counter |
| the two row cells dropped from `table()`, header left intact | `test_the_table_prints_issued_and_adopted_reads_per_token` | FAIL: the row's last two cells read `['4', '26']` |

The second is the one worth keeping. A table whose header promises a column its rows do not carry does not crash:
the neighbouring columns shift into the price slots, and a reader keying `x` and `y` off it is quoted an eviction
count as a cost per read.

### The fixture was the bug, and both sides are named

Widening `PRINTED_CELLS` broke 18 tests, every one reading `NOT MEASURED: the base arm logged no
prefetch_issued_per_token`. Production was right — a printed column an arm did not log must refuse — and the
suite's own `arm()` fixture was wrong: it logged five of the seven cells the table prints. The inventory of every
test-file change, none of which loosens an assertion:

| change | what it is |
|---|---|
| `CELLS` 5 → 7 names | the fixture logs a complete row, so the gate no longer refuses the fixture |
| fixture counters gained `39.45` / `18.86` | a fixture row reads like a real ring-on arm, valued at §5ag's shipped tier |
| `arm()` docstring "five" → "seven" | prose |
| the `sync_event` row literal ends `… 26 39.45 18.86` | re-pinned: the table genuinely has two more columns |
| all-`None` refusal row `["--"] * 5` → `["--"] * 7` | re-pinned: it still refuses, over seven cells now |
| one-missing-column row appends `39.45`, `18.86` | re-pinned for the same reason |
| header test gained `issued/pt`, `adopted/pt` | stricter, not looser |

No run in this file was withdrawn, because there is no #205 or #206 measurement to withdraw.

### The page now carries its own inputs

`table()` called on §5aq round 1's published rows and §5ag's published ring-off row; cells those tables do not
carry are `None`, so they print the refusal token. No model process, no server, nothing measured on hardware for
this render:

```
arm               tok/s  vs base   hit%  io_hid%   io_ms   evict   waits issued/pt adopted/pt
---------------------------------------------------------------------------------------------
base               4.55    +0.0%   77.5       --    99.4   42313      --     42.75      22.85
slots_64           4.47    -1.8%   77.4       --    99.1   42392      --     42.72      22.64
layer_depth25      4.00   -12.1%   75.5       --   117.0   45927      --     76.03      13.43
slots64_depth25    4.11    -9.7%   75.8       --   112.7   45489      --     76.22      14.58
base_again         4.33    -4.8%   77.1       --   101.6   43004      --     42.57      21.05
prefetch_off       4.79    +5.3%   80.7       --    85.3      --      --      0.00       0.00
```

The last row is the point of the render: an arm that issued nothing prints `0.00 0.00`, a value, not a refusal.
Its `+5.3 %` is §5as deviation 3 made visible — the row comes from §5ag's batch, whose whole tier ran ~11 %
faster than §5aq's, so a ring-off arm beats a §5aq baseline on the clock while being the arm that *loses* 5.6 %
when it is paired inside its own batch. That is the arithmetic reason the trio has to be one batch, printed here
by accident rather than designed.

### The geometry, re-derived rather than remembered

§5as's whole argument is a condition-number claim sourced from a throwaway script. Recomputed from published
counters alone, with the convention stated rather than implied — rows live in the (`Δissued`, `−Δadopted`) plane,
so the second coordinate is **positive** where a step makes adopted reads *fall*:

| pair | rows | this section | §5as published |
|---|---|---|---|
| ring-off × depth-2 | (−39.45, +18.86), (+33.28, +9.42) | **41.36°, cond 1.51** | 41.4°, cond 1.5 |
| depth-2 × depth-3 | (+33.28, +9.42), (+55.28, +13.02) | **2.55°, cond 22.47** | 2.6°, cond 22 |

**This table printed its ring row negated until #207's screen could not reproduce E1 from it.** The step is ring-on
→ ring-off, so `Δissued` is `0 − 39.45 = −39.45` and adoptions *fall*, which the convention stated two lines above
makes a positive second coordinate: `(−39.45, +18.86)`. Negating a whole row leaves `det`'s magnitude alone, so the
41.36° and cond 1.51 were never at risk — which is exactly why nobody noticed. What the negation does change is the
solve: paired with the same `+11.748 ms` step this section quotes, the printed row gives `x = 0.681` / `y = 0.802`,
**inside** §5ar's band and the opposite verdict to the one published below. The corrected row is E1 verbatim
(`18.86 y − 39.45 x = 11.7 ms`), and it is what reproduces the `y = 1.584` of the table after it. The coordinates
are now pinned to the prices by `Sign` in `benchmark/test_tinytitan_price_pair.py`, so the convention fails a test
instead of a reader (§5au).

The ring step's wall cost from §5ag's two rates is 11.748 ms/token against the 11.7 ms §5as quoted. Both claims
hold. The sign is the instrument: read with the opposite sign the identifying pair collapses to **9.75°, condition
5.91** — not a good design but one of the worse rows in §5as's own table — and my first pass at this arithmetic
reported it as ~170°, the supplement of the same near-parallel line. The classification was right and the number
was not, which is the reason this section recomputes from published counters instead of remembering.

The composition §5as bracketed is now computed both ways, and the difference is a column choice its table did not
state: the published row solved the depth step's Δms from `body_ms` (29.591 ms); on the wall-clock Δms (30.220 ms)
the same two rows give

| depth Δms read from | `x` ms/issued | `y` ms/adopted | `p*` | vs the admissible band 0.788-1.301 |
|---|---:|---:|---:|---|
| `body_ms` | 0.448 | 1.559 | 0.287 | above |
| wall | 0.460 | 1.584 | 0.290 | above |

The out-of-band verdict does not depend on the choice, but the two prices differ by 2-3 %, and the campaign has
been mixing the columns without saying so: §5ar's band is a **service time** — one 2.641 MiB read at a measured
wall rate — which is a wall quantity, so the equation it is tested against should carry a wall Δms. A screen that
takes the column as an argument and names it in its own output is the only way to stop that being a reader's job,
and that is the next unit (#207). §5as deviation 4 refused the September `p* = 0.105` as "a judgement, not a
measurement": the refusal now has arithmetic behind it and still is not a price, because a band violation computed
across two batches cannot be told apart from the 10 % drift between them. Only an in-batch trio separates the two.

### Why the live round is still not run

At 19:10 `ps -A -r` put **RobloxPlayer (75778) at 121.1 % CPU**, the same resident it has been since 17:52, with
WindowServer at 21.8 %, 5.71 of 7.17 GB of swap used, 64 % of memory free, load 3.09 / 3.49 / 3.97, on AC at
100 %, and no model process running. The trio's requirement is ±1 ms/token, which §5aq's rounds 2-3 price at
"unusable while a game is resident". One instrument note is worth keeping, because it very nearly sent a 40-minute
batch to a machine I had mis-read as idle: `top -l 1 -o cpu` reported **every** process at 0.0 % CPU on its first
sample — its CPU column is a delta over the sample interval and the first interval is empty — while `ps -r` named
the 121 % occupant in the same minute. Nothing belonging to anyone else was terminated.

### Verdict

#206's command now prints the two columns its own equations are read from, and refuses a page that lacks either.
Two things this commit deliberately does not do: it does not **solve** the pair — the arithmetic is still done
outside the driver, and the shipped `price_pair` screen that reads this page, states which column it solved from,
and answers with a status is #207 — and it moves no runtime default, no engine code and no shipped configuration.

Deviations.

1. **Nothing was measured in this section.** Every number is published-counter arithmetic, a synthetic `table()`
   render, or a gate/test run. No model process was started and no arm ran.
2. **The ring row's Δms is inferred from §5ag's two rates, which are a different batch from every other row here.**
   The geometry uses only its *sign*, which is unambiguous (ring on is the faster tier); 11.748 ms carries §5ag's
   batch drift and supports no claim of its own. §5as deviation 3 is the standing warning and the render above
   shows the trap.
3. **The depth-3 row is §5as's synthesis, not an arm that ran**, so 2.55°/22.47 is a property of that synthetic
   row and #206's out-of-sample prediction against it is still unpaid.
4. **The `body_ms`/wall choice was unstated when §5as's composition rows were written.** It is disclosed there as
   an added note rather than silently corrected: §5as's table numbers are unchanged, and both readings land out of
   band, so no conclusion of that section moves.
5. **`/tmp/seam206/verify.py`** holds this section's arithmetic. It is a throwaway — not committed, not proposed
   as a tool, superseded by #207. The two mutation runs edited `benchmark/tinytitan_knob_sweep.py` in place and
   were reverted from a copy taken first; the committed diff is the counter seam and nothing else.
6. **Provenance.** §5as landed at `1b24b09` and the depth-3 arm at `2327dfd`, both already on `main`; the counter
   seam, the two pins, the three repointed line pointers and this section are the commit that carries them.
   `benchmark/test_knob_sweep_verdict.py` 53 tests OK, `tools/lint.sh python` ok (ruff 0.16.7, 201 scripts parse
   under 3.13), `tools/lint.sh docs` ok, markdown links 96 checked / 0 broken.

## 5au. #207 ships the price solve §5as and §5at left in a throwaway — and the page's own printed digits turn out to bound what a band call can mean; fed a real page, the screen's widest-angle rule then picks a row that lies entirely inside that page's drift control (2026-10-16)

§5at closed the seam that let #206's command *print* the two columns the price equations are read from, and named
what it still could not do: solve them. The arithmetic lived in `/tmp/seam206/verify.py`, a throwaway with no
status, no notion of which column it had used, and no way to say that a page was too coarse to decide anything.
[benchmark/tinytitan_price_pair.py](../benchmark/tinytitan_price_pair.py) is that arithmetic shipped, and it
answers with the statuses this tree's other drivers use — `0` identified and inside §5ar's band, `1` identified a
price and that price refutes the two-price model, `2` could not solve, with the reason named — over a page read
from stdin or `--page`. Nothing is measured on hardware: it is arithmetic over a table somebody else printed, so
it runs in CI with no model, no server and nothing fetched.

### The digits are part of the measurement

Every price the campaign has published was solved from rates *as printed*, and nobody had said how much the
printing costs. The screen does, from the digits of the page it was handed: it records the coarsest printed row
per column ([`:121`](../benchmark/tinytitan_price_pair.py#L121), [`:321`](../benchmark/tinytitan_price_pair.py#L321)),
turns a half-ulp into a width on each equation's `Δms` and on each counter coordinate
([`:384`](../benchmark/tinytitan_price_pair.py#L384)), propagates all six widths through the Cramer partials
([`:448`](../benchmark/tinytitan_price_pair.py#L448)), and prints the result:

```
digits: the page prints tok/s to 2 decimals (±0.005), so y is known only to ±0.033 ms and x to ±0.013 ms
```

`Price` carries `x_bound` and `y_bound` as *required* fields rather than defaults, because a price that silently
had no bound would read as an exact one. Bounds are rounded up to the millimetre, so printing one cannot
understate it. The consequence is the part that matters for the campaign: `_verdict`
([`:808`](../benchmark/tinytitan_price_pair.py#L808)) compares the bound to the distance from the median `y` to
the nearer band edge and, when the bound reaches it, appends `_cannot_decide`
([`:726`](../benchmark/tinytitan_price_pair.py#L726)) — in *both* the REFUTED and the ADMISSIBLE branch. A band
call that falls inside the resolution of the rates it was computed from is not a refutation of the model; it is
the page running out of digits.

*(Amended by §5az: `y_bound` is no longer the digits alone. `Price` now carries `x_digits`/`x_drift` and
`y_digits`/`y_drift` beside the totals, because the campaign's own pages put ~20× more width on a price through
machine drift than through printing, and a bound printed under the sentence above would have read as a rounding
note. Every `bound on y` in the table below is a digits half and is still correct as one; §5az prints what each
of those pages may actually price by.)*

### Four pages, and what each one's digits cost

The identifying trio — ring-on baseline, ring-off, depth-2 — printed at three precisions, and a second one-decimal
trio whose counters come from §5aq rather than §5ag, built on purpose to price *inside* the band:

| page | printed `tok/s` | best pair | `y` | bound on `y` | distance to nearer edge | verdict |
|---|---|---|---:|---:|---:|---|
| `ks.table()` render | 2 decimals | depth-2 × ring-off, 41.4° cond 1.5 | 1.598 | ±0.033 | 0.297 outside | REFUTED, exit 1 |
| §5at's published table | 3 decimals | the same, 41.4° cond 1.5 | 1.585 | ±0.004 | 0.284 outside | REFUTED, exit 1 |
| one decimal, out-of-band arm | 1 decimal | the same, 41.4° cond 1.5 | 1.362 | **±0.310** | 0.061 outside | REFUTED + *cannot decide*, exit 1 |
| one decimal, in-band arm | 1 decimal | depth-2 × ring-off, 43.9° cond 1.4 | 1.004 | **±0.283** | 0.216 inside | ADMISSIBLE, `p*=0.237` pays, + *cannot decide*, exit 0 |

*(Amended by §5az: the last row's status. #209 scores the ratio's own widths, and `x = 0.238 ±0.121` over
`y = 1.004 ±0.283` makes `p*` run 0.091 to 0.498 — the gate's 0.155 is inside its own interval, so that page now
prints `cannot decide the gate` and exits **2** rather than drawing a `pays` it cannot separate. The first three
rows keep their statuses and their bound column, which is the digits half; on their totals the two refutations
stand at ±0.147 against a 0.297 ms margin and ±0.107 against 0.284, so they survive the drift term with roughly
half the room they were published with.)*

Three things come out of that table.

**The sweep's own page and the published page agree, and only the bound knows it.** The 2-decimal render answers
`y=1.598` where the 3-decimal page answers `1.585`: a 0.013 ms spread from nothing but printing. It is inside the
coarser page's own ±0.033, so the two pages are the same measurement — and neither reproduces the other's third
decimal, which is why §5ar's prices may be quoted to three decimals but may not be *argued* at that precision.
§5at's published `1.584` is the same solve again: the exact value off these printed rows is `1.58465`, so `1.584`
and `1.585` are one number displayed two ways, and the ±0.004 bound covers the difference.

**One decimal is enough to un-decide a band call.** Half-ulp on a one-decimal `tok/s` is 0.05, which is ~4.3 ms
on a step of this size, and 0.039 ms of `y` per ms of wall turns that into ±0.31. Against a band 0.513 ms wide,
that is not a rounding note — it is the difference between a refutation and no answer. The campaign has published
tables at one and two decimals; anything read off them is now known to be un-decidable at the edges.

**The caveat had to be built on the side that agrees with the model.** The last row above is the reason. A
`cannot decide` clause that only ever prints next to a refutation is a clause that has never been tested, and it
would be worth nothing, because the dangerous error is the flattering one: an arm that *costs* an implausible
18 ms per adopted read gets questioned, while an arm that lands at 1.004 ms — right inside the band, gate pays,
`+0.44 ms` of the token — gets believed. Both fixtures are in `Prices`, and the in-band one
([`test_tinytitan_price_pair.py:518`](../benchmark/test_tinytitan_price_pair.py#L518)) is written so its hand
arithmetic is in the comment next to it.

### The sign §5at's table printed backwards

§5at's geometry table carried the ring-off row as `(+39.45, −18.86)`. The step is ring-on → ring-off, so issued
reads *fall* (`0 − 39.45`) and adoptions fall, which under the convention printed two lines above that table makes
the second coordinate positive: `(−39.45, +18.86)`. The row is corrected in place there and the disclosure is
there with it; what belongs here is what the error did and did not cost. Negating a **whole row** leaves `det`'s
magnitude alone, so the 41.36° and cond 1.51 were never wrong — which is exactly why nobody noticed. Negating one
row while pairing it against the same `+11.748 ms` solves `x = 0.681` / `y = 0.802`: **inside** §5ar's band, the
opposite verdict to the one published below the table. A reader who took the printed row and did the arithmetic the
next section ships would have concluded the two-price model was confirmed.

The screen now pins the convention rather than describing it: `Sign`
([`:308`](../benchmark/test_tinytitan_price_pair.py#L308)) asserts all three cases — a step that makes adoptions
fall is positive, the ring-off step is E1 verbatim (`18.86 y − 39.45 x = 11.7 ms`), and a step that makes them
*rise* is negative — and `Prices` asserts that those coordinates solve the published `x=0.460` / `y=1.58`. The
other mis-signing, negating **one coordinate of one row**, is the one that does change `det` and collapses the
identifying pair to §5at's 9.75°/cond 5.91; that pair of numbers stays prose, because no test in this suite
asserts it. What the suite closes is the mistake §5at actually made, not every neighbouring one.

### On a real page, the widest angle points at nothing

Feeding the screen §5at's own rendered six-row page is the first time it has read a table assembled from more than
one batch, and it found a design defect in the screen itself. `best_pair` picks the widest-angle candidate
([`:597`](../benchmark/tinytitan_price_pair.py#L597)); every candidate on that page is below, with the row that
generates each:

| pair | `deg` | cond | `x` | `y` |
|---|---:|---:|---:|---:|
| slots64_depth25 × slots_64 | **84.25** | 1.01 | −3.791 | **18.189** |
| layer_depth25 × slots_64 | 82.33 | 1.01 | −4.223 | 18.127 |
| base_again × slots64_depth25 | 81.83 | 1.01 | −0.810 | 6.123 |
| base_again × layer_depth25 | 79.91 | 1.02 | −0.825 | 6.121 |
| base_again × prefetch_off | 56.16 | 1.20 | 3.775 | 6.581 |
| prefetch_off × slots_64 | 53.75 | 1.24 | 11.118 | 20.319 |
| layer_depth25 × prefetch_off | 43.93 | 1.44 | 0.683 | 0.796 |
| prefetch_off × slots64_depth25 | 42.00 | 1.49 | 0.562 | 0.570 |
| layer_depth25 × slots64_depth25 | 1.93 | 29.77 | −0.706 | 5.702 |
| base_again × slots_64 | 2.42 | 23.69 | −292.294 | −23.026 |

The rule picks the 84.25° pair and prints `y=18.189` — 14× the top of the band — and the ordering is not noise:
every pair above 79° prices `y` at 6.1–18.2 ms, and the only pair on the page that prices `y` *inside* the band is
`layer_depth25 × prefetch_off` at 43.93°. On a page like this the widest angle is anti-correlated with a plausible
price, and the reason is geometric. A row whose counters barely move points in whatever direction its noise happens
to lie, and a near-vertical row is *automatically* near-orthogonal to a row that is not. `slots_64`'s step is
`(−0.03, +0.21)`, a magnitude 0.212; `layer_depth25`'s is `(+33.28, +9.42)`, a magnitude 34.587. The smaller row is
0.6 % of the larger and the angle between them is 82.33° — not because the design is good, but because the small
row's `Δadopted` happens to be seven times its `Δissued`. Conditioning is not signal.

What kills the chosen pair is on the same page. `base_again` is the drift control: the same configuration as
`base`, re-measured, so its step *is* the page's own drift sample, and it moved `(−0.18, +1.80)` in counters and
11.17 ms on the wall. `slots_64`'s entire contribution to the solve is `(−0.03, +0.21)` and 3.93 ms — all three
smaller than the drift control's. The screen used a row lying wholly inside the drift that the same page measured,
and priced adopted reads at 18.189 ± 2.771 ms. Note that the resolution bound does not catch this and cannot: the
page's digits are precise to ±0.005 tok/s, the inputs are fine. **The digits bound the answer's precision, the
angle bounds the design's conditioning, and neither bounds the row's signal.** Three separate bars, and the screen
has two.

The same page shows what the in-band pair at 43.93° would have said had the rule picked it instead: `y=0.796`,
inside the band, `exit 0`, model confirmed. Both possible verdicts from that page are artifacts of the ring-off row
coming from §5ag's batch and the other five from §5aq's — which is precisely the trap §5as deviation 3 warned about
and §5at made visible. A page assembled across batches cannot price anything, and a screen that prints one number
and a status cannot say so. That is a finding, not a fix: the candidate rule is a minimum-signal bar keyed to the
page's own control row, plus printing the *spread* of `y` across the candidates rather than one selection, and it is
filed as #208. It is deliberately not patched here, because a bar tuned until this page returns the price the
campaign already believes is a bar fitted to the answer.

### What the tests pin, and where they idealize

22 tests in [benchmark/test_tinytitan_price_pair.py](../benchmark/test_tinytitan_price_pair.py), 75 OK alongside
`test_knob_sweep_verdict`. Every fixture page is built by calling the real `ks.table()`, so the two files fail
together the moment the sweep's page and the screen's reader stop agreeing. Five traps are pinned: the sign
(`Sign`), the sweep's cumulative page — four tables per batch, only the last is the batch
(`ReadingAPage:173`) — the zero row stepped over as the drift control (`:215`), out-of-band `y` reported as a
refutation rather than a price (`:241`), and the digits (`:270`, `:280`, `:289`, `:294`).

The drift-control test idealizes, and the real page says so: it asserts `base_again` is stepped over because its
printed counters are *identical* to the baseline's, which is what a fixture does and what §5at's render shows a
real control does not — there it moves `(−0.18, +1.80)`, so the exact-zero rule at `:400` does not fire and the
control becomes a priceable row. The test is not wrong; it pins the case the rule actually handles. The one above
it in the table is the case it doesn't, and that gap is #208 rather than a loosened assertion.

### Verdict

The price solve is shipped: it names the column it solved from, refuses a page missing either counter, reports the
condition number and how much 1 ms of wall moves `y`, bounds its own answer by the digits it was handed, and
declines to decide a band call inside that bound. It moves no runtime default, no engine code and no shipped
configuration, and it settles nothing about the two-price model: the pages it can read are the ones §5at already
wrote, and every one of them is a cross-batch artifact. #206's in-batch trio is still the only thing that can
produce a page this screen's answer means anything for, and #208 now names the bar that page has to clear before
the selection is trustworthy.

Deviations.

1. **Nothing was measured in this section.** No model process, no server, no arm. Every number is either
   published-counter arithmetic or the screen's own output over a page built from published rows, run to
   `exit` status and quoted verbatim.
2. **The three-decimal and one-decimal fixtures are synthetic.** They exist to measure what printed precision
   costs a band call, which is a property of the *screen*, not of any run. Their arms' counters are §5ag's and
   §5aq's, so they inherit §5at deviations 2 and 3: nothing in them prices a read.
3. **The §5at-page candidate table is the screen's first read of a multi-batch page**, and its widest pair's
   `y=18.189` is offered as evidence against the selection rule, not as a measurement of anything. The 43.93°
   pair's `y=0.796` is equally an artifact; the difference between the two is which artifact the rule happened to
   pick.
4. **`/tmp/au207/`** holds the four pages quoted above. Throwaways, not committed, not proposed as tools; the
   committed artifacts are the screen, its suite, the CI registration and §5at's corrected row.
5. **CI registration is verified by the gate, not by CI.** `test_docs_facts_suites.test_this_repository_is_registered`
   failed until both new files were tracked — the gate reading `ci.yml` against the working tree — which is the
   designed behaviour (AUD-211) and not evidence that a run has executed the suite. The 22 tests ran locally under
   the pinned ruff 0.16.7, which also parses the two files clean.
6. **Provenance.** §5at landed at `ac69918` and its shipped counter seam at `894ecfa`, both on `main`. This
   section, the two new python files, the `ci.yml` row and comment, and §5at's negated-ring-row correction are
   the commit that carries them. `tools/lint.sh python` ok (ruff 0.16.7, check and format clean, 203 scripts parse
   under 3.13), `tools/lint.sh docs` ok (87 documents, 101/105 python suites registered in CI), the CI
   markdown-link step run locally 109 checked / 0 broken, and `test_docs_facts_suites` + both benchmark suites
   87 tests OK together.

## 5aw. #208 gives the price screen a third bar — a row must move more than the page's own drift to be signal — and the live round it was aimed at turns out to have needed it badly: the rule shipped at `cf29cd2` priced that page off `base_again`, the drift control itself (2026-10-16)

§5au filed two changes and did not patch them, on the grounds that a bar tuned until a page returns the price the
campaign already believes is a bar fitted to the answer. [benchmark/tinytitan_price_pair.py](../benchmark/tinytitan_price_pair.py)
now carries both: a minimum-signal bar keyed to the page's own control row
([`:491`](../benchmark/tinytitan_price_pair.py#L491)) and the printing of every design that survives it rather
than one selection ([`:608`](../benchmark/tinytitan_price_pair.py#L608)), with a refusal when those designs
disagree ([`:618`](../benchmark/tinytitan_price_pair.py#L618)). The statuses are the same three; nothing was
renumbered. Implementing it turned up two defects in the screen shipped four hours earlier, and one of them
changes what §5au's page actually searched.

*(Amended by §5az: the drift sample does more than bar rows now — `_solve` turns it into widths on the price, so
the same control row that inked `slots_64` also sets what the answer is worth. Nothing was renumbered then, but
status 2's meaning has widened: it now covers a page that solves a price and cannot draw a gate from it.)*

### The bar, and what it is keyed to

`base_again` is the same configuration as `base`, re-measured, so its step is this page's drift with no config
change in it — a measurement of the machine, not of an arm. The screen now treats it as a floor: a row is stepped
over when it moves less than the control **in every coordinate the solve reads**, both counters and the column
([`:530`](../benchmark/tinytitan_price_pair.py#L530)), and the control is never a solve partner however far it
moved ([`:528`](../benchmark/tinytitan_price_pair.py#L528)). AND over three coordinates is the conservative
choice — it removes only rows the drift can explain in full, and leaves any row that moves more than the control
in even one of them. `--control` names the row ([`:896`](../benchmark/tinytitan_price_pair.py#L896)), and a page
that carries no control is not a refusal: the screen says `no drift sample (base_again is not on this page), so
no row was checked` and solves, because a bar with nothing to key to must not silently become a veto.

### Three bars, on one page at last

§5au's page — §5at's six rendered rows, assembled from two batches — through the screen shipped at `cf29cd2`, and
through the same file with the bar:

```
round 1: best pair slots64_depth25 x slots_64 at 84.3 deg cond 1.0  x=-3.791 y=18.189
digits: ... so y is known only to ±2.771 ms and x to ±0.703 ms
REFUTED: y=18.189 is outside the admissible band 0.788-1.301 ms ...                     exit 1
```
```
round 1: best pair layer_depth25 x prefetch_off at 43.9 deg cond 1.4  x=0.683 y=0.796
  signal: drift sample base_again moved (-0.18, +1.80) and +11.17 ms; inside it, not signal, stepped over: slots_64
  designs: 2 of 3 solvable pair(s) reach 20 deg: y=0.796 (layer_depth25 x prefetch_off, 43.9 deg),
          y=0.570 (prefetch_off x slots64_depth25, 42.0 deg); y spread 0.226 ms
table 1: REFUSED: the 2 designs this page can price do not agree on the band ...          exit 2
```

The 18.189 ms artifact is gone, and not because a smarter number replaced it: the pair that produced it no longer
qualifies, and what remains says out loud that the page holds two prices. Told there is no drift sample
(`--control nosucharm`), the same page prints its whole candidate pool and the disagreement is worse, not better —
`8 of 10 solvable pair(s) reach 20 deg`, six above the band, one inside it, one below, `y spread 19.749 ms`. That
is the honest shape of the thing §5au described by hand: the spread, not the selection, is the finding.

**Why a straddle exits 2 and not 1.** Status 1 means *identified a price, and that price refutes the two-price
model* — a measurement with a verdict. A page whose conditioned designs land on opposite sides of a band edge
identified no single price, so it has no verdict to be refuted by; it could not be solved, which is status 2, and
the reason is named. Reporting the inside one (`y=0.796`, gate would pay) would have flattered the model, and
reporting the below one would have refuted it, on a page that cannot tell them apart.

### Two defects in the file shipped as `cf29cd2`

**The candidate set was a window, not a search.** `best_pair` walked the sorted arm names two at a time
(`zip(names, names[1:])`) while its own docstring promised "the widest non-degenerate pair a table offers", and
§5au's published table lists ten pairs. On five priceable rows a window sees four of ten. §5au's answer happened to
survive the difference because its widest pair — `slots64_depth25 x slots_64` — is adjacent in sorted order, which
is exactly how a defect like this hides: it changed nothing on the page that exposed it. `itertools.combinations`
([`:576`](../benchmark/tinytitan_price_pair.py#L576)) is the fix, and it retro-validates §5au's table: the pool the
fixed screen prints is the ten rows that section published by hand, same angles same prices. The bug is pinned by a
page where the widest pair is *not* adjacent
([`test_tinytitan_price_pair.py:393`](../benchmark/test_tinytitan_price_pair.py#L393)); RED there showed the window
picking a 39.3 deg neighbour.

**Dead arithmetic that was never dead code.** `_solve` carried a second `return Price(...)` after the first,
predating the bounds and omitting the now-required `x_bound` / `y_bound` — unreachable, and would have raised
`TypeError` had it ever been reached. Deleted; no behaviour moved.

**A refusal that named the wrong cause.** Once the bar can empty the pool, a page whose every row is inked printed
`nothing priceable`, which is true of the arithmetic and false about the cause: three rows *were* priceable and the
drift sample covered all of them. Driving that note out was the last test written here
([`:450`](../benchmark/test_tinytitan_price_pair.py#L450)), RED against the version that shipped an hour earlier,
and the pre-existing note `only N priceable arm(s) with signal` lost `with signal` — its count is the pre-bar one,
which is the second time in this campaign a label has promised a check the number behind it does not perform.

### The live round that needed the bar

#206's round 1 (`/tmp/trio206/round1.log`, measured on this Mac for the trio, not here) is the first live page the
screen has read, and the old rule priced it off the drift control:

| screen | best pair | `x` | `y` | out-of-sample residual | status |
|---|---|---:|---:|---:|---|
| `cf29cd2` | `base_again` x `layer_depth25`, 74.2 deg | 4.403 | −10.977 | −182.67 ms/token (−76.17 %) | 1 REFUTED |
| this section | `layer_depth25` x `prefetch_off`, 47.8 deg | 0.148 | −0.197 | +25.31 ms/token (+10.55 %) | 1 REFUTED |

Same status, same word, and only one of them is about a config change. The sweep's own footer is what makes round 1
readable at all: `baseline drift over the sweep: 9.9% (4.17 -> 3.76)`, `INCONCLUSIVE: baseline drift 9.9% is
larger than the smallest win claimed, +4.5% on prefetch_off`, exit 1, response text identical across 5 arms
(`md5 3c0c3391`). Round 1 is contaminated and is not used as a price.

**And the bar's own limit, measured on that page.** It inked nothing there — `no row lies inside it` — because the
control moved counters by only `(+0.23, −2.29)` while its wall moved +26.15 ms. A bar keyed on the counters cannot
see a drift that is all clock; it kept the selection honest by refusing to partner the control, which is the other
half of the rule. Keying it on wall alone would ink real arms whose counters move and whose clocks drift, so the
asymmetry stays, and it is a finding: **the signal bar bounds a row's counter signal, the sweep's drift footer
bounds its clock, and round 1 is the case where only the second one fires.**

### Tests, and the one test-file edit made for the pin

30 tests in [benchmark/test_tinytitan_price_pair.py](../benchmark/test_tinytitan_price_pair.py) — 29 plus the
cause-naming one — 95 OK alongside `test_knob_sweep_verdict` and `test_docs_facts_suites`; `tools/lint.sh python`
ok. Seven tests are new: the enumeration ([`:393`](../benchmark/test_tinytitan_price_pair.py#L393)) and
[`Signal`](../benchmark/test_tinytitan_price_pair.py#L405)'s six (row inside drift `:416`, control never a partner
`:427`, straddle refuses `:439`, no control says so `:452`, one survivor prices exactly as it did unbarred `:461`,
zero drift bars nothing `:477`), plus the inked-page cause `:487`. One edit was made for the linter rather than for
behaviour and is inventoried here: the new `best_line` helper's bare `assert` became `raise ValueError`, because
pinned ruff 0.16.7 enforces S101 — the check's strength is unchanged (it still fails the test unless exactly one
`best pair` line is printed), and it is code added in this section, not a pre-existing assertion weakened.

Deviations.

1. **One live page was read, none was measured here.** `/tmp/trio206/round1.log` is #206's round 1, run on this Mac
   earlier; every other number is this screen's output over pages built from published rows, run to `exit` status
   and quoted verbatim. Nothing in this section prices a read.
2. **Round 1 is contaminated and stays unusable.** 9.9 % baseline drift, `INCONCLUSIVE` from the harness itself,
   and its `y=−0.197` is offered as a demonstration of the two rules disagreeing about the same table, not as a
   refutation of the model. #206's rounds 2-3 are the measurement; this section is the instrument that has to read
   them.
3. **`ALL_INKED`, `BIG_DRIFT` and `WIDEST_NOT_ADJACENT` are synthetic**, built to exercise three branches of the
   screen; `PAGE5AT`/`PAGE5AT_ONE` are §5at's published rows across two batches and inherit §5at deviations 2 and 3,
   including that both verdicts available to them are artifacts.
4. **The bar is counter-and-column keyed, so a clock-only drift passes it** — measured above on the live page. Not
   fixed: the alternative keys ink every row whose counters move. The sweep's drift footer is the check for that
   case, and it fired.
5. **The two `cf29cd2` defects are disclosed as shipped defects, not as design.** The window enumeration is the
   more serious of the two and it changed no published number in §5au, which is the reason it is described above
   with the page that hid it rather than as a cleanup.
6. **`/tmp/aw208/old_screen.py`** is `cf29cd2`'s file extracted for the before/after comparison. Throwaway, not
   committed, not proposed as a tool; the committed artifacts are the screen, its suite and this section.
7. **No new file, so no CI registration change**: `cf29cd2`'s `ci.yml` row already names this suite, and
   `tools/lint.sh docs` verifies that against the working tree rather than a run having executed it (AUD-211).
8. **Provenance.** §5au and the screen it shipped landed at `cf29cd2` on `main`; #208's screen, suite and the two
   defects above landed at `fd3823e`, and this section is the commit that carries it. Seen locally before
   committing, on that tree: `tools/lint.sh python` ok (ruff 0.16.7, check and format clean, 203 scripts parse
   under 3.13), `tools/lint.sh docs` ok (87 documents, 101/105 python suites registered in CI), the CI
   markdown-link step run locally 122 checked / 0 broken, and the 30 tests plus `test_knob_sweep_verdict` and
   `test_docs_facts_suites` 95 OK together. None of that is evidence a CI run executed them; it is evidence the
   tree the run will check is the tree this section describes.

## 5ax. #206 lands two clean paired rounds and pins `y` at last — and the six-link gate's answer flips between them, because `x` is a *difference* of the two solving rows' steps while `y` is a *sum*, so the sweep's blocked single-pass order is what bounds it (2026-10-16)

§5aw shipped the instrument and said plainly that it was the instrument, not the measurement: "#206's rounds 2-3 are
the measurement". They are here, both clean, and together they do what §5as, §5at and §5au could not — identify the
two prices off one page — while answering a different question from the one the campaign went to ask. `y` is now
pinned by two independent rounds. `x`, the number the §5ar withholding lever turns on, is not identified by this
sweep design at all, and the reason is structural rather than noisy.

**What ran.** Two separate invocations, one per round, on AC power, after the full precondition block (no model
process, `memory_pressure -Q` 83 % free, macOS 27.0.1, Swift 6.4 target `arm64-apple-macosx27.0.0`, 279 Gi free,
qwen38 4-bit install with its `verified-install.json`), on the tree at `7ebbba1` with nothing modified and a
release binary newer than every file under `sources/`:

```bash
TINYTITAN_BENCH_RAM_BUDGET=12 SWEEP_TOKENS=384 \
  python3 benchmark/tinytitan_knob_sweep.py --arms base,prefetch_off,layer_depth25,layer_depth325,base_again
```

`--ram 12` is pinned explicitly rather than inherited, so the operator's ceiling is on the command line rather than
in a default. Both rounds exit 0. Both report `response text identical across 5 arms (md5 3c0c3391)`, so every rate
below is a rate at the same work. Logs: `/tmp/trio206/round2.log`, `/tmp/trio206/round3.log`.

| arm | round 2 tok/s | ms/token | issued/pt | adopted/pt | round 3 tok/s | ms/token | issued/pt | adopted/pt |
|---|---|---|---|---|---|---|---|---|
| base | 4.42 | 226.24 | 42.65 | 22.12 | 4.32 | 231.48 | 42.58 | 21.36 |
| prefetch_off | 4.15 | 240.96 | 0.00 | 0.00 | 4.17 | 239.81 | 0.00 | 0.00 |
| layer_depth25 | 4.15 | 240.96 | 75.69 | 10.15 | 4.03 | 248.14 | 75.67 | 10.33 |
| layer_depth325 | 3.69 | 271.00 | 102.85 | 6.67 | 3.85 | 259.74 | 102.80 | 5.86 |
| base_again | 4.49 | 222.72 | 42.51 | 21.86 | 4.55 | 219.78 | 42.56 | 21.64 |

The counters reproduce across rounds to the third digit where the machine is not involved (`prefetch_off` 0/0 both
rounds, `layer_depth25` issued 75.69 against 75.67, `layer_depth325` 102.85 against 102.80) and to the first where
it is (base adopted 22.12 against 21.36). Baseline drift: 1.7 % in round 2, 5.4 % in round 3 — in wall terms
−3.53 ms and −11.70 ms per token.

**Both rounds, through the screen.** `benchmark/tinytitan_price_pair.py --predict layer_depth325` over the two
pages concatenated:

```text
round 1: best pair layer_depth25 x prefetch_off at 47.3 deg cond 1.4  x=0.120 y=0.897
  signal: drift sample base_again moved (-0.14, +0.26) and -3.53 ms; no row lies inside it
out-of-sample layer_depth325: rows (+60.20, +15.45) predicted +21.11 ms, measured +44.76 ms, residual +23.65 ms/token (+10.45 % of a token)
round 2: best pair layer_depth25 x prefetch_off at 45.1 deg cond 1.4  x=0.224 y=0.837
  signal: drift sample base_again moved (-0.02, -0.28) and -11.70 ms; no row lies inside it
out-of-sample layer_depth325: rows (+60.22, +15.50) predicted +26.49 ms, measured +28.26 ms, residual +1.77 ms/token (+0.78 % of a token)
median x=0.172 y=0.867  spread x=0.104 y=0.060
ADMISSIBLE: y=0.867 is inside 0.788-1.301 ms, so the two rows price both reads
p*=0.199 against the six-link gate's L/S = 0.82/5.30 = 0.155 -> pays (+0.20 ms of a 226.2 ms token, +0.09 %)
```

*(Amended by §5az: that transcript is what the screen printed the day it was written, and §5az re-runs this very
page through #209's screen — it now exits **1**, because the same page's held-out row misses by 1.5× the width
the page itself cannot resolve, and the pooled `pays` above is not scored from a model that page refutes. The
`y=0.867` band call also loses its separation: round 3's drift puts ±0.805 ms on `y` where its digits put
±0.037, and the median sits 0.079 ms from the near edge.)*

Round 2 scored alone said the opposite: `p*=0.134 … does not pay (-0.10 ms …, -0.04 %)`. So the first
single-batch identification of `x` and `y` in the campaign does not settle §5ar's lever — it moves it across the
threshold and back depending on which round you read. `y` never moves across the band: 0.897 and 0.837 are both
inside 0.788-1.301 ms and agree to 7 %. The asymmetry is not luck, and the screen's own Cramer terms name it.

**Why `x` is the unstable one — the arithmetic, worked from the published rows.** With row 1 = `layer_depth25`
and row 2 = `prefetch_off`, Cramer over the [`:417`](../benchmark/tinytitan_price_pair.py#L417) solve gives, per
round, the two partials of each price with respect to each row's wall step:

| | det | ∂x/∂m₁ | ∂x/∂m₂ | ∂y/∂m₁ | ∂y/∂m₂ | x per ms common | x per ms of step **difference** | y per ms common |
|---|---|---|---|---|---|---|---|---|
| round 2 | 1241.4 | +0.01782 | −0.00964 | +0.03436 | +0.02662 | +0.0082 | +0.0275 | +0.0610 |
| round 3 | 1176.5 | +0.01816 | −0.00938 | +0.03619 | +0.02813 | +0.0088 | +0.0275 | +0.0643 |

`x`'s two partials have **opposite** signs, so a drift that lifts both rows' steps together barely touches it
(+0.008 per ms), and a drift that lands *unevenly* between them hits it at 0.0275 per ms of difference. `y`'s
partials have the **same** sign, so it is the opposite: nearly immune to a difference (0.008) and very exposed to a
common shift (0.062). Two consequences, both checked against the tables above:

1. The observed swing is the difference mode. Round 3 moved `layer_depth25` and `prefetch_off` 8.33 ms apart where
   round 2 had them 0.00 ms apart, and the round-2 partials predict Δx = +0.096 against an observed +0.104 — a 6-8
   ms difference between two arms' rates is worth about the whole of `x`.
2. The control's 11.70 ms cannot be read as common drift. Common-mode would have moved `y` by +0.75 ms, twelve
   times the 0.060 the two rounds actually differ by. It did not, because the sweep runs each arm **once, in a
   fixed order** — `base`, `prefetch_off`, `layer_depth25`, `layer_depth325`, `base_again` — so a machine that
   gets quieter during a round gives the later arms a head start the earlier ones did not get. Position is
   confounded with arm, and round 3's control is the fastest rate in either round (4.55 against a 4.42 best).

That is the finding, and it is a design statement rather than a result: **a blocked single-pass sweep can identify
the coefficient that is a sum of its rows (`y`), and cannot identify the one that is a difference (`x`)**, because
only the former cancels the drift the page's own control measures. §5ap's four-round interleaving worked at the
level of whole rounds; inside one page there is nothing to interleave.

**What §5aw predicted about its own bar, confirmed live.** Both rounds print `no row lies inside it` — the bar
inked nothing, in either page. §5aw's deviation 4 said a counter-keyed AND-bar is blind to drift that moves only
the clock, and round 3 is exactly that case: the control's counters moved (−0.02, −0.28), its wall moved
−11.70 ms, so every row cleared a bar set by counters that did not drift while the column the solve reads did.
The bar did its job on the page it was built from (§5aw: it stepped over `slots_64`, a row that lay inside the
control in all three coordinates) and does nothing on these pages. That is the honest scope of it, and #209 is
filed to put the drift term where it belongs — on the price's error bar, not on the row list. *(§5az shipped
that, and the term is 6-20× the digits on both of these rounds.)*

**The out-of-sample point, and why the screen's status does not speak for it.** Round 2 misses by
+23.65 ms/token (+10.45 %) and round 3 by +1.77 ms (+0.78 %) on the same held-out row. The residual is printed at
[`benchmark/tinytitan_price_pair.py`:703](../benchmark/tinytitan_price_pair.py#L703) and returned nothing; the
status came only from the band class of `y`. A page whose held-out point sits 6.7× its own drift away from the
prediction therefore exited 0 as `ADMISSIBLE`. The campaign reads that as the model being valid *interpolating*
between du = −42.65 and +33.04 and not extrapolating to +60.20 — round 2's miss is entirely in the extrapolation —
but the instrument did not say so, and #209 names that as its defect rather than as a measurement finding. *(§5az
shipped that: the residual is now scored against the page's own width and does move the status.)*

**What the lever is worth even when it pays.** +0.20 ms of a 226.2 ms token: **+0.09 %**, about +0.004 tok/s
against 4.42. With `x` bounded only to ±0.1 at best and the threshold sitting between the two rounds' answers, the
correct decision is not to spend a third and fourth round on it. *(§5az removes the `even when it pays`: re-run
with the drift in the bound and the residual scored, this page exits 1 and prints no gate answer at all, so the
+0.09 % is the size of the effect the model would have licensed had it held.)* Every I/O-policy lever the campaign has priced
since §5ar lands in this ±0.1 %-of-a-token class, and the reason is a bytes floor, not a hiding floor.

**The headroom bound, corrected.** The first version of this section bounded the feeder prize with `io_hidden_pct`
(99.7-99.9 % across the ten arms, hence "0.5-0.7 ms of a 226-248 ms token is all any feeder can reclaim"). That is
the move §5ap refuses on the record — *"deliberately not quoted … unfit for an overlap claim"* — and §5w says why:
the counter moves 0.06 points across a 12.2 % clock change, so it measures host blocking, not device idle. Against
the device ledger the same campaign publishes, the idle is **52.803 ms/token — a quarter of the token** — so the
first version understated the prize it was dismissing by 75-100×. The bound has to come from §5w's other line,
which is the one that survives: the gap is *predicted* from the read bill at the drive ceiling on five arms
(203.55 MB/token ÷ 3.92 GB/s = 51.93 ms against 52.803 measured, ratio 1.017), and the in-burst delivered rate is
97 % of that ceiling. *(Amended by §5ay: §5af retired both the 3.92 figure and this 97 % quotient — it argues the
gap credits the whole demand volume to one window while the ring's speculative reads fly in it. The rate that
survives is §5af's measured 3.3-3.4 GB/s wall, and the shipped reader is already at it, so "the pipe is flat out"
holds; what changes is that a byte costs the drive 0.299 ms, not 0.255.)* A constant feeder therefore cannot shorten
this wait by scheduling it better — the burst already runs at the device's own wall, and moving it earlier
relocates the wait rather than removing it. Only fewer bytes cut it, at **0.255 ms per MB** *(amended by §5ay:
0.255 ms/MB is what a byte costs the **drive** at the ceiling §5af retired; §5w's paired table measures what it
costs the **token** at 0.140 ms/MB, because ~45 % of a marginal read is already absorbed by the host-idle family —
use 0.140 for screening)*, and the prices below stay inside §5w's own 217.34 ms token so no clock crosses a batch:
+1 % of tok/s needs 2.15 ms, i.e. 8.4 MB/token — 3.1 of the 73.5 experts a token reads,
4.1 % of the bill; +5 % needs 19.9 % of it. *(§5ay re-prices both: 14.0 MB and 32.6 %, on the paired arms' own
198.71 ms token.)* That is a capacity question, and §5w's closing line already answered it —
*"No capacity route is left that does not need more RAM"* — under the operator's 12 GiB cap. So the
read path is closed on the byte floor and on the ceiling it is read at, not on a host-blocking percentage that was
never able to measure it. The overlap half of the question is separately closed too: reclaiming a window needs
something to consume it, and §5w measured that this idle "is not spare worker capacity". One extrapolation of §5w's
predictor to its own limit, stated as arithmetic on published figures and not as a measurement: take the bill to
zero bytes, so the gap goes too, and §5w's batch is 164.5 ms/token — **6.08 tok/s** (6.05 from the predicted gap;
the five arms put that predictor within ±6.4 %). *(Amended by §5ay: that step subtracts the whole gap from the
capture arm's 217.34 ms token, which combines two things the paired table says not to combine — §5w's deviation 1
excludes 4.601 tok/s from the published levels, and the gap does not convert 1:1 into the token. Staying inside the
paired arms gives a bracket: **5.90 tok/s** at the measured 0.140 ms/MB conversion, **6.96** only if the host-idle
family stays flat while the reads go, which the same table argues it will not.)* So 6 tok/s on this composition
means every routed expert byte gone, which is the RAM §5w says is not there, and 7 tok/s means cutting a further
21.7 ms out of what is left *after* the reads — the other half of §5w's ledger, not a feeder. *(§5ay states the
sharper version: 7 needs 55.85 ms, the entire read-bound gap is 55.11 ms, so the read path misses 7 by 0.74 ms even
in the branch where it is perfect and nothing re-expands.)* Nothing on the read path reaches the goal even at
its ideal, which is a stronger statement than the one the `io_hidden_pct` paragraph made and the one worth carrying
forward.

**prefetch_off's sign, settled against §5aw's round 1.** Round 1 read `prefetch_off` at +4.5 % over a baseline
that itself fell 9.9 % during the sweep. The two clean rounds read it at −6.0 % and −3.5 % with a 1.7 % and 5.4 %
drift, and its counters are identical in all three (0.00 issued, 0.00 adopted, `evict` 51086, `waits` 15188). The
predictive ring earns its place on the shipped tier; the round that said otherwise was the contaminated one, and is
withdrawn exactly as §5aw withdrew it for other purposes.

**Deviations.**
1. **Nothing here is a quality measurement.** All ten arms of both rounds streamed the same text
   (md5 `3c0c3391`), so this section prices read policy, not output.
2. **The two rounds disagree and both are published.** `p*` is quoted per round (0.134, 0.268) as well as pooled
   (0.199), because the pooled median of a two-element list is its mean and the mean hides the flip. A magnitude
   needs 3+ rounds per the screen's own line; this section does not claim one.
3. **The partials in the sensitivity table were computed for this section**, from the published rows and the same
   Cramer terms the screen uses, not read out of it — the screen printed `dy_per_ms` only
   ([`:460`](../benchmark/tinytitan_price_pair.py#L460), [`:980`](../benchmark/tinytitan_price_pair.py#L980)) and
   prints it as the single-row max, which is neither the common-mode nor the difference-mode number. The x
   sensitivities above are reproducible from the table of rates and counters. *(Amended by §5az: #209 added the
   difference-mode pair to that line — `difference-mode sensitivities 0.061 ms of y and 0.027 ms of x per ms of
   drift` — so the table's two right-hand columns are now on the page, and they are this section's numbers to the
   third digit. The common-mode columns remain computed here and not read out of the screen, because `_spread`
   adds widths independently by construction: for `y`, whose partials share a sign, the two are numerically equal
   (0.0610), and for `x`, whose do not, only the difference mode is printed.)*
4. **Round 2 was contaminated by this session, and less than round 1.** The machine was not idle in the strict
   sense: `Google Chrome Helper (Renderer)` was the top-CPU process at the first sample and the agent's own
   session was running, and swap carried 5.7 GiB of 7.2 GiB in from before. The mitigation was to go silent for
   the rest of each sweep; the 30 s sampler (`/tmp/trio206/round2.machine`) is the record of what was on the
   machine. Both rounds still passed their own drift gate.
5. **`/tmp/trio206/` is throwaway.** Rounds 1-3 logs and the precondition blocks are not committed; the committed
   artifacts are the screen, its suite and this section.
6. **A shipped error in this section, and what the correction changed.** §5ax's first version bounded the feeder
   prize with `io_hidden_pct` — 99.7-99.9 %, hence "0.5-0.7 ms … is the entire SSD-exposed budget any feeder can
   reclaim" — which is the exact move §5ap refuses on the record and §5w declares unfit (the counter moves 0.06
   points across a 12.2 % clock change). It understated the idle it was dismissing by 75-100× (52.803 ms/token on
   the device ledger) and argued from a host-blocking gauge to a device-side conclusion. The correction is in the
   commit that follows `c069115`; it replaces that one paragraph and its conclusion, which becomes stronger rather
   than softer (the byte floor plus the 97 %-of-ceiling burst rate put the read path's *ideal* at 6.08 tok/s — since
   re-priced by §5ay as a 5.90-6.96 bracket on the paired arms, so no feeder lever of any size reaches the goal). The
   two rounds' numbers, the sensitivity table, the `p*` verdict and
   the `prefetch_off` sign are untouched, and so are the other deviations in this list. A reader holding `c069115`
   alone should read the headroom bound as the byte floor, not as `io_hidden_pct`, and should take the ms-per-MB
   price from §5ay rather than from this section. The sibling uses of the same
   counter were checked and amended the same way, not rewritten: §5ao's "the demand reads are almost entirely
   hidden already" clause now carries an amendment saying which part of its verdict the counter was actually doing,
   and [`docs/plan-qwen38-decode-12gb.md:60`](plan-qwen38-decode-12gb.md#L60)'s falsifier bullet now names the
   device-side lines as the counter to pair. §5m's 0.075 % prize had already been amended by §5w itself, so this is
   the second time since that ruling the gauge was reached for as an overlap argument — the count is the finding.
7. **Section numbering.** `## 5av` does not exist in this document and no other document refers to it; the sequence
   runs §5au → §5aw → §5ax → §5ay. Recorded so a reader looking for a lost section does not spend a session on it.
8. **Provenance.** §5aw's screen landed at `fd3823e` and its notes at `7ebbba1`, both on `main`; both rounds ran
   against that tree, unmodified, with a binary newer than every source file. Gates seen locally before
   committing this section: `tools/lint.sh docs`, `tools/lint.sh python`, the CI markdown-link step, and the price
   screen's own suite. None of it is evidence a CI run executed them; it is evidence that the tree this section
   describes is the tree the next run will check.

## 5ay. #209's premise, priced from §5w's own paired table: a routed-expert MB removed buys 0.140 ms of token and not the 0.255 ms that `bytes ÷ ceiling` says, because 45 % of a marginal read is already absorbed — and the measured price puts 7 tok/s beyond the entire read-bound gap (2026-10-16)

Nothing here is a new run. It is arithmetic on the five arms §5w published, done because §5ax's corrected
paragraph — the one that replaced the `io_hidden_pct` bound — carried an assumption that §5w's own table
contradicts: it priced a byte at the drive ceiling (`1 ÷ 3.92 GB/s` = 0.255 ms/MB) and then subtracted the whole
gap from the token, i.e. it assumed every millisecond of read-bound device idle converts 1:1 into a millisecond of
token. §5w measured that conversion and did not price it; §5ax used 1.0 where the pair says 0.55.

### The conversion, read off the paired rows

Both §5w A/B rounds move the read bill and the clock together, so the token's price of a MB is a difference of two
published arms, and it is the **removal** direction that matters (48 → 96 slots cuts bytes):

| paired round | MB/token | ms/token | Δ bill | Δ token | **token price** | vs 0.255 | Δ gap | gap price | absorbed by family 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round 1 | 380.19 → 207.63 | 218.63 → 199.40 | −172.56 | −19.23 | **0.1114 ms/MB** | 43.7 % | −39.708 | 0.2301 | −20.48 ms, 0.1187/MB (46.5 %) |
| round 2 | 380.72 → 207.01 | 227.38 → 198.02 | −173.71 | −29.36 | **0.1690 ms/MB** | 66.2 % | −49.127 | 0.2828 | −19.77 ms, 0.1138/MB (44.6 %) |
| mean | | | −173.14 | −24.30 | **0.1402 ms/MB** | **55.0 %** | −44.418 | 0.2566 | 0.1163/MB (45.6 %) |

The gap column is the check that the two sections are describing the same physics. The read-bound idle moves at
**0.2566 ms/MB**, which is 100.6 % of the ceiling price §5w's predictor uses (`bytes ÷ 3.92 GB/s`, five arms inside
6.4 %) and **86 %** of §5af's planning wall of 3.3-3.4 GB/s — and §5af retired 3.92 as a *rate*, arguing that
`gap_total / gap_count` credits the whole demand volume to the hit→fixup window while the ring's ~42.8 speculative
reads per token fly in it too. The difference form is much less exposed to that critique than the absolute one, but
it is not exempt: on the planning wall, about 0.042 ms/MB of a marginal read's device time lands somewhere other
than this gap. So the bytes do cost drive time — most of it inside the gap — and what the token sees is 47-55 % of
it, and the missing part is exactly §5w's family trade: at 48 slots the host-turnaround family fell to
19.930/20.386 ms/token from 42.835, i.e. the deeper queue buried about 20 ms of host bookkeeping per round, and
−20.48/−19.77 is what the table needs it to be. The two halves are the same 20 ms measured from opposite sides.

The stable quantity is the absorbed share, not the price. The two rounds disagree about the token price by 52 %
(0.1114 vs 0.1690) and about the gap price by 23 %, but they agree about the absorption to 4 points (46.5 %,
44.6 %). That is what §5ax's sensitivity result predicts from the other direction: the wall clock is the noisy
column and the counters are near-deterministic, so the *difference* between the two idle families is the durable
number. **Use 0.140 ms/MB (mean) when screening a byte-cutting lever; the `bytes ÷ rate` figures — 0.255 at the
retired 3.92 ceiling, 0.299 at §5af's planning wall — say what a MB costs the *drive*, which is not what it costs
the user.**

### What the bill is worth, priced correctly

On the shipped 96-slot arm's own means inside that batch — 207.32 MB/token, 198.71 ms/token, 55.110 ms/token of
read-bound gap, 5.0325 tok/s — the read-driven share of a token is 207.32 × 0.1402 = **29.07 ms = 14.6 %**, not the
24 % the gap line alone suggests. Two independent routes reach the same figure: 74.9 misses/token (the paired arms'
74.99 and 74.76; §5w's capture arm read 73.51, and its deviation 6 says the two are not the same run) × 0.388 ms
per read (0.1402 × 2.7689 MB) = 28.5 ms, 2 % off. So the extrapolations are:

- **Bill → 0 at the measured price:** 198.71 − 29.07 = 169.64 ms → **5.90 tok/s** (per round 5.67 and 6.13).
- **Gap → 0 with nothing re-exposed:** 198.71 − 55.11 = 143.60 ms → **6.96 tok/s.**

§5ax published 6.08 for that second shape using §5w's *capture* arm clock (217.34 ms) rather than the paired arms'
own, and with no branch distinction. The bracket is the honest answer, and the pair decides which end applies: in
the measured 83 % of the bill, host idle re-expanded as the queue shallowed, so the lower end is the supported
extrapolation and the upper end requires that trend to stop exactly at zero. Either way the goal is out of reach on
this path, and by a margin worth stating plainly: **6 tok/s needs 32.04 ms and the entire bill buys 29.07 — 91 % of
the way there is gone before a single byte is saved; 7 tok/s needs 55.85 ms, which is 192 % of the bill, and the
whole read-bound gap is 55.11 ms.** Seven is short by 0.74 ms even in the branch where the reads vanish *and* the
host idle stays flat — a branch the same table argues against.

The screening prices that follow are 1.8× what §5ax printed: at the shipped level a **+1 %** gain needs 1.97 ms =
**14.0 MB/token = 5.1 of the 74.9 experts a token reads = 6.8 % of the bill** (not 8.4 MB / 3.1 experts / 4.1 %);
**+5 %** needs 67.5 MB = **32.6 %** of it (not 19.9 %). Every residency lever the campaign has ever screened has to
deliver several percent of the bill *per* quality-preserving change to be measurable at all, and §5w's closing line
— *"No capacity route is left that does not need more RAM"* — is now backed by a price rather than by a rate.

### Two prices this corrects in the same direction

1. **The ceiling is the optimistic one.** §5af measured the device wall at 3.3-3.4 GB/s across 30 slice × worker
   shapes and named 3.3-3.4 the planning number, superseding the 3.92 the `ParallelExpertReader` header records
   (1 thread 2.43, 4 threads 3.92, 8 threads 3.92 saturated —
   [`sources/TinyTitan/Infrastructure/Streaming/ParallelExpertReader.swift:9-11`](../sources/TinyTitan/Infrastructure/Streaming/ParallelExpertReader.swift#L9)).
   Against §5af's 3.35 GB/s the conversion is **47 %** instead of 55 %, so the 0.140 ms/MB is generous to the read
   path by both constructions at once. It is not a ceiling in the "best case" sense; it is the high end.
2. **§5ar's withholding lever is a token effect, not a drive effect.** Its band is 0.788-1.301 ms per *issued*
   read — service time — while a removed read is worth 0.388 ms of token, so the lever's published +0.05 % to
   +0.32 % should be read as roughly +0.03 % to +0.18 %. That is the same ±0.1 %-of-a-token class §5ax concluded,
   reached from the price side instead of the gate side, and it moves §5ar's lever further from the six-link
   threshold, not closer. Nothing reopens.

### Deviations, all of them

1. **No model run.** Every figure is arithmetic on §5w's published rows (arms `r1a`/`r1b`/`r2a`/`r2b` and the
   family totals), plus §5af's and §5ar's published prices. ms/token comes from `1000 ÷ tok/s`, which is how §5w
   itself states its 217.34 ms token.
2. **n = 2 pairs, over a 1.83× swing.** The price is two numbers 52 % apart around a mean. It is not a fitted
   curve, it carries no error bar, and the 83 %-of-bill coverage cuts both ways: the extrapolation to zero bytes
   is a 1.2× extension of a measured range, in the measured direction, but an extension still.
3. **The 12.2 % agreement is not independent evidence.** The paired slot advantage and the token price come from
   the same four arms, so "the prediction lands inside 6.4 %" and "the token moves 12.2 %" cannot corroborate each
   other. The only cross-check that is partly independent is the family balance (20 ms from the totals versus 20 ms
   from the difference) and the miss-count route to 28.5 ms, and both are within a few percent.
4. **The linear model is a model.** It assumes family 2 re-expands proportionally as bytes retreat. The mechanism
   §5w gives is queue depth, and queue depth is not linear in bytes all the way to zero — at some point the GPU
   runs out of already-queued work to hide the host behind and the absorption stops. That is the whole difference
   between 5.90 and 6.96 and this section does not resolve it; it publishes both ends and says which one the data
   reaches.
5. **§5w's own contamination applies to these rows.** Its deviation 3 records swap at 7.63 → 7.90 GB across the
   batch and calls round 2's slower 48-slot arm a plausible pressure effect, which is precisely the arm that sets
   round 2's higher price. If that arm was slow because of swap rather than bytes, round 2's 0.1690 is too high and
   the mean over-prices a MB.
6. **One correction per day, same section.** §5ax was fixed for using `io_hidden_pct` and is fixed again here for a
   conversion assumption; both are its own text, both are disclosed rather than silently rewritten. The
   distinguishing lesson: the first error reached for a gauge the campaign had ruled out; the second used a gauge
   it had ruled *in* (the ceiling predictor) for a quantity the same batch had already measured. Reading a
   published table is not the same as using it.

## 5az. #209 scores the page's own drift into the price bound and its held-out row into the status — and §5ax's first single-batch identification of `x` and `y` refutes itself on the page that published it, while the six-link gate turns out to need a baseline that repeats to 0.75 ms, which this machine has never given (2026-10-16)

§5ax closed the campaign's identifying sweep with `p*=0.199 … -> pays` and a design statement: a blocked single-pass
sweep can identify the coefficient that is a sum of its rows and not the one that is a difference. #209 spends no
new round — it spends the two already published, because §5aw left the drift sample doing exactly one job (choosing
which rows may be paired) while §5ax showed the same control moving a row 11.70 ms, a quarter of the token those
rows were being used to price. [benchmark/tinytitan_price_pair.py](../benchmark/tinytitan_price_pair.py) now reads
that movement as a width on the solve, scores the held-out residual it used to print and discard, and refuses any
`p*` whose own interval contains the threshold. Over §5ax's own page it answers a different question than the one
that section printed.

### The four rules, as shipped

1. **The drift is a width, not only a bar.** `Drift` ([`:134`](../benchmark/tinytitan_price_pair.py#L134)) carries
   the control's step in the column the solve reads and in both counters; `_solve` ([`:423`](../benchmark/tinytitan_price_pair.py#L423))
   feeds it through the same six Cramer partials the printed digits go through, as `w_drift` beside `w_digits`
   ([`:448`](../benchmark/tinytitan_price_pair.py#L448)). Sum-of-absolute-input-widths, i.e. the difference mode:
   two rows drift independently, so their errors add. §5ax derived that mode by hand because the screen printed
   only the single-row maximum; it prints it now (`difference-mode sensitivities 0.061 ms of y and 0.027 ms of x
   per ms of drift`), and the numbers land on §5ax's sensitivity table's two right-hand columns to the third digit.
2. **A bound names its provenance.** `Price` carries six half-widths, not two: `x_digits`/`y_digits`,
   `x_drift`/`y_drift`, and the totals `x_bound`/`y_bound` ([`:160`](../benchmark/tinytitan_price_pair.py#L160)).
   `_digits_line` ([`:647`](../benchmark/tinytitan_price_pair.py#L647)) prints the digits and nothing else;
   `_drift_line` ([`:664`](../benchmark/tinytitan_price_pair.py#L664)) prints the drift, then the total once, as a
   total. All three are rounded *up* at 0.001 ms ([`:395`](../benchmark/tinytitan_price_pair.py#L395)).
3. **A `p*` is an interval.** `pi_interval` ([`:688`](../benchmark/tinytitan_price_pair.py#L688)) runs the ratio
   from `(x − wx)/(y + wy)` to `(x + wx)/(y − wy)`, and if that reaches the gate's `L/S`, no gate line is printed
   ([`:772`](../benchmark/tinytitan_price_pair.py#L772)). `y − wy ≤ 0` is not a wide ratio but no ratio at all, and
   the refusal says `unbounded` rather than inventing an endpoint.
4. **The held-out row has a status.** `_off_sample` ([`:700`](../benchmark/tinytitan_price_pair.py#L700)) compares
   the residual to the width this page cannot resolve about that row — its own printed step, plus the control's
   clock, plus what the two prices' widths reach into the prediction
   ([`:1002`](../benchmark/tinytitan_price_pair.py#L1002)) — and a miss past that is `REFUTED OUT OF SAMPLE`, which
   withdraws the gate from a model the page itself falsifies
   ([`:853`](../benchmark/tinytitan_price_pair.py#L853)).

The statuses are the same three numbers, with the second and third widened, because a page can now solve a price
and still not be allowed to spend it:

```
0  the design identifies, `y` is inside the band, and the gate is drawn
1  it identified a price, and that price or a held-out row refutes the model
2  it could not solve, or its own widths cannot draw the gate, with the reason named
```

### The defect in this section's own first cut, disclosed before the results

The first version of #209 folded the drift into `y_bound` and `x_bound` and left `_digits_line` and the band caveat
printing those totals under sentences about *printed decimals*. On §5at's page that reads:

> digits: the page prints tok/s to 2 decimals (±0.005), so y is known only to ±0.866 ms

which attributes 0.832 ms of machine movement to 0.005 tok/s of rounding, and then prescribes "more printed
decimals is what would settle both" — a cure for a third of the illness that hides the other two thirds. It is the
same shape of error §5ax was amended for: a bound carried under a sentence that names the wrong source. Fixed
test-first, with three tests (`:604`, `:615`, `:624`), the third of them on the branch where the printing *is* the
whole bound, so the caveat cannot only ever be read in its drift form.

### §5ax's page, re-run

The two clean rounds through the shipped screen, `--predict layer_depth325`, **exit 1** (the screen numbers its
tables from 1; §5ax names the sweeps 2 and 3, so its round 2 is `round 1` below). Every number and verdict below is
the run's own text; `...` marks prose elided in the middle of a line, and the two `signal:` lines and the two
`designs:` lines are dropped whole:

```
round 1: best pair layer_depth25 x prefetch_off at 47.3 deg cond 1.4  x=0.120 y=0.897
digits: ... so y is known only to ±0.034 ms and x to ±0.016 ms
  drift: ... ±3.527 ms of wall, ±0.14 issued, ±0.26 adopted ... drift alone puts ±0.231 ms on y
         and ±0.104 on x where the digits put ±0.034 and ±0.016; the digits and drift together put
         ±0.264 ms on y and ±0.119 on x, and that is the bound this page may price by
round 2: best pair layer_depth25 x prefetch_off at 45.1 deg cond 1.4  x=0.224 y=0.837
digits: ... so y is known only to ±0.037 ms and x to ±0.016 ms
  drift: ... ±11.701 ms of wall ... drift alone puts ±0.768 ms on y and ±0.329 on x where the digits
         put ±0.037 and ±0.016; the digits and drift together put ±0.805 ms on y and ±0.345 on x
median x=0.172 y=0.867  spread x=0.104 y=0.060
ADMISSIBLE: y=0.867 is inside 0.788-1.301 ms, so the two rows price both reads
cannot decide: this page's bound on y is ±0.805 ms -- its digits ±0.037 ms and its own drift ±0.768 --
and it sits only 0.079 ms inside the nearer band edge (0.788 of 0.788-1.301 ms) ... printing more
decimals would not settle it, because the machine's own drift is the wider half of what this page
cannot resolve
out-of-sample layer_depth325: ... residual +23.65 ms/token (+10.45 % of a token) against 15.393 ms
this page cannot resolve: REFUTED OUT OF SAMPLE, ... by 1.5x the page's own width ...
out-of-sample layer_depth325: ... residual +1.77 ms/token (+0.78 % of a token) against 45.560 ms this
page cannot resolve: out-of-sample consistent ...
the six-link gate is not scored from a model this page's own held-out row refutes, however well its
two solved rows fit
```

Three verdicts §5ax published move, and they move in the direction of saying less: the band call `y=0.867 is
inside` now carries a *cannot decide* it did not have (±0.805 against a 0.079 ms margin, 10×); the pooled
`p*=0.199 -> pays` is gone; and the status is 1 rather than 0, on a page whose two solved rows still fit the band.
Nothing about the rows changed — the same rates, the same counters, the same arms. What changed is that the screen
is no longer willing to price a token by a bound it did not have.

### The refutation exists because one round happened to be quieter

The two rounds hold out the *same* row and disagree about whether that refutes anything. Their widths decompose:

| round | control's clock `dm` | row's own printing | prices' reach into the prediction | **width** | residual | verdict |
|---|---:|---:|---:|---:|---:|---|
| §5ax round 2 | 3.527 | 0.62 | 60.20×0.119 + 15.45×0.264 = 11.24 | **15.393** | +23.65 | refutes, at 1.5× |
| §5ax round 3 | 11.701 | 0.61 | 60.22×0.345 + 15.50×0.805 = 33.25 | **45.560** | +1.77 | cannot refute |

The reach is the term that multiplies: a held-out row at `du = +60.20` inherits sixty copies of `x`'s uncertainty.
Round 3's price widths are 3× round 2's because its control moved 3.3× further, so its width is 3× again, and a
model it can disprove only by a 45.6 ms miss — a fifth of a token — disproves nothing at 1.77 ms. That is not the
screen being conservative; it is the campaign's out-of-sample test existing or not existing depending on how quiet
the machine happened to be in a round that was already chosen for a different reason. **The test is bought with a
repeating baseline, not with decimals**, which is the same lesson §5ax drew for `x` itself, now arriving through
the residual.

### What the six-link gate would take, in milliseconds

On these rows (`med_x = 0.172`, `med_y = 0.867`, `L/S = 0.155`), printing a `pays` needs
`(0.172 − wx)/(0.867 + wy) > 0.155`, i.e. `wx < 0.038 − 0.155·wy` — and 0.038 ms is the loosest reading of that,
taken as `wy → 0`. The page's digits already cost `wx = 0.016` at two decimals, leaving 0.022 ms for the machine;
`x` moves at ≈0.030 ms per ms of control drift (0.104 at `dm = 3.527`), so:

- **a baseline that repeats within ≈0.75 ms** (0.33 % of a 226 ms token) scores the gate at 2 decimals;
- within ≈1.2 ms at 3 decimals, where the digits cost 0.002;
- **at no printed precision on this sweep**, whose quietest round repeated to 3.53 ms — its `x_drift` alone is
  0.104 ms, 2.7× the entire budget.

That replaces §5ax's "the correct decision is not to spend a third and fourth round on it" with a number and a
reason. The gate is not unspent because the effect is small (+0.09 % of a token, as that section says); it is
unreachable on this machine's clock at any printing precision the sweep offers, and more decimals on a page whose
drift term dominates is arithmetic performed on a digit that was never the constraint. §5ay's independent route to
the same place — a withheld read is worth 0.388 ms of token, so the lever's whole class is ±0.1 % — is now bounded
from the measurement side as well as the price side.

### The other pages, and the one status §5au published that no longer stands

The same §5ax page **without** `--predict` is **exit 2** and says so differently: with `layer_depth325` back in the
pool the page offers two conditioned designs, `y=0.897` inside the band and `y=1.404` above it, and §5aw's straddle
refusal fires — *nothing here is a measurement of `y`*. Whether the campaign's best page prices the adopted read or
refuses to was decided by which arm was held out, which is the sharpest statement of §5aw's "the spread, not the
selection, is the finding" that the sweep has yet produced.

§5at's six-arm cross-batch page stays **exit 2** on the same 0.796/0.570 straddle, and now prints why its own
control was there: digits ±0.034 against drift ±0.832 on `y`. Of §5au's four-page table, three keep their statuses,
and the fourth — the in-band one-decimal trio published as `ADMISSIBLE, p*=0.237 pays … exit 0` — is **exit 2**:
`x = 0.238 ±0.121` over `y = 1.004 ±0.283` makes `p*` run 0.091 to 0.498 straight across the threshold. Its own
digits, not the machine, do that one in, and the band call still prints. §5au's bound column is untouched (it is
the digits half) but the two refutations in its first rows now stand at ±0.147 against a 0.297 ms margin and
±0.107 against 0.284: they survive the drift term with about half the room they were published with.

The exit-0 path is still reachable and is pinned: `HOLD_OK` carries no re-measured baseline, so its bound is its
digits alone (1.694 ms), its held-out row lands inside that, and the gate is drawn and pays. That is also a seam
worth naming rather than burying — **a page with no control row is priced on its digits and can exit 0 on a much
narrower bound than a page with one** (#209's rule only fires on pages that repeat their baseline), so a reader
should note which kind of page any future `pays` came from. Whether the screen should refuse to draw a gate at all
without a drift sample is a design question this section does not decide.

*(Amended by §5be: it is decided, and the answer is no. The page described above — three rows, no re-measured
baseline, digits-only bound — now prints `ADMISSIBLE` and exits **2** with the gate unscored, so the exit-0 path is
pinned on `HOLD_OK` only because that fixture gained a zero `base_again`; §5be names which published statuses that
changes and which it leaves alone.)*

### Tests, gates and provenance

39 tests in [benchmark/test_tinytitan_price_pair.py](../benchmark/test_tinytitan_price_pair.py) — 30 at §5aw, nine
new in `DriftAndHoldout` ([`:579`](../benchmark/test_tinytitan_price_pair.py#L579)) — 104 OK alongside
`test_knob_sweep_verdict` and `test_docs_facts_suites` (95 before). Every test-file change is inventoried: the
module docstring gained an eighth "must not get wrong" and the widened status block; the `DOC1_IN` fixture comment
gained the interval that #209 derives from it; `holdout_page()` and its two fixtures are new test-side helpers;
`DriftAndHoldout` is new. Two tests had their expected **status** changed, from 0 to 2, and they are the two the
section is about —
[`test_one_surviving_design_prices_exactly_as_it_did_unbarred`:461](../benchmark/test_tinytitan_price_pair.py#L461) and
[`test_a_bound_wider_than_the_band_on_the_admissible_side_says_that_too`:555](../benchmark/test_tinytitan_price_pair.py#L555).
The production behaviour they pinned was the defect, so it is the tests that moved; each keeps its older assertion
(`ADMISSIBLE`, `cannot decide`, the price itself) so the band call is still pinned exactly as §5au wrote it, and
neither was rewritten to reach green. One assertion was re-pointed once more inside this section, when the
misattribution fix moved the sentence it reads: `test_the_bound_names_the_digits_and_the_drift_apart` now binds the
three quantities separately instead of one total, which is stronger, not weaker.

RED, as far as this session can attest: the nine drove the implementation and failed against the screen as §5aw
left it — no drift term, no interval, no residual status. The three provenance tests are the exception and are
disclosed as such: they were RED against *my own first cut of #209*, which had already folded the drift into the
totals and printed them under the digits sentence, so their failure was of a sentence rather than of a number. That
is the class of defect no pre-existing test could have caught, because the string it names was invented in the same
change that misused it.

Deviations.

1. **No model run, and no new measurement.** Every number above is the shipped screen's output over pages built
   from rows §5ax, §5at and §5au published; the runs are quoted verbatim from
   `python3 benchmark/tinytitan_price_pair.py --page … [--predict layer_depth325]`, with the elided spans marked
   `...` being the drift line's middle clause and the fixture prose, not values.
2. **The `/tmp/az209/` pages are throwaways** (`ax_two_rounds.md`, `at_six_arm.md`, `four_pages.py`), rebuilt from
   published tables, not committed and not proposed as tools; the committed artifacts are the screen, its suite and
   this section. §5ax's logs remain `/tmp/trio206/`.
3. **The 0.75 ms and 1.2 ms gate requirements are arithmetic on the screen's own printed widths**, at the §5ax
   medians, and they inherit its round-2 solve (`x=0.120`, `y=0.897`). They are a *bound on reachability*, not a
   measurement of drift: they say no page this quiet can score the gate, not that some page that quiet exists.
4. **`_spread` adds widths independently, so every bound printed here is a worst case.** Where two partials share a
   sign (`y`) the common and difference modes coincide; where they do not (`x`) the screen prints only the
   difference mode, which is why §5ax's common-mode `x` column (+0.008 per ms) is still not an output of the screen.
   A bound is not a distribution and nothing here should be read as one.
5. **Round 3 is the contaminated-of-a-different-kind round.** §5ax deviation 4 records Chrome and this session on
   the machine and 5.7 GiB of swap carried in; its control moving 11.70 ms is that record showing up as a width.
   The refutation therefore rests on round 2's quieter clock, which is the same round §5ax's `prefetch_off` sign
   came from, and both readings of that round are published.
6. **Provenance.** `#L149`-style anchors in §5au, §5aw, §5ax and §5ay were re-measured against this file as part of
   this commit; the four amendments they gained are pointers, not rewrites, and every one names what moved. Gates
   seen locally before committing: `tools/lint.sh python` ok (ruff 0.16.7, check and format clean, 203 scripts parse
   under 3.13), `tools/lint.sh docs` ok (87 documents, 101/105 python suites registered in CI), the CI markdown-link
   step run locally 145 checked / 0 broken, and `python3 -m unittest test_tinytitan_price_pair
   test_knob_sweep_verdict test_docs_facts_suites` — 104 OK in 0.747 s. One anchor written by this section itself was
   wrong on first draft (`:406` pointed inside a test body rather than at its `def`), found by reading the line each
   cited number names, and is `:404`. None of it is evidence a CI run executed them.

## 5ba. The page the campaign has been quoting all night cannot price anything — and the one ray it does carry says the operator's feeder direction is the *cheap* one (2026-10-16)

*Read the §5ba correction before quoting this heading: the refusal and the ray below stand, but the "cheap one"
clause rested on `io_hidden_pct`, which §5w rules unfit for an overlap claim, and is withdrawn. The ray is a shape
statement about an unstable page, not a price.*

§5aa closed the cache-change reading of the constant-feeder premise by handing the scheduling reading to another
instrument: "answering it needs the live paired A/B method, not more offline ranking". The engine names one such A/B.
[`RealForwardRunner+DecodeLayers.swift:280`](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift#L280)
still carries the comment "the prefetch gate that used them (`TINYTITAN_PREFETCH_MIN_MARGIN`) measured -4.2% and is
gone", and the run that produced that −4.2 % is section 8 of
[qwen38-decode-profile-2026-09-05.md](qwen38-decode-profile-2026-09-05.md) — the table this campaign has quoted for
"prefetch loses", for the withholding lever, and for the loss-per-speculative-read, and which #209's screen has never
been pointed at. No model run: this is two committed tables re-read.

**What ran.** `python3 benchmark/tinytitan_price_pair.py --page /tmp/ba209/page.md --baseline base`, the same with
`--column io_ms`, and a throwaway `/tmp/ba209/ray.py` over the same numbers. Both exits 2. The page is a transposition
of [the §8 table](qwen38-decode-profile-2026-09-05.md#L262): the source prints `tok/s` twice (two rounds) and the
counters once per arm, so the two round tables below carry identical `issued/pt` and `adopted/pt`, and its
`reads issued / adopted per token` cell is split into the two columns the screen reads. Nothing was invented and
nothing was dropped.

### The screen's answer, verbatim

```text
column wall  (ms/token stepped against base)
table 1: REFUSED: the widest pair here, gate06 x ring_rank, is 6.0 deg apart (condition 9.6), below the 20 deg a two-row solve needs; no prices
  signal: no drift sample (base_again is not on this page), so no row was checked
```

`--column io_ms` prints the identical refusal, because the angle is a property of the counter rows, not of the column
the step is read from, and this page carries no repeated baseline for the drift sample to find
([`_drift_sample:508`](../benchmark/tinytitan_price_pair.py#L508)).

### Why it refuses, in one line of arithmetic

| arm | Δissued | Δadopted | adopted/issued | its row's line |
|---|---:|---:|---:|---:|
| ring_rank (2 slots, rank order) | +57.0 | +4.2 | 0.0737 | −4.21° |
| probe-margin gate 0.03 | +17.0 | +2.7 | 0.1588 | −9.02° |
| probe-margin gate 0.06 | +7.8 | +1.4 | 0.1795 | −10.18° |

All three arms differ only in *how much* of one change they make: more reads, proportionally more adoptions. Their
lines span 5.97°, where the shipped rule asks for 20° and §5ax's identifying pair had 47.33° — a page that varies a
scalar prices a sum, and this one does not even vary that. So §8 is exactly what its own header claimed, a within-batch
comparison of rates, and never a price: it closes the *lever* only in the sense that its `prefetch off` row is faster
than its three prefetch rows. That is still evidence. It is not an `x` and a `y`, and this campaign has been reading it
as if it were both.

### The one price the page does carry, and the model's failure against it

Along its ray the page gives a per-issued-read cost `c = Δwall / Δissued`. Both rounds, against §5ax's medians
`x=0.172 y=0.867`, where the model's own `c` is `x − y·(adopted/Δissued)`:

| arm | Δwall r1 | Δwall r2 | c r1 | c r2 | c model | model Δwall | measured/model r1 | r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ring_rank | +40.65 | +43.07 | 0.713 | 0.756 | +0.108 | +6.16 | 6.6× | 7.0× |
| gate 0.03 | +29.90 | +18.59 | 1.759 | 1.093 | +0.034 | +0.58 | 51.3× | 31.9× |
| gate 0.06 | +23.96 | +13.32 | 3.072 | 1.707 | +0.016 | +0.13 | 187.5× | 104.2× |

Three things, none of them the October prices:

1. **`c` is not a constant, and its trend has the wrong sign.** It falls 4.3× as the feeder gets deeper
   (3.072 → 0.713 ms/read), in both rounds. `dm = x·du − y·adopted` with both prices positive makes `c` *fall* as the
   adoption ratio rises, and the adoption ratio here rises in the same direction (0.0737 → 0.1795). Measured `c` rises
   with it. A two-price model cannot express that, and no choice of `x` and `y` fixes it.
2. **The saturated arm prices below the drive floor.** 0.713 and 0.756 ms per issued read sit under §5ar's band low
   edge 0.788 ms — the time the band says one 2.641 MiB read cannot beat at the measured wall. Either those 57 reads
   overlapped on the drive or they were not all single-expert reads; §8 does not say which, and this section does not
   guess.
3. **The October prices do not transfer, by 7–188×.** `x=0.172, y=0.867` are the local slope of a *throttling*
   experiment run around 42–103 reads/token with Δadopted negative against a shipped ring; §8's arms *add* reads to a
   baseline with none. §5az already withdrew the gate drawn from those two numbers; this is a second and independent
   reason not to carry them onto another page.

### Where this leaves the operator's premise — in the one place this page is the only evidence

The premise is about *adding* reads while compute runs. Every October arm thins or removes reads; the September batch
is the only live A/B in the adding direction the campaign has. Read as a schedule instead of a price, its ray says the
per-byte cost of a speculative read is roughly its own transfer time when the queue is full (0.71 ms at 57 reads/pt)
and several times it when the queue is nearly empty (3.07 ms at 7.8). A constant feeder is the cheap shape of the same
bytes; a trickle is the expensive shape. That is the operator's intuition, priced in the direction it was stated, and
it is the first number tonight pointing *with* the premise rather than against it.

And by the counter the premise is actually about, the shipped engine already lives at the cheap end. `io_ms` is
*accumulated* read duration
([`COUNTERS`](../benchmark/tinytitan_knob_sweep.py#L257)), not critical-path time: the exposed part is
`totalExposedIoNanos`, written where a read's completion misses the layer that needed it
([`+DecodeMoE.swift:318`](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L318),
[`+DecodeMoEFinalize.swift:63`](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoEFinalize.swift#L63)),
and the summary prints the ratio as `io_hidden_pct`
([`+Diagnostics.swift:326`](../sources/TinyTitanLib/ServerModelSession+Diagnostics.swift#L326)). September's own
baseline reads `66 (19%)`, `67 (18%)`, `70 (18%)` hidden
([§2](qwen38-decode-profile-2026-09-05.md#L65)); §8's rows carry 64.5–86.3 ms/token of a ~172–215 ms token, which is
why that section could name exposed expert I/O as the limiting resource. The shipped October round reads
`io_hidden=99.8%` on `io_ms=101.7` of a 226.2 ms token, at 42.65 issued and 22.12 adopted per token — 51.9 % precision
against September's 7.4 %. So "the SSD can read constantly while compute is doing stuff" is not an untried idea: it is
the delivered state, by the engine's own instrument, and the two things it did not buy are bytes and prediction
quality — which is where §5ay and §5z already said the remaining tok/s lives.

*Amended by the §5ba correction (same day): the paragraph above is the move §5w forbids, and §5w's own table refutes
it.* The standing ruling is that **"`io_hidden_pct` is unfit for any future overlap claim"** (§5w's verdicts, doc line
1744), and the reason is in the same section (line 1553): the counter answers "did the **host** have to block", not
"did the **GPU** go idle", because the routed command buffer is event-gated on the device
([`+DecodeMoE.swift:374`](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeMoE.swift#L374)), so when
bytes are late the device waits, the host does not, and the token still reports as fully hidden. §5v was overturned
for making this move over a 0.075% prize, §5ax was amended for a 99.89% clause, §5ay's deviation 6 recorded the
pattern — and this section made it a third time, with §5w's refuting ledger already committed some 4,650 lines earlier
in this same file. So "the engine's own overlap counter says reads are now essentially never the reason a layer
waits" is withdrawn, and it was never what the data said. On §5w's device ledger the shipped 96-slot default carries
`moe_phase1_hit → moe_phase1_miss_fixup` = **52.803 ms/token** in its capture arm and 56.348 / 53.872 ms/token on the
two shipped-default arms of its falsifier batch, while `io_hidden_pct` across those four arms reads 99.84, 99.90,
99.86 and 99.85 — a 0.06-point spread over a batch whose clocks differ by 12.2% and whose gap line moves by 70%. A
quarter of every token *is* the device waiting for bytes, on the arm that reports 99.8% hidden.

The re-basing matters more than the withdrawal, because that paragraph was about the operator's premise and not only
about a gauge. The host-side half of the September-to-October contrast is real: §2's baseline was 18–19% hidden and
today's shipped round is 99.8%, and §5c already recorded what part of that jump is accounting — "the
`io_hidden_pct` jump is the accounting seeing the fixup CB complete inside the I/O window" (doc line 152), against a
wall-clock prize of +2.5%. What has not moved is the device half: 52.8 ms/token of GPU idle waiting for bytes, and
§5w's five-arm table showing the rate *inside* each burst (3.85, 3.68, 3.84, 3.95, 3.69 GB/s) essentially flat in
queue depth. Those arms varied the *demand* burst's size, not the marginal speculative read, so that table neither
confirms nor refutes this section's ray — it removes the only evidence the paragraph offered that the shipped engine
already sits at the cheap end. The premise's status is §5w's, unchanged and undisputed by anything here: the drive
does burst and wait, the wait is a quarter of the token, and the burst already runs at 97% of the measured ceiling, so
the way out is earlier bytes rather than a fuller or faster pipe. The ray stands as the shape statement it was written
as; the "delivered state" clause did not.

### Falsifier, and what this section refuses to claim

The falsifier written before running: if the screen solved §8 at ≥20° and put `y` inside 0.788–1.301 ms, the premise
would have had a price from the adding direction and this section would have carried it as a lever. It refused; the
ray reading above is what is left, and it is a *shape* statement, not a price.

1. **No clock.** No process was started and nothing here measures today's build. The 2026-09-05 and 2026-10-16 rates
   are never compared to each other — only each batch's within-batch deltas — per the rule that qwen38 clocks are not
   comparable across batches. The two headers say so themselves (5.8–6.1 against 4.4 tok/s is a different machine
   state, a different reclaim rule and a different I/O path).
2. **The `wall / Δio` ratios are not carried across batches.** Within September they read 1.94, 1.73, 1.72 (round 1)
   and 2.00, 2.48, 2.02 (round 2); the same construction on the October round reads +0.87, +0.95, +1.25 and −1.36 for
   the control. Those are not one quantity measured twice — the September column and today's `io_ms` are the same
   family of accumulated duration with different amounts of it exposed — so no amplification factor is claimed from
   them, and the six September ratios are reported only as the fact that its I/O counter moved about half as far as
   its wall did.
3. **That page's counters are not stable.** `gate03`'s I/O column moves 81.8 → 72.3 ms/token between two rounds of
   one batch — 9.5 ms on a counter that should be near-deterministic at identical work — and its wall moves −11.3 ms
   the other way. It is the same shape as §5ax's contaminated round, and it is a third reason no price is taken from
   §8's I/O column.
4. **Transposition loss.** §8 prints one counter pair per arm for two rounds, so the paired `du`/`dv` here cannot
   carry a per-round width; the screen's refusal therefore has no drift term in it (`no drift sample`), which is the
   unfiled §5az question about pages with no `base_again` arriving again on this page. It does not change the angle:
   geometry needs no widths.
5. **Throwaway.** `/tmp/ba209/` is not committed and needs no re-run: both source tables are in this repository, and
   every number above is arithmetic on the rows quoted at
   [§8](qwen38-decode-profile-2026-09-05.md#L264) and the two October logs' printed rows.
6. **One candidate this leaves open, named not claimed.** `io_hidden_pct` at 99.8 % is a large statement by a
   counter this campaign has never independently checked. Before it is used as evidence that exposure is solved, it
   should be read beside `wait_ms` and `io_host_waits` on the shipped default and against §5w's measured exposed wait;
   if §5w's wait is not ≈0.2 ms/token, the ratio is measuring a path the current engine no longer takes. That is a
   one-arm run, not an argument.

   *Amended by the §5ba correction:* withdrawn, twice over. First, it asked a host-side gauge to settle a device-side
   question — the move §5w rules out — and `wait_ms` and `io_host_waits` are both host-side too: `wait_ms` is
   `totalWaitNanos`, the host blocked on a GPU command buffer
   ([`+DecodeLayers.swift:262`](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+DecodeLayers.swift#L262)), and
   `io_host_waits` counts host waits, so the comparison was a category error before it was a forbidden one. Second,
   its expected value was wrong: §5w's wait is **52.803 ms/token**, not ≈0.2, so the test as written would have
   "failed" by predicting a number the campaign had already published, in this same file, four and a half thousand
   lines above it. The device-side instrument is already in the build and already what §5w published:
   `TINYTITAN_KERNEL_STATS=1` feeds `kernelGPUOccupancy()`
   ([`+Diagnostics.swift:70`](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift#L70)) and
   `kernelGPUGaps()` ([`:102`](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift#L102)).
   What genuinely remains open is a re-run of those two lines on today's build: §5w's ledger is from `8c61f14`, and
   two commits have touched `sources/` since — `9c858cd`'s use-count cap and `dcebe75`'s per-link depth knob, both
   off by default, so the shipped arm should be unchanged but has not been re-measured. That is a one-arm run, not an
   argument, and it is the run to ask for.

7. **One anchor the campaign has been quoting is stale.** §5w cites the event gate as
   `RealForwardRunner+DecodeMoE.swift:372` (doc line 1562, and §5v repeats it at 1459). It is on **374** on this
   tree — `dcebe75` put two lines above it. The code claim is unaffected
   (`routedCB.encodeWaitForEvent(token.event, value: token.value)` is still that file's only `encodeWaitForEvent`,
   verified by grep rather than by reading a remembered number), but a reader who opens 372 lands on
   `let ioStatus = …`, which is the same class of error §5az disclosed on its own first draft. §5w's text is left as
   published; this is where the drift is recorded.

## 5bb. The feeder priced on the device ledger: the shipped ring is worth +6.5 % / +8.8 % mirrored, and its prize closes to 0.4 ms/token (2026-10-16)

The run §5ba's own amendment asked for. §5w ruled `io_hidden_pct` unfit for an overlap
claim and published the device ledger instead; every feeder verdict since — #193's wall
A/B, §5az's "+14.72 ms/token of exposure" — has priced this ring on wall clock alone. This
section re-runs the two §5w lines on today's build and answers the question the host counter
could not: **what does the shipped speculative ring actually remove from device idle, and
does the byte floor predict it.** It does, and the ledger closes.

Provenance: release build `a8c6fdd`, clean tree, 24 GB M3 MacBook Pro (Mac15,3), macOS
27.0.1, Swift 6.4, AC power, 100 % charged. `models/qwen3.8-flash-next_125B_A6B_4Bit`, essay
prompt, `--max-new 384 --temperature 0 --max-context 262144`, no slot or budget flag — the
load line reads `ram profile cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G` and
the profile line reads `prefetch=1`, so this is the shipped default. Every arm ran with
`TINYTITAN_RUNNER_STATS=1 TINYTITAN_KERNEL_STATS=1`; the instrumented rates are **not**
comparable to §5p's uninstrumented 5.045 and are not quoted against any other batch.
Arms: `base1, off1, off2, base2` (control first and last, the two `off` arms adjacent in the
middle), each `off` arm setting `TINYTITAN_PREDICTIVE_PREFETCH=0`. Exit 0 all four.
**All four responses share one md5, `c172890df88cdbbfa0627bd320d16614`** — the ring is
numerics-neutral, so nothing here asks the model to be dumber.

| arm | ring | tok/s | busy ms/tok | top gap ms/tok | events/tok | ms/event | adopted/tok | hit | io_host_waits_avoided | io_hidden_pct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base1 | on | 2.863 | 202.151 | 58.970 | 29.95 | 1.969 | 19.58 | 0.8474 | 6871 | 99.71 |
| off1 | off | 2.688 | 208.310 | 75.493 | 34.57 | 2.184 | 0.00 | 0.8070 | 5080 | 99.67 |
| off2 | off | 4.086 | 118.916 | 74.679 | 34.59 | 2.159 | 0.00 | 0.8070 | 5080 | 99.88 |
| base2 | on | 4.444 | 115.156 | 56.698 | 30.10 | 1.884 | 19.20 | 0.8467 | 6815 | 99.86 |

`top gap` is `moe_phase1_hit → moe_phase1_miss_fixup`, §5w's largest line and the device's
wait on the experts the hit path assumed were resident.

Findings:

- **The ring is worth +6.51 % (round 1) and +8.76 % (round 2) on the wall, mirrored.** Paired
  per token: −22.71 ms and −19.72 ms. §5an's "≈6 %" was right and was a wall-clock guess; the
  direction is now confirmed on the device and the magnitude is a little larger.
- **The gap line moved, so the falsifier did not fire.** −16.523 ms/token in round 1, −17.981
  in round 2, on a line §5w measured at 52.803 on `8c61f14`. The withdrawn §5ba clause claimed
  reads are "essentially never the reason a layer waits"; this line *is* the reason a layer
  waits, and the ring shortens it by a quarter.
- **The byte floor predicts it.** Adoption measured 19.58 and 19.20 reads/token (§5az's trace
  said 22.12), at the campaign's established 2.7689 MiB per read = 56.85 and 55.74 MB/token;
  ÷ 3.92 GB/s → **14.50 and 14.22 ms/token predicted** (14.96 and 14.67 at the 3.80 burst
  mean). Measured 16.52 and 17.98: the line fell by 1.10–1.27× the floor, which is the
  arithmetic's own slop.
- **And the decomposition closes exactly**, which is the part worth re-deriving. The line falls
  for two reasons, and both are countable: whole fixup *events* are converted away (1774 and
  1721 fewer = 4.62 and 4.48/token, at the off-arm's 2.184 and 2.159 ms each → **10.09 and
  9.68 ms/token**), and the events that survive are individually shorter (29.95 and 30.10/token
  × 0.215 and 0.275 ms → **6.44 and 8.28 ms/token**). Sum: 16.53 and 17.96 against measured
  16.52 and 17.98. `io_host_waits_avoided` rises by 1791 and 1735, which is the same conversion
  counted on the host side of the seam.
- **The partition holds on today's build.** Round 2: GPU busy −3.760 ms/token and the six named
  decode gaps −15.543 = **−19.30**, against a wall step of −19.72; unattributed 0.41 ms/token
  (2 %). Round 1 leaves 1.81 (8 %), and its prefill rows sit inside the top-8 window with a
  5.30 s against a 3.00 s prefill, which is where the difference lives.
- **The ring pays, and the ledger names what it pays.** `shared_expert → moe_phase1_2_routed`
  grows +4.409 ms/token in round 2 (+6.828 in round 1) and its event count is the adopted count
  itself (6815 and 6871). Net of that, the device idle removed is −15.543, not the gross
  −17.981. Any future depth claim has to carry both terms.
- **Demand bytes really do disappear, not just move.** `expert_read_mib` falls 19849.6 MiB
  between the round-1 pair (51.7 MiB/token) for 7517 fewer demand misses — 2.641 MiB per
  avoided read against the 2.7689 stride, 4.6 % under, unexplained at the counter's 0.1 MiB
  rounding and not chased. The ring's own 39.5 issued reads/token appear in no read counter, so
  `expert_read_mib` is demand-only; a claim that the ring "reads the same bytes earlier" is not
  what these four arms show. **It reads fewer bytes, because the demand path never asks for
  them.**
- **With the feeder off the run is bit-reproducible; with it on, it is not.** The two `off` arms
  agree to the digit on every cache counter (hits 150131, misses 35901, read 94801.1, rdadvise
  33518 calls / 89012.8 MiB). The two `base` arms differ: misses 28384 vs 28527, adoption 19.58
  vs 19.20 — while `evictions` and `reloads` are identical in all four (31293 / 22986). The
  noise in this design sits entirely on the ring side, which is what a timing-dependent
  mechanism looks like in a counter, and why the prize is quoted per round.
- **§5w's ruling reproduced with a live mechanism.** `io_hidden_pct` spans 99.67–99.86, a
  0.21-point band, across arms whose device line differs by 18 ms/token and whose wall differs
  by 23 %. The host counter moved by nothing measurable while the thing it claims to describe
  moved by a quarter of the largest gap. Two independent falsifiers now say the same thing
  about that number.

What is left, priced. The line is still 56.698 ms/token = 25.2 % of a 225 ms token, and it is
30.10 events at 1.884 ms. Converting events is therefore worth ~2.17 ms of device idle each at
the off-arm mean (2.184 and 2.159), and issuing is 48.7–49.5 % adoptive (19.58 of 39.53, 19.20
of 39.46). A deeper ring's prize is
`(events converted × 2.17) − (added bytes ÷ 3.92 GB/s) − (the phase1_2 line's own growth)`, and
every term is now measurable offline, which is more than §5aq had when it measured deeper rings
loss-making on wall clock alone. That is the screen to run next; the ceiling named here is what
the whole remaining line is worth, not a claim that any of it is reachable.

Deviations:

1. `TINYTITAN_KERNEL_STATS=1` was on, so all four walls are instrumented. §5w's ledger was too,
   but on `8c61f14` at a different run length, so §5w's 52.803 ms/token is context, not a
   paired control.
2. **Round-to-round drift of 52–55 %** — base 2.863 → 4.444, off 2.688 → 4.086, prefill 5.30 /
   3.85 s → 3.02 / 3.00 s, GPU busy 202 / 208 → 119 / 115 ms/token. The machine changed state
   between the rounds by more than the effect under test, and the likeliest cause is this
   agent's own session, which was active through round 1 and idle through round 2 (the
   documented noise source). The paired deltas survive it (+6.51 %, +8.76 %) because both arms
   of each pair moved together; nothing here pools the two rounds.
3. Swap was carried in, not created by these arms: 4.01 GB used of 5.12 GB before `base1`, the
   pool grown to 10.24 GB by `base2`. `memory_pressure -Q` read 75–78 % free across all four
   arms, which is why the drift is attributed to contention rather than to the working set
   crossing §5f's cliff.
4. Adoption (19.58 vs 19.20) is the moving part, so the byte-floor prediction is a range
   14.5–15.0 ms/token, not a point.

## 5bc. #213 runs §5bb's named screen: the byte-floor bar was the wrong price, the exposed-idle band is 11.93–33.52 %, and on two live traces slots 0–2 pay under every reading while slot 3 is undecidable offline (2026-10-16)

*Read the §5bc correction below before quoting this heading. The band correction, the reproducibility
findings, the adoption bias and the duty table stand — they are depth-1 and byte statements. The title's
verdict on slots 1–2 does not: this screen's extrapolation past depth 1 was already falsified on live arms
by §5ak (doc line 3854), and §5al priced the identical read pool at −6.20 % (doc lines 3980, 4046).*

§5bb ended by naming the next screen: *"(events converted × 2.17) − (added bytes ÷ 3.92 GB/s) −
(the phase1_2 line's own growth) … That is the screen to run next."* It is run, and the first
thing the run found is that the formula's three terms do not divide by the same denominator —
which is a correction of §5bb's own published text, recorded here rather than edited into it.

Provenance. Two model runs, release build `.build/release/TinyTitanCLI` stamped 2026-10-16
14:37:28; `find sources -newer` against it is empty and **no commit between `a8c6fdd` and HEAD
`9df2973` touches `sources/`** (`daf3817`, `9df2973` add `benchmark/` python, `d2e3e95` a doc
row), so the capture ran on the same engine as §5bb's four arms. 24 GB M3 MacBook Pro
(Mac15,3), macOS 27.0.1, Swift 6.4, AC power, 100 % charged, one model process at a time,
`pgrep` guard empty before and after. `models/qwen3.8-flash-next_125B_A6B_4Bit`,
`--max-new 384 --temperature 0 --max-context 262144`, no slot or budget flag; both arms set
`TINYTITAN_PREDICTIVE_PREFETCH=0 TINYTITAN_RUNNER_STATS=1 TINYTITAN_PREFETCH_TRACE=<file>`,
and `TINYTITAN_KERNEL_STATS` is **off** because this batch needs no device ledger. The load lines
read `ram profile cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G` and
`profile … prefetch=0`. Exit 0 both, 18 384 records each (383 positions × 48 layers, 18 001
scored links). essay 21:51:28–21:53:14 UTC, code 21:53:14–21:54:56 UTC; `memory_pressure -Q`
82 % then 77 % free, swap used 7.23 → 8.57 → 8.62 GiB of a pool the system grew from 8 192 to
9 216 MiB between the arms. Prompts: §5bb's essay line, and a second one chosen to be a
different access pattern — `Refactor this Python function to avoid the repeated file read, then
write tests for it: def load(path): return open(path).read().split()`. Response md5s: essay
`c172890df88cdbbfa0627bd320d16614` (the same hash all four §5bb arms printed), code
`b004f8f2d0ccc0055da6f9e3e55d6e9d` (single arm, so no invariance pair for it). **No wall clock
from this batch is quoted anywhere** — the session was active while it ran; the driver at
`/tmp/bc213/driver.sh` records that as its own header, and the traces and logs are not committed.

### The price, corrected

`PriceBook` ([benchmark/tinytitan_prefetch_trace_analysis.py:168](../benchmark/tinytitan_prefetch_trace_analysis.py#L168))
now holds one round's ledger and derives the two prices a depth decision needs, paired **within
a round**. `daf3817` had paired them across kinds, and both errors are named in `9df2973`:

| price | what it is | round 1 | round 2 |
| --- | --- | ---: | ---: |
| prize, countable | events converted × ms/event ÷ adopted reads | 0.5153 ms/read | 0.5038 ms/read |
| prize, whole line | measured gap-line fall ÷ adopted reads | 0.8439 ms/read | 0.9365 ms/read |
| cost, exposed | ring's own line growth ÷ issued reads | 0.1727 ms/issue | 0.1117 ms/issue |
| cost, drive busy | 2.7689 MiB ÷ 3.92 GB/s | 0.74066 ms/read | 0.74066 ms/read |

The two prize readings × the two rounds give four break-even precisions: **33.52 %, 22.18 %,
20.47 %, 11.93 %** — the pessimistic corner is round 1's countable reading and the optimistic
corner is round 2's whole-line one.

Two things that were wrong: 2.17 ms is the value of a converted **event**, and only 23.6 % of
adopted reads convert one (4.62 of 19.58), so charging it per read inflated the prize ~4×; and
0.74066 ms is the drive's **busy** time, which §5bb itself showed is not what a read costs —
the same arms moved `io_host_waits_avoided` by 1 791 and 1 735 while `io_hidden_pct` never left
99.67–99.86, i.e. the host was not blocked. **The exposed cost is 4.3–6.6× smaller than the
byte floor.** The break-even precision is therefore a band, not a bar: **11.93 %–33.52 %**
([:219](../benchmark/tinytitan_prefetch_trace_analysis.py#L219)). The wrong bar, 34.15 %, landed
0.6 points under the corrected pessimistic corner, which is why none of the earlier fixture
verdicts showed the error. `slot_report`
([:244](../benchmark/tinytitan_prefetch_trace_analysis.py#L244)) reports both corners per issue
index, and `first_loss_making_slot` / `first_unproven_slot`
([:272](../benchmark/tinytitan_prefetch_trace_analysis.py#L272),
[:280](../benchmark/tinytitan_prefetch_trace_analysis.py#L280)) are the stop rule and the
"honesty" flag: a slot between the corners is *undecided*, and the screen says so instead of
choosing a side. 11 tests OK (`python3 -m unittest`, serial), `tools/lint.sh python` ok on ruff
0.16.7; RED observed as 4 errors (`KeyError: drive_busy_ms_per_read`, `break_even_high`,
`pays_under_every_reading` ×2). Two of the eleven tests were deleted and one renamed, all inside
the block `daf3817` added — no pre-existing test was edited or reformatted, verified by hunk
range. The falsifier on the instrument (`test_the_shipped_depth_clears_its_own_band`,
[benchmark/test_tinytitan_prefetch_trace_analysis.py:138](../benchmark/test_tinytitan_prefetch_trace_analysis.py#L138))
requires the shipped ring's own measured adoption (48.7–49.5 %) to clear the high corner; it
does, so the band prices an arm the machine already proved profitable rather than a phantom.

### The two traces, scored

`recorded` ranking (the engine's own `next_layer_prediction`), `filter` policy (the shipped
filter-then-cap rule, §5ak), `top_m = 4`, 18 001 links. A *slot* is the issue index within one
L→L+1 link, so "depth d" here means the ring may stage the first d non-resident names of the
plan — the same axis as the shipped `prefetchDepth`
([sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift:124](../sources/TinyTitan/Runtime/Configuration/RuntimeConfiguration.swift#L124),
currently 1 at this tier).

| slot | essay precision | essay net ms/tok worst → best | code precision | code net ms/tok worst → best | verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 (shipped) | 52.23 % | +3.840 → +15.032 | 55.09 % | +4.486 → +16.312 | pays under every reading |
| 1 (depth 2) | 41.39 % | +1.200 → +8.165 | 43.90 % | +1.676 → +9.379 | pays under every reading |
| 2 (depth 3) | 34.01 % | +0.052 → +4.191 | 37.83 % | +0.484 → +5.277 | pays under every reading |
| 3 (depth 4) | 29.85 % | −0.243 → +2.155 | 32.01 % | −0.111 → +2.683 | **inside the band: undecidable** |

`first_loss_making_slot` is `None` on both traces and `first_unproven_slot` is `3` on both. Read
against the band corners (33.52 % / 11.93 %): the marginal precision ladder crosses the
pessimistic corner **between slot 2 and slot 3**, on two different access patterns, and nothing
crosses the optimistic corner. The screen's stop rule therefore puts the useful width of the
feeder at **three reads per link**, not one — a claim §5aq could only reach on wall clock.

**Amended (same day, after publication).** That paragraph is withdrawn for slots 1 and 2 — the
"pays under every reading" verdict on them and the "three reads per link" width with it. The reason
is not that the arithmetic is wrong; it is that **slot 1 is not a new arm.** §5al already scored the
same pool on the same trace: *"Slot 2 pooled over the trace is 4 692 useful against 11 337 extra
reads = 0.4139"* (doc line 3980), which is this table's slot-1 precision 41.39 % to the digit, and
11 337 ÷ 384 = 29.52 reads/token, which is §5al's uniform-depth-2 row exactly (doc line 4046). §5al
prices that arm at **−6.20 %**; this band prices it at **+1.200 → +8.165 ms/token**. One of the two
is wrong about an arm the machine has already run, and the tie-breaker is not a model.

The tie-breaker is §5ak, which ruled on this sentence before this section existed: *"The screen's
depth-1 row validates; its extrapolation to depth > 1 is **falsified by the engine**"* (doc line
3854), on a live ladder measured **M=1 +12.2 %, M=2 +6.0 %, M=4 −9.8 %** against one control in one
batch (doc lines 3867-3870). It also named the term the band cannot see: the drive is saturated
while it reads, so *"a second speculative read steals service from the demand burst it was meant to
anticipate and lands too late to be adopted"*, and *"The `useful` counter has no arrival time in it,
so depth beyond one is outside what the screen measures"* (doc lines 3872-3875). §5al found the same
term from the price side: its optimistic fit reproduces the depth-1 and depth-2 anchors exactly and
then predicts **−15.1 %** against a **measured −9.8 %** at depth 4, because *"a constant per-read
cost under-prices the third and fourth reads, because each one contends longer with the demand
burst"* (doc lines 4034-4037). This band is the **fifth** curve on that trace to say depth 2 and 4
beat depth 1 — §5ak's three priced ladders (doc lines 3858-3866) plus §5al's optimistic fit are the
first four — and a fifth model of an axis is not evidence against the arms that closed it.

Where the band's cost term comes from is the whole error. `ms_per_issued_read`
([:189](../benchmark/tinytitan_prefetch_trace_analysis.py#L189)) is the ring's own device line growth
divided by its issues **on the shipped depth-1 arm**: one read per link, issued one layer ahead, most
of it landing inside the window. Slot 1's marginal read is charged that in-time price because the
screen has no other, and §5bb's ledger — which is what the band is built from — never contained a
depth-2 arm. **What this section keeps:** the correction of the two denominators, which is a depth-1
statement and a real one (2.17 ms is per converted *event*, not per read, and 0.74066 ms is drive
*busy* time, not a cost the host pays); the byte-reproducible capture; the +2.7–3.6-point adoption
bias, which if anything understates slot 1 since a read issued later must miss more often; and the
duty table, which counts bytes rather than prizes and so is untouched by any of this. The depth lever
stays where §5ak left it: *"the depth lever is closed by the engine's own paired series, and this
section adds nothing to it."*

### Three cross-checks, and what they say about the instrument

- **The capture reproduces a frozen file to the byte.** `/tmp/bc213/essay.jsonl` and
  `/tmp/trace194/off.jsonl` — the trace §5ak–§5am were built on, md5
  `67d221e2546baca4b26d1e66aaa360ae` as §5am recorded it — are **identical**, 18 384 records,
  despite #194 having been captured at HEAD `cd229f6` on the 04:29 build and this one on the
  14:37 build. So a greedy ring-off route trace is reproducible across builds as well as across
  arms, and the slot tallies match §5ak's published table exactly at every `top_m`
  (15 257 / 26 594 / 34 356 / 39 274 issues, 0.5223 → 0.4269 precision). **The consequence for
  this section is a limitation, and it is stated as one: the essay arm is not new routing
  evidence.** The load-bearing new datum is the code prompt, which is hungrier (hit 0.7719
  against 0.8070, 43 039 misses against 35 901 = 112.4 demand reads/token against 93.7) and
  still lands within 2.9 points of the essay ladder at every slot.
- **The replica prices the shipped arm against the engine.** At `top_m = 1` the screen issues
  15 257 / 18 001 = 0.8476 reads/link = **39.84/token** against §5bb's measured **39.53**
  (+0.78 %), and calls **52.23 %** of them useful where the engine adopted
  **49.53 % / 48.66 %** — an upper-bound bias of 2.7–3.6 points, the same size and sign as
  §5ah's ~7 % "right but late" pool counted as a share of the useful set. Every slot verdict
  above inherits that bias, so slots near a corner are exactly where it matters — and slot 2, at
  34.01 % against a 33.52 % corner, is near a corner.
- **The trace is a clean instrument.** The essay arm's counters are bit-identical to §5bb's two
  untraced ring-off arms (hits 150131, misses 35901, evictions 31293, reloads 22986,
  `expert_read_mib` 94 801.1, `rdadvise_calls` 33 518 / 89 012.8 MiB,
  `io_host_waits_avoided` 5 080) even though those arms carried `TINYTITAN_KERNEL_STATS=1` and
  this one carried the trace instead, and the response hash is §5bb's. Per-miss bytes come out
  at 2.6406 MiB on both prompts — §5bb's 2.641 again.

### The falsifier, and why it was weak as written

Registered before measuring: *"if no depth in the captured traces holds marginal adoption
> 32.5 %, the ring is already at its optimum and the remaining 56.7 ms/token is not buyable with
earlier bytes — the next prize must be sought elsewhere (host turnaround family)."* It did not
fire: slots 0–2 clear 32.5 % on both prompts. **It could not have fired unless even the shipped
slot failed**, since slot 0 is at 52 % on any reading, so the probe tested something already
known. The informative version is the ladder-against-the-corners statement above, and the live
`daf3817` message's claim that *"no `TINYTITAN_PREFETCH_TRACE` capture exists on disk"* was
simply false — the shipped screen has been scoring `/tmp/trace194/off.jsonl` since §5ak. Caught
while scoring, disclosed here, no verdict of that section depended on it.

### Reconciling with §5aq, which measured depth 2 loss-making

§5aq priced `layer_depth25` and `slots64_depth25` at **−8.1 % / −12.1 %** wall against the
shipped baseline; this screen says slot 1 pays +1.2/+1.7 ms/token even on the pessimistic
reading. They are not the same arm — §5aq's depth was applied to **25 selected links** (gate
0.45, §5al) while the screen's slot 1 is a read on **every** link — but the sign cannot be
reconciled by selection alone, because uniform depth 2 was also retired. Two candidate terms,
one of which is checkable offline:

1. **Lateness grows with width.** §5z found the shipped ring *"issuing 2× the depth it keeps"*
   with `inflight = 0.01` (doc lines 2229, 2282). A second staged read lands later relative to
   its fixup than a first one, so the marginal read's conversion rate is not its precision; the
   replica prices precision and calls it the prize.
2. **The ledger's cost term is fitted at depth 1 and extrapolated to depth 2.** `+6.828 /
   +4.409 ms per token` over the ring's own line is one round's slope; if the line grows
   superlinearly once `free.count` truncation and eviction bite, the band's corners move and the
   ladder's crossing moves with them.

**Amended:** neither term was open. Term (1) is §5ak's named mechanism (doc lines 3872-3875) and
§5al's depth-4 falsifier (doc lines 4034-4037, −15.1 % predicted against −9.8 % measured); term (2)
is §5al's optimistic fit itself, solved from the M=1 and M=2 anchors, reproducing both exactly, and
wrong by 5.3 points one rung up. §5ap's live gate-0.45 arm (**−8.4 %** over four interleaved rounds)
is the closest of the three, and its own conclusion is the sentence this band should have been
written against: *"the cheaper-than-predicted traffic still cost 8.4 %, because the loss is not in
the bytes, it is in adoption"* (doc line 4631) — the counters showed 33 extra reads a token *removing*
10.5 adopted reads, one for each demand hit the arm loses (doc line 4536). §5aq's −8.1 % / −12.1 %
are further live data on the same side. The sign was settled before this section ran; what this
section added was a fifth model that re-opened a settled question, and the section above shows why
the models lose here.

**Blind spots, named:** residency is held fixed at the ring-off world, so the eviction a deeper
ring causes is invisible to the screen; lateness is not modelled at all; and the band is two
rounds' worth of one build's one prompt pair, not a fitted curve.

### The operator's premise, answered in duty rather than in silence

The overnight brief asks why the drive reads *"for a couple of seconds, then stops"* instead of
feeding constantly. At the campaign's own prices (2.7689 MiB/read ÷ 3.92 GB/s = 0.74066 ms of
drive time each, against §5bb's 225 ms instrumented token):

| feeder | drive time/token | duty |
| --- | ---: | ---: |
| demand only, ring off (essay) | 69.4 ms | **30.9 %** |
| demand only, ring off (code) | 83.2 ms | 37.0 % |
| shipped depth 1 (issued 39.53 + demand left) | 84.2 ms | 37.4 % |
| depth 4 replica (issued 102.5 + demand left) | 113.0 ms | 50.2 % |

So the drive is **not** the constraint on a constant feeder: at the shipped configuration it is
idle roughly two thirds of every token, and there is room to read continuously — the shipped
depth-1 ring is already using part of that room, and a depth-4 feeder would take half the token.
The binding question is not bandwidth but whether a *further* read converts device idle, which
is what the band prices. One clarification of §5bb's "reads fewer bytes": **demand** bytes do
fall (19 849.6 MiB between §5bb's round-1 pair), and **total** bytes rise — depth 1 moves
×1.212 of the ring-off traffic, and depth 4 ×1.63. A feeder that "keeps reading while compute
runs" spends bytes to buy idle, and this table is the price of the bytes.

Deviations.

1. **No wall clock, no paired control, one round per prompt.** This is an offline scoring
   exercise over two traces; the interleave-and-mirror rule does not apply to a batch that
   reports no rate. Both arms ran while this agent's session was active, which is why the driver
   header forbids quoting their rates.
2. **The essay trace duplicates a file already priced.** §5ak–§5am scored the identical bytes,
   so the essay column is a reproduction check, not a second measurement. The cross-prompt claim
   rests on n = 1 for the code prompt, single session, no mirrored repeat.
3. **The band is extrapolated.** Both corners come from §5bb's depth-1 arms; every slot ≥ 1 is a
   counterfactual width that build never ran. Slot 2's 34.01 % is 0.5 point above a corner that
   is itself a two-round mean of a slope fitted at a different depth.
4. **Replica adoption is biased high** by 2.7–3.6 points against the engine's own counters, so
   the ladder's true crossing is at or left of slot 2, not right of it. That cuts against the
   section's own headline, and is stated as the risk rather than as a caveat.
5. **The trace's `misses` are post-adoption**, so both arms were run with the ring off on
   purpose (§5ah). The probe still records `next_layer_prediction` because
   [sources/TinyTitan/Runtime/Inference/RealForwardRunner+CommandBuffers.swift:10](../sources/TinyTitan/Runtime/Inference/RealForwardRunner+CommandBuffers.swift#L10)
   enables it whenever the trace fd is open; `next2_layer_prediction` is empty in all 18 384 rows
   of both captures, as in #194's, because nothing sets the two-layer probe without the ring.
6. **Swap was carried in, not created by these arms** (7.23 GiB used before the first launch,
   pool grown by the system between them), and `memory_pressure -Q` fell 82 % → 77 % free.
   Counters are bit-reproducible under that pressure, which is the finding, not the exception.
7. Nothing in `sources/` changed; no numerics, default or runtime behaviour moved. The two
   python commits it cites add `benchmark/` code and tests only, so `tools/golden-baseline.sh`
   is **not re-run** for them and is **not** needed by them; the golden baseline is **not
   checked** by this section, which is docs-only.

Next, offline and before any further live arm: give the residency simulator
([benchmark/flat_pair_cache_sim.py:79](../benchmark/flat_pair_cache_sim.py#L79)) the ring's
staged reads as well as the demand line, so a depth-d counterfactual replays against a pool that
the ring itself occupies. That is the only way to price candidate term (1) and (2) without a
model run, and the prefetch trace already carries both lists it needs (`experts` and the
per-slot issued set), so no new capture is required. If the evicted-slot term swallows slot 1's
+1.2 ms/token, this section's ladder is wrong and §5aq stands; if it does not, then the depth
question is ready for one paired live arm on the 25-link and uniform widths in the same batch.

**Amended, and this paragraph is withdrawn with the ladder.** A residency replay prices the pool
occupancy a deeper ring causes; it does not price arrival time, which is the term §5ak says the
trace cannot carry (*"The `useful` counter has no arrival time in it"*), so running it would have
produced a sixth offline depth curve over an axis the engine closed with one batch and one control.
Task #214 — screen the lateness term offline — is retired for the same reason: an honest offline
lateness price needs a per-read arrival timestamp, the trace has none, and adding one is a change to
the engine's instrument rather than to `benchmark/`, made to re-open a lever that §5ak, §5al, §5ap
and §5aq all closed in the same direction. Nothing it could screen is open, so it is not worth a
`sources/` change.

What is left of the feeder question is therefore slot 0 only — the shipped depth, priced correctly
for the first time by the band above — and the campaign's one positive-side candidate, §5ar's
withholding arm: a six-link depth-0 list that transfers out-of-sample at S/L = 6.44 and is worth
+0.05 % to +0.32 %, which §5ax and §5az then found the instrument cannot settle on this machine
because `x` is a difference of two solving rows' steps. The duty table's answer to the operator's
premise stands regardless: the drive is idle about two thirds of every token at the shipped
configuration, and filling that idle with *more* reads is not the lever — §5ap's counters say extra
reads cost adoption rather than buy it.

## 5bd. The 15 % bar was never this Mac's idle — the utilisation key separates a loaded host from this one by 35 points, so it is an instrument with the wrong zero, and §5s's missing "record the contention" half is now shipped (2026-10-16)

§5s ended on a sentence with a hole in it. The wiki-protocol arm refused at 54 % (doc line 1149),
and the note that followed said its threshold "is binary (15% or off)" and that there is no
"record the contention and carry on" mode, "which is the right design for a published table"
(doc line 1162).
§5v had already written the reader off: *"The `ioreg` AGX "Device Utilization%" reader showed 61%
with the machine idle and 69.6% mean during decode, so it is not an occupancy instrument on this
host and no conclusion here rests on it"* (doc line 1513). Those two statements are about one number,
and only one of them survives being tested.

**Falsifier, registered before measuring.** From the throwaway probe that preceded the tool:

> the gauge has signal only if the mean of the busy-window readings exceeds the MAXIMUM idle-window
> reading by more than the idle spread (max-min over the idle windows). If a saturating load lands
> inside the idle band, the key carries no information about contention on this host, and the gate
> that reads it cannot certify a clean machine at all.

**Measured.** Five windows, three of them on the shipped tool, from a clean tree at `05b9ff9`, AC
power, no model process, one heavy process at a time.

| window | `Device Utilization %` | load proof |
|---|---|---|
| gate0's own 30 s sample, §5s's refused arm (2026-10-15) | held 43–68, refused at 54 | — (the arm never started) |
| probe idle windows (2026-10-16) | 0–68, max 68 | — |
| probe loaded windows (2026-10-16) | 71–97, mean 94.8 | 1 830–1 858 GFLOP/s, 2048³ fp32, `mx.eval` per matmul |
| shipped gauge, no load | floor 60–71 median 65 → run 59–69 median 63, Δ −2 → **`indistinguishable`** | — |
| shipped gauge, with load | floor 56–71 median 64 → run 48–99 median 99, Δ +35 → **`elevated`** | 2 107 GFLOP/s, 4096³, materialised |

**What the data says.** Three things, and the third is against this campaign's own interest.

1. **The key is an instrument.** The bands are disjoint on both days: idle max 68 against loaded min
   71, and on the shipped tool Δ +35 against a floor spread of 15. A saturating load never lands
   inside the idle band, so the registered falsifier did not fire and §5v's "not an occupancy
   instrument" was too strong a verdict for what was actually a single untested assumption — that a
   reader whose floor is 60 % cannot see anything. It sees 35 points.
2. **The 15 % bar has a zero-point defect, not a sensitivity one.** Across five samples on two days
   this host's floor never once read below 43 %; the only sub-43 values anywhere in them are the
   transient `0`s that answer as the first read after a load stops, which is the artifact the
   shipped tool's median rule exists for. The consumers are the desktop's own compositor, browser
   and chat renderers (§5s names them at doc lines 1153–1155), and TinyTitan launches none of
   them. So `preflight(15)` (gate0:661, reader at gate0:121-134) is not misreading
   the machine; it is correctly saying that a host idling at 56–71 % is not the reference host a
   published table assumes. §5s's conclusion therefore stands as written — the refusal was the
   finding, `--allow-busy-gpu` was not used, and the wiki row was reported *not measured*. What
   §5bd adds is the other half §5s called missing: a way to **record** the contention and carry on,
   offline, without touching the gate that decides what gets published.
3. **The same reader cannot certify GPU headroom either.** Engine decode measured 69.6 % mean (§5v),
   and this host's floor is 56–71 % with a spread of 11–15 points. Decode's occupancy is *inside* the
   desktop's swing, so on this instrument it grades `indistinguishable` — which means no premise in
   this campaign may be priced from "the GPU is only 70 % busy, so 30 % is free for a third worker".
   That is a bound, and it cuts against the tri-worker idea rather than for it; §8's compute-offload
   closures remain the live evidence there, and §5v's own "no conclusion rests on it" was correct
   even though the reason given for it was not.

**Shipped: `benchmark/tinytitan_gpu_contention.py`.** It measures the host it is on — a floor window,
a run window beside it, a verdict graded against the floor's own spread instead of an absolute zero
the machine never had. Four rules the suite pins, all of them this campaign kept needing: a run
inside the floor band is `indistinguishable`, never `clean`; a side with fewer than three samples is
`unreadable`, and an `unreadable` floor cannot report `meets_absolute_limit`, so a check that read
nothing hands out no pass; the band centre is the **median**, because the first read after a load
stops answers 0 — a mean over the suite's `[0, 62, 65, 64, 61]` is 50.4 where the median is 62, so
the average labels a freshly contended host twelve points quieter than it is, and the loaded window
above carries exactly that artifact as its 48; `--cmd` is split into argv and launched
directly, so no shell ever sees it and a typo'd load is refused before any sampling rather than
traced. It changes no gate and refuses nothing — a run is bounded by what it reports. 16 tests,
registered in CI beside the suites that need neither a model nor macOS, and they run on Linux because
nothing in the suite calls `ioreg`.

**Disclosed defect in a throwaway, caught before it was quoted.** The first version of the load
printed **20 616 GFLOP/s**, ten times the headline rate of the machine, because MLX builds lazily:
ten `a @ a` expressions chained inside a chunk and one `mx.eval` at the end materialise **one**
matmul, and the other nine are never computed but still get counted. Evaluating each chunk fixes it,
and the corrected 4096³ load measures 2 107 GFLOP/s. It is the same shape as the AUD rows about
printed work that was never done, and it is the reason §5bd quotes the probe's 2048³ figures
(`mx.eval` inside the loop, honest) as well as the shipped tool's.

**Verdicts.**
- §5v's deviation (2) is **amended, not withdrawn**: "no conclusion rests on it" still holds, and
  the reason changes from *the reader cannot see* to *the reader sees a floor that is the desktop's,
  so a single sample against an absolute bar cannot be a host statement*.
- §5s's missing mode now exists as a separate offline tool. The gate's binary 15 %-or-off design
  stays exactly as §5s called it — the right design for a published table. Relaxing a shipped
  cross-host gate is the operator's call, not an overnight experiment's, so nothing in `benchmark/`
  that refuses a run was edited here.
- #163 is a **decision**, not another run: the wiki's Qwen3.8 row needs either a window with the
  GPU-drawing apps quit, which is the operator's machine to make, or a row published with a floor/run
  pair beside it, which this tool makes cheap to produce. It has not been refreshed on the strength
  of §5bd alone. **Amended the same day:** the decision arrived, and it was a third route neither of
  those two — the operator chose to publish the campaign's greedy 384-token CLI arms as their own
  labelled protocol rather than force a gate0 median out of this host or wait for a quiet window.
  #163/TT-049 closed with wiki `c2591e7`, whose gate0 sentence was re-worded in place by `5e13783`;
  §5t carries the closing amendment. "Decision, not another run" was the right reading, and §5bd's
  own figures are still not published as a level.

**Not settled.** Nothing about decode tok/s moved; no lever was opened or closed here. The floor
drifts with what this session is doing, which is the same noise source the paired-benchmarking record
already names, so a genuinely quiet machine remains the only host that can price a sub-1 % effect on
wall clock.

**Next.** Nothing in §5bd is a prerequisite for anything. If the operator wants the wiki row, the
shortest path is one gate0 batch taken while nothing draws to the GPU, with a floor/run pair recorded
before and after; if the operator would rather publish with the contention labelled, §5bd's tool is
the label. Either way §5bc's amendment — the feeder question is slot 0 and §5ar's withholding list —
is unchanged by anything here.

**Amended 2026-10-17:** the wiki row was refreshed on neither of those two paths — the operator took
the CLI protocol line described in §5s's and §5t's amendments instead — so both remain available and
neither is now tied to an open task. A gate0-certified level for this model is still unwritten, and
the only thing that would make it writable on this Mac is a preflight that measures TinyTitan's own
occupancy rather than the desktop's.

**Provenance.** Commit `05b9ff9` (tree clean for the two shipped-tool rows; the 2026-10-16 probe rows
were taken from a dirty tree carrying the not-yet-committed gauge and are labelled as a
discrimination test, not as a benchmark, because they measure `ioreg` and not the engine). 24 GB
M3 MacBook Pro (Mac15,3), macOS 27.0.1, Swift 6.4/Xcode 27, AC power, 100 % charged, `pgrep` guard
empty, no model loaded. Commands, verbatim, with exit codes:

```
$ python3 tinytitan_gpu_contention.py --floor-seconds 10 --run-seconds 10 --interval 0.5
{"absolute_limit": 15, "cmd_status": 0, "delta": -2, "floor_band": [60, 71], "floor_median": 65,
 "floor_spread": 11, "meets_absolute_limit": false, "run_band": [59, 69], "run_median": 63,
 "verdict": "indistinguishable"}                                    exit 0
$ python3 tinytitan_gpu_contention.py --floor-seconds 10 --run-seconds 20 --interval 0.5 \
      --cmd "python3 /tmp/gauge215/load.py 15"
matmuls=230 gflops_per_s=2107
{"absolute_limit": 15, "cmd_status": 0, "delta": 35.0, "floor_band": [56, 71], "floor_median": 64,
 "floor_spread": 15, "meets_absolute_limit": false, "run_band": [48, 99], "run_median": 99.0,
 "verdict": "elevated"}                                              exit 0
$ python3 -m unittest test_tinytitan_gpu_contention   # 16 tests, OK
$ PATH="$HOME/.local/bin:$PATH" tools/lint.sh python  # ok, ruff 0.16.7
$ PATH="$HOME/.local/bin:$PATH" tools/lint.sh docs    # ok, 102/106 python suites registered
```

Deviations: the two shipped rows ran back-to-back rather than interleaved, because they are
utilisation bands and not clocks — there is no paired delta to protect, and each row carries its own
floor measured in the seconds before it. The `/tmp/gauge215/load.py` helper is deliberately not repo
code: it is an MLX-dependent load, and `benchmark/requirements.txt` does not carry MLX.

## 5be. #215 decides the question §5az left open — a page that never re-measured its baseline has an *unmeasured* drift, not a zero one, and the screen was paying out a six-link `pays` for a term it had not read (2026-10-17)

§5az closed by naming a seam rather than burying it: *a page with no control row is priced on its digits and can
exit 0 on a much narrower bound than a page with one … Whether the screen should refuse to draw a gate at all
without a drift sample is a design question this section does not decide.* It is decided, and the answer is that it
should. Not on taste. The campaign's whole audit queue is one defect class — **a gate that did not run reading as a
pass** (AUD-210, AUD-212, AUD-233) — and here it arrives in arithmetic form: the channel nobody read contributed a
`0.0` to the bound, and `0.0` is exactly the number a page whose control moved *nothing* prints. The two pages are
indistinguishable downstream, and one of them is being graded on a term that was never measured.

### The false pass, printed whole

The screen as §5az left it, over a three-decimal in-band trio carrying no `base_again` row (fixture
`INBAND_NO_CONTROL`):

```
status 0
digits: the page prints tok/s to 3 decimals (±0.0005), so y is known only to ±0.004 ms and x to ±0.002 ms
  signal: no drift sample (base_again is not on this page), so no row was checked
median x=0.199 y=1.002  spread x=0.000 y=0.000
ADMISSIBLE: y=1.002 is inside 0.788-1.301 ms, so the two rows price both reads
p*=0.199 against the six-link gate's L/S = 0.82/5.30 = 0.155 -> pays (+0.24 ms of a 200.0 ms token, +0.12 %)
```

Every clause of that is defensible but the last. The screen had already *said* the drift was unread — `no drift
sample (base_again is not on this page)` — and then priced as though it were zero. §5az's own numbers say what that
zero costs on this Mac: the quietest round of the identifying sweep moved its control **3.527 ms**, §5ax's noisier
one **11.701 ms**, §5at's page carries **±0.832 ms** of drift against **±0.034 ms** of printing, and scoring the
gate wants a baseline repeating to **0.75 ms**, which no page here has ever given. A digits-only bound on this
machine is not a conservative bound; it is a bound with the dominant term removed.

### The rule, as shipped: one required field and one conjunct

`Price.drift_sampled: bool` ([`:175`](../benchmark/tinytitan_price_pair.py#L175)) is a required field, not a
default, for the reason #207 made `x_bound` and `y_bound` required — a price that silently carried no provenance
would read as a price that needed none. It is set from the only thing that actually separates the two cases, whether
a control row was found ([`:467`](../benchmark/tinytitan_price_pair.py#L467), `drift_sampled=drift is not None`),
because one line later the *widths* agree: `y_drift` is `0.0` both when the baseline repeated exactly and when
nobody asked it. Any number written into the drift column would have to be a number, and every number means "the
machine moved this far" — which is why this is a flag and not a sentinel.

`_verdict` then gates the gate: `licensed = all(p.drift_sampled for p in pairs)`
([`:861`](../benchmark/tinytitan_price_pair.py#L861)), with a refusal of its own
([`:864`](../benchmark/tinytitan_price_pair.py#L864)):

```
the six-link gate is not scored: a round in this pool carried no re-measured baseline, so the bound its
price carries is the printing alone, and the median of several solves is no better known than its
least-read member -- a threshold on p* drawn from prices whose machine movement was never read is a
statement about the arithmetic, not about this Mac
```

`all` over the pool, not over the median: one uncontrolled round withdraws the gate from a batch of otherwise
controlled ones, because a median cannot recover what its least-read member never carried.

`_cannot_decide` ([`:726`](../benchmark/tinytitan_price_pair.py#L726)) took the same flag as an eighth parameter and
now has **three** provenances instead of two:

| the page's drift | width clause printed | remedy printed |
|---|---|---|
| never read | `the printing alone, ±{digits}` | *more printed decimals would narrow only that half, because this page carries no re-measured baseline and its drift is unmeasured rather than zero* ([`:747`](../benchmark/tinytitan_price_pair.py#L747)) |
| read, and moved | `its digits ±… and its own drift ±…` | §5az's two forms, chosen by which half is wider |
| read, and repeated | `the printing alone, ±{digits}` | *more printed decimals is what would settle both* ([`:761`](../benchmark/tinytitan_price_pair.py#L761)) |

Rows 1 and 3 print the **same width string** and now get opposite statuses. That coincidence is the finding: the
screen's advice, not its arithmetic, is what tells an unread channel from a quiet one, so the advice had to become
load-bearing before a status could depend on it.

### What #215 does not refuse

Such a page still prices and still classifies — `ADMISSIBLE: y=1.002 is inside 0.788-1.301 ms` prints above the
refusal — because #208 ruled that a missing control is not a refusal to *solve*, and the arithmetic genuinely does
not need the machine. What it may no longer do is *spend* the answer on this Mac, which is what a gate draw does.
The same distinction is why the new refusal is **added after** #209's interval refusal rather than returned in front
of it ([`:872`](../benchmark/tinytitan_price_pair.py#L872), `if not separable or not licensed: return lines, 2`): a
page can carry both defects, and #209's sentence is a finding about that page — which half of its bound dominates,
and how wide its own `p*` runs — that a licensing rule has no business eating. I found that by writing the second
refusal as an early return and watching `DOC1_IN` lose its interval line, which is the defect [`:730`](../benchmark/test_tinytitan_price_pair.py#L730) now pins.

### The blast radius: eleven pages, two statuses

Every page the campaign has quoted, through the screen before and after. The probe that ran them,
`/tmp/be215_probe.py`, is a throwaway; the full text of both runs is in `/tmp/be215_old.txt` and
`/tmp/be215_new.txt`:

| page | `base_again` on it | before | after |
|---|---|---:|---:|
| §5ag's trio | yes | 1 | 1 |
| §5at's published table | yes | 1 | 1 |
| one decimal, out-of-band (`DOC1`) | no | 1 | 1 |
| one decimal, in-band (`DOC1_IN`) | no | 2 | 2 |
| three decimals, in-band, no control | no | **0** | **2** |
| the same trio, control repeats exactly | yes, zero | 0 | 0 |
| `DOC1_IN`, control repeats exactly | yes, zero | 2 | 2 |
| §5at's six-arm page | yes | 2 | 2 |
| `HOLD_OK` as #209 shipped it | yes, zero | 0 | 0 |
| `HOLD_OK` as §5az wrote it | no | **0** | **2** |
| `HOLD_BAD` (held-out row refutes) | yes | 1 | 1 |

Two statuses flip, both `0 → 2`, and both are the same shape — a control-less page drawing a `pays`. **No published
§5 verdict changes**, because every page the campaign actually measured carried `base_again`: §5ax's sweep and
§5at's batch both ran a re-measured baseline, so the flip lands on a fixture and on a hypothetical. That is the
honest size of it, and it is also the argument for paying now rather than after a real page had spent it. §5au's
fourth row, already `2` since #209, now prints two refusals — one about widths, one about provenance — and both are
true of that page.

### Two test premises that were wrong, and which side moved

The suite failed twice on the first run after the production change, and neither failure was the production code's.

1. **`holdout_page()` had no control row.** The out-of-sample tests build their page at
   ([`:265`](../benchmark/test_tinytitan_price_pair.py#L265)) from `base`, two arms and a held-out arm — no
   `base_again`. Under #215 that fixture's exit-0 premise *is* the false pass this section is about, and the test
   failed `2 != 0`. The fixture was asserting the defect, so the fixture moved: it gained
   `arm("base_again", 5.00, 40.00, 20.00)` ([`:286`](../benchmark/test_tinytitan_price_pair.py#L286)) carrying the
   baseline's rate and counters verbatim, which makes its step zero and leaves the bound at its digits. `HOLD_OK`'s
   published behaviour is unchanged; only the page it is pinned on became one the screen may score.
2. **`DOC1`'s remedy assertion pinned the rejected claim.**
   [`test_a_page_with_no_drift_sample_puts_the_whole_bound_on_the_printing`:624](../benchmark/test_tinytitan_price_pair.py#L624)
   required `more printed decimals is what would settle both` on a page with no sample. Its arithmetic half
   (`±0.310 ms -- the printing alone, ±0.310 ms`) still passes and is kept; the advice clause is gone, and the test
   comment says so rather than hiding it.

Three more tests were **re-pointed at sampled fixtures**, not re-asserted:
`test_an_in_band_y_is_admissible_and_the_gate_is_scored_from_it` now runs `INBAND`
([`:182`](../benchmark/test_tinytitan_price_pair.py#L182)), `test_a_p_star_whose_own_width_spans_the_gate_is_refused`
runs `DOC1_IN_SAMPLED` ([`:195`](../benchmark/test_tinytitan_price_pair.py#L195)), and
`test_a_bound_wider_than_the_band_on_the_admissible_side_says_that_too` deliberately keeps the un-sampled `DOC1_IN`,
because it pins the band caveat and not the gate. New fixture `INBAND_NO_CONTROL`
([`:174`](../benchmark/test_tinytitan_price_pair.py#L174)) is the page from the false pass above. The module
docstring gained a ninth *must not get wrong* — *"That an unmeasured drift is not a zero one (#215)"* — and its
status legend's second line now reads `2  it could not solve, or the page cannot license a gate draw — its own
widths straddle the threshold, or it never sampled its drift — reason named`.

### Tests, gates and provenance

44 tests in [benchmark/test_tinytitan_price_pair.py](../benchmark/test_tinytitan_price_pair.py) — 39 as §5az left
them plus five new in `DriftAndHoldout` ([`:579`](../benchmark/test_tinytitan_price_pair.py#L579)) — and 109 OK
alongside `test_knob_sweep_verdict` and `test_docs_facts_suites` (104 before). **Four of the five are RED against
the screen as §5az left it**, re-measured for this section rather than remembered: `:671` (the status), `:693` (an
unsampled page and a measured-zero page must not be the same page), `:708` (the remedy names the row it lacks) and
`:730` (the missing license does not swallow the interval refusal). The fifth,
[`test_a_measured_zero_drift_still_points_at_the_printing`:719](../benchmark/test_tinytitan_price_pair.py#L719),
passes before and after and is disclosed as **RED-invalid**: it is a regression pin on the branch that already
worked, not a demonstration of the fix.

Deviations.

1. **No model run, and no new measurement.** Every number is the shipped screen's output over fixture pages built
   from rows §5ag, §5at and §5ax published. `/tmp/be215_probe.py`, `/tmp/be215_old.txt`, `/tmp/be215_new.txt` and
   `/tmp/be215_red.txt` are throwaways, not repo code and not proposed as tools; the committed artifacts are the
   screen, its suite and this section.
2. **The RED re-measurement used a reversible in-place swap**, not a copy: `git show HEAD:benchmark/tinytitan_price_pair.py`
   over the working file, run the five tests, restore from `/tmp/pp215_final_mine.py`, `diff -q` clean, and the full
   suite re-run to 109 OK afterwards. A copy outside `benchmark/` cannot work — importing the screen pulls
   `tinytitan_knob_sweep`, which pulls `tinytitan_profile` — so the fixture builders' transitive import is what makes
   the screen un-copyable. No other file was touched while the pre-fix screen was in place.
3. **A count kept earlier in this session said "3 of 5" RED. The measured number is 4.** `:730` fails pre-fix too,
   because the pre-fix screen prints no license line at all. The section states the re-measurement.
4. **Anchors.** This change adds 51 lines to the screen and 125 to its suite, which moved 38 `#L` anchors in §5au
   through §5ba; all were renumbered by diff alignment against `HEAD`, each with its visible label, and every result
   re-checked to land on a non-blank line (one legitimately lands on the comment that documents the six half-widths,
   `:160`). Three labels whose text is a file or function name rather than `:NNN` were fixed by hand. Two *bare*
   inline ref lists in §5au's tests paragraph — its `ReadingAPage:173`, `:215`, `:241`, `:270`, `:280`, `:289`,
   `:294` and the `:400` in the following paragraph — were **already stale before this commit**: the constructs they
   name had moved in #208 and #209, and those sections re-measured their links but not their prose. They are left
   alone and named here, because renumbering them means re-deriving what the author pointed at, which is a separate
   fix and not this one.
5. **Gates and provenance.** `PATH="$HOME/.local/bin:$PATH" tools/lint.sh python` ok (ruff 0.16.7, check and format
   clean, 205 scripts parse at the declared 3.13 floor); `tools/lint.sh docs` ok (87 documents, 102/106 python suites
   registered in CI); the CI markdown-link step (`ci.yml:1008-1038`) run locally on this tree: **187 local links
   checked, 0 broken, exit 0**. `test_tinytitan_price_pair` is already named in CI (`.github/workflows/ci.yml:974`),
   so AUD-211 does not apply. `ruff format` was run on the two
   files this change touches, and nothing else. Machine: M3 MacBook Pro, 24 GiB, macOS 27, Xcode 27 / Swift 6.4, on
   AC power at 100 %; HEAD `f1b8346`; no model process was started and nothing under `sources/` was touched. None of
   it is evidence that a CI run executed the suites.

## 5bf. #216 answers the question §5ar left as an estimate — would more rounds settle the band? — and no pool this campaign can publish does, while the one rule that shrinks had to be stopped from dividing a scatter it never measured (2026-10-17)

Two sentences have been carrying this campaign's "we need more data" since the day they were written. §5ar closed its
withholding arm on an estimate: at "+0.05…+0.32 % against §5ap's observed paired spread (1.6 points over four rounds,
sd ≈ 0.7), resolving the sign needs on the order of **30-60 clean rounds**". §5az closed its deviation on a limit of
language: *"they say no page this quiet can score the gate, not that some page that quiet exists"*. Both are claims
about a **pool** of rounds, and a pool of pages already on disk is an arithmetic question that can be answered before
any more live time is spent. It is answered, offline, on pages the campaign published, and the answer is status `2`.

[`tinytitan_round_budget.py`](../benchmark/tinytitan_round_budget.py) reads the same markdown
[`tinytitan_price_pair.py`](../benchmark/tinytitan_price_pair.py) reads, keeps from every table the pair that table
offers ([`pick_pair`:328](../benchmark/tinytitan_round_budget.py#L328)), solves `dm = x·du + y·dv` **inside each page**
([`pool`:355](../benchmark/tinytitan_round_budget.py#L355)), and prices the pooled `y` three ways. It answers the two
questions the campaign actually asks of a pooled price — is `y` inside `0.788-1.301 ms`, and does `p* = x/y` clear the
six-link gate's `L/S = 0.82/5.30 = 0.155` — and refuses to answer either on a width it has not earned.

### Three provences, and the only one that has to earn its licence

| rule | estimator | width | does more rounds help? |
|---|---|---|---|
| 1 median / widest | median | the **widest** member's bound | **never** — one wild control caps the pool forever |
| 2 mean / triangle | mean | the **mean** of the bounds | toward the average bound, and no further |
| 3 mean / sqrt(n) | mean | `t(df) · sd_y / √n` | **toward zero**, if and only if each round's error is independent zero-mean noise |

Rules 1 and 2 are distribution-free. [`median_rule`:229](../benchmark/tinytitan_round_budget.py#L229) *is* the shipped
screen's rule and is pinned to it to the digit ([`test:210`](../benchmark/test_tinytitan_round_budget.py#L210)); two
screens that price one pool differently are two bugs, not a choice.
[`mean_rule`:245](../benchmark/tinytitan_round_budget.py#L245) is `|mean(z_r) − z| ≤ mean(b_r)` — a triangle inequality
with no independence assumption, which is why it goes with the **mean** and not the median. On §5ax's two rounds plus a
repeat of the quieter one the pool lands at **±0.445 ms** where rule 1 still says **±0.805 ms**: the average of its
bounds, and no further, however many quiet rounds are added to a pool that keeps one wide control
([`test:251`](../benchmark/test_tinytitan_round_budget.py#L251)).

Rule 3 ([`sqrt_n_rule`:262](../benchmark/tinytitan_round_budget.py#L262)) is the one that would license "30-60 rounds",
and it is the one that assumes. The pool carries its own `n` samples of the assumption — the **signed** step each
round's `base_again` row took, the field #215 made required and per-round — so the screen reads them and licenses the
shrink only if they support it ([`license`:179](../benchmark/tinytitan_round_budget.py#L179)). Four refusals, in the
order that says the most, and the two licences:

| the pool's drift samples | verdict | pinned by |
|---|---|---|
| one round never carried a `base_again` | refused — "the pool has an unread half" | [`test:362`](../benchmark/test_tinytitan_round_budget.py#L362) |
| `−3.527, −11.701` (unanimous) | refused — "averaging rounds cancels nothing that has a direction" | [`test:265`](../benchmark/test_tinytitan_round_budget.py#L265) |
| two rounds, any signs | refused — "an sd from two points is the two points talking to themselves" | [`test:282`](../benchmark/test_tinytitan_round_budget.py#L282) |
| net step wider than their own spread | refused — "a bias does not average out" | [`test:293`](../benchmark/test_tinytitan_round_budget.py#L293) |
| mixed signs, spread the wider | **licensed** | [`test:305`](../benchmark/test_tinytitan_round_budget.py#L305) |
| every round stepped exactly `0.000` | **licensed** — "the drift never moved … the channel measured no bias rather than failing to measure" | [`test:335`](../benchmark/test_tinytitan_round_budget.py#L335) |

The last row is §5as's rule applied to a pool: a control that steps exactly zero is a *measured* zero, which is a
different statement from a page that never read the channel, and refusing it would have made the steadiest pool in the
campaign the least poolable one. That refusal is a bug I wrote and then removed; it is in the next subsection because
the real guard turned out to sit on a different quantity.

### §5ax's two clean rounds, pooled — printed whole

The fixture is §5ax's two published tables, carried in the suite verbatim as `AX`, which is the campaign's best pair:
two paired rounds, both with a re-measured baseline, both admitting the same `layer_depth25 × prefetch_off` solve. The
screen echoes the path it was handed, so `table 1` and `table 2` below are §5ax's own two tables read out of a scratch
copy of that page:

```
pair: layer_depth25 x prefetch_off is priced by every round, tightest at 45.1 deg
pool of 2 round(s), column wall, gate L/S = 0.155
  /tmp/az209/ax_two_rounds.md table 1: x=0.120 y=0.897 y bound ±0.264 deg 47.3 drift sample -3.527 ms
  /tmp/az209/ax_two_rounds.md table 2: x=0.224 y=0.837 y bound ±0.805 deg 45.1 drift sample -11.701 ms
  median / widest  : y=0.867 ±0.805 -> cannot settle the band, its bound is 0.805 ms against a margin of 0.079 ms
  mean / triangle  : y=0.867 ±0.535 -> cannot settle the band, its bound is 0.535 ms against a margin of 0.079 ms
  mean / sqrt(n)   : y=0.867 ±0.384 -> not licensed: the pool's own drift samples (-3.527, -11.701 ms) moved the same direction in every round, and averaging rounds cancels nothing that has a direction
  the sqrt(n) budget would clear this margin at 4 rounds of the same scatter -- 2 here
  and that number is a planning estimate only: not licensed: the pool's own drift samples (-3.527, -11.701 ms) moved the same direction in every round, and averaging rounds cancels nothing that has a direction. If more rounds are pooled, their drift samples have to keep pointing different ways for it to mean anything about this Mac rather than about the arithmetic
cannot settle: no licensed rule narrows y enough -- the widest the pool may claim distribution-free is the mean of its bounds, and the rule that shrinks is refused by the pool's own drift samples
exit 2
```

The campaign's headline number survives pooling exactly as §5ax left it — `y = 0.867`, 0.079 ms inside the near edge —
and no distribution-free pooling closes a 0.079 ms margin with a 0.805 ms bound. The projection line is worth reading
carefully, because it reaches for the same decision §5ar's "30-60" was reaching for while measuring a different
quantity: it projects rounds at *the same price scatter* for *this band margin*, on the shrinking rule, and the line
immediately below it marks it "a planning estimate only" and re-quotes the refusal. §5ar's 30-60 was a rate estimate
off a paired `tok/s` spread; this is a price estimate off a solved-`y` scatter. This screen does not re-derive §5ar's
number and does not contradict it.

### A third round from §5at's batch licenses the shrinking rule, and it still loses

A three-round pool adds §5at's published table — a different launch, same day:

```
pair: layer_depth25 x prefetch_off is priced by every round, tightest at 43.9 deg
pool of 3 round(s), column wall, gate L/S = 0.155
  /tmp/az209/pool3.md table 1: x=0.120 y=0.897 y bound ±0.264 deg 47.3 drift sample -3.527 ms
  /tmp/az209/pool3.md table 2: x=0.224 y=0.837 y bound ±0.805 deg 45.1 drift sample -11.701 ms
  /tmp/az209/pool3.md table 3: x=0.683 y=0.796 y bound ±0.866 deg 43.9 drift sample +11.167 ms
  median / widest  : y=0.837 ±0.866 -> cannot settle the band, its bound is 0.866 ms against a margin of 0.049 ms
  mean / triangle  : y=0.843 ±0.645 -> cannot settle the band, its bound is 0.645 ms against a margin of 0.055 ms
  mean / sqrt(n)   : y=0.843 ±0.128 -> cannot settle the band, its bound is 0.128 ms against a margin of 0.055 ms
  the sqrt(n) budget would clear this margin at 7 rounds of the same scatter -- 3 here
cannot settle: no licensed rule narrows y enough -- the widest the pool may claim distribution-free is the mean of its bounds, and the rule that shrinks, licensed by this pool's own drift samples, is still ±0.128 against a margin of 0.055 ms
exit 2
```

That is the campaign's answer in one line: **the shrinking rule is licensed here, shrinks from 0.645 to 0.128, and
still does not clear the margin.** And the licence has a price that the screen prints rather than hides — the third
round is what licenses it, because its control stepped **+11.167 ms** while the other two stepped −3.527 and −11.701.
The machine has to move in different directions for `√n` to mean anything about this Mac, and the movement that grants
the licence is the same movement that puts ±0.83 ms on that round's own price (§5az's `±0.832 ms` from drift against
`±0.034 ms` from digits). So §5az's sentence about a page becomes §5bf's sentence about a pool: *no pool this campaign
can publish both licenses the shrink and is quiet enough for the shrink to land*, and the arithmetic here bounds that
claim rather than the silence of one page.

### Two defects the screen had to be built past, and which side was wrong on each

1. **A bound of `±0.000 ms/token`, printed as a settlement.** The first pool built to test the licence — five rounds
   whose drift samples moved in both directions, so licensed, and whose solved prices were all `0.897` — handed rule 3
   `sd_y = 0`. `t·0/√n` is `0`, `_ceil_ms(0.0)` is `0.0`, and the shipped formatter, on that budget object, prints:
   ```
   mean / sqrt(n)   : y=0.897 ±0.000 -> settles the band inside 0.788-1.301 ms, and the gate does not pay
   ```
   An absence of measurement wearing the costume of the tightest budget in the campaign — the same defect class as
   #215's `0.0` for a channel nobody read, arriving one rule later. My **first** fix was wrong about the cause: I
   refused the licence when the *drift* `sd` was zero, which is over-refusal, because a control that re-measures the
   same number is the steadiest evidence the channel gives. The real guard is scatter in the **prices**, so
   `sqrt_n_rule` refuses itself ([`:291`](../benchmark/tinytitan_round_budget.py#L291)) and the licence now accepts the
   measured zero ([`:215`](../benchmark/tinytitan_round_budget.py#L215)). What the guard prints instead:
   ```
   mean / sqrt(n)   : y=0.897 ±0.000 -> the pool's 5 prices are all 0.897 ms/token: its drift samples moved between rounds but the pair did not, so t*sd/sqrt(n) is t*0/sqrt(n) = 0 and the absence of a measurement would print as the tightest budget in the campaign. n copies of one price is one price, and a scatter never measured cannot be divided by root n
   ```
   The RED was `AssertionError: True is not false` on `budget.licensed`; it is pinned by
   [`test:316`](../benchmark/test_tinytitan_round_budget.py#L316), which also asserts `n_needed is None` so the
   projection line cannot be printed for a budget that never existed.
2. **The closing sentence blamed the drift on a pool whose drift had licensed the rule.** Pool 2 above fails to settle
   on *width*, not on provenance, and the draft ending said so wrongly: "…the rule that shrinks is refused by the
   pool's own drift samples". The RED was
   `"refused by the pool's own drift samples" unexpectedly found`; the ending now branches on `last.licensed`
   ([`:538`](../benchmark/tinytitan_round_budget.py#L538)) and the licensed branch quotes both numbers — ±0.128 against
   a margin of 0.055 — so the sentence is true of the pool that printed it and is pinned by
   [`test:428`](../benchmark/test_tinytitan_round_budget.py#L428).

**One test premise was wrong and the implementation was right.** `test_a_baseline_that_never_moved…`
([`test:355`](../benchmark/test_tinytitan_round_budget.py#L355)) first asserted `assertLess(budget.y_bound, 0.02)` and
failed `0.02 not less than 0.02`. Bounds here round **up** to the millimetre, which is #209's shipped rule and the
whole point of the screen; `sd = 0.01581`, `t(4) = 2.776`, `√5` gives `0.01963`, which prints `±0.020`. The assertion
was re-aimed to `0.019 < bound < 0.021` with that arithmetic spelled out in the comment. No shipped rounding rule was
touched, and nothing was widened to accommodate a failure.

### Sibling audit: every other place a scatter is divided by root n

The screen is new, so the defect class had to be looked for in the files that already had it. Two call sites divide by
`√n` ([`quant_perplexity_ab.py:100`](../benchmark/quant_perplexity_ab.py#L100),
[`memory_master.py:534`](../benchmark/memory_master.py#L534)), and only the first is the same shape:
[`paired()`](../benchmark/quant_perplexity_ab.py#L94) returns `mean, 0.0, math.inf` for a single paired position
([`:99`](../benchmark/quant_perplexity_ab.py#L99)) and maps a zero `stderr` to `t = inf`
([`:101`](../benchmark/quant_perplexity_ab.py#L101)) — so one sample, or `n` samples that differ by exactly the same
amount, prints `±se 0.000000` beside a `t` of `inf`. That is this section's defect, one rule over. It is **half**
defended already, and that is why it is not bundled here: the row prints its own `n` immediately after the `t`
([`:147`](../benchmark/quant_perplexity_ab.py#L147), header at [`:139`](../benchmark/quant_perplexity_ab.py#L139)), so
the emptiness is disclosed rather than hidden, and the quantity is a per-token-position mean over ~1024 continuous NLLs,
where `n` identical differences would themselves be the finding. The fix is to print `n/a` where the statistic is not a
statistic, and it belongs in its own unit with its own failing test, not in this one.
`memory_master.py:534` is not the same shape at all: `stdev(values) if len(values) > 1 else 0.0` is a *reported* spread
column in a memory matrix, not a denominator, and no verdict divides by it.

### The arithmetic is shared, not copied

`dm_step` ([`price_pair:357`](../benchmark/tinytitan_price_pair.py#L357)), `missing_cells`
([`:635`](../benchmark/tinytitan_price_pair.py#L635)) and `pi_interval`
([`:688`](../benchmark/tinytitan_price_pair.py#L688)) were private and are now public, so the pool screen calls the
shipped screen's own solve, its own refusals and its own interval rather than a second copy of them. The renames are
identifier-only — no line of `tinytitan_price_pair.py` moved, so every `#L` anchor §5au through §5be wrote at it is
still on its line, and `_pi_interval` in §5be's prose was updated in place. The reuse is not cosmetic:
[`test:210`](../benchmark/test_tinytitan_round_budget.py#L210) pins rule 1 to the shipped screen's published `±0.805`
on §5ax's real pair, and the live check agrees independently — on §5at's own page the pool screen prints that page's
`y=0.796`, bound `±0.866`, angle `43.9 deg`, drift `+11.167 ms`, which is what the shipped screen says for the same
table.

### Deviations, gates and provenance

1. **Pool 2 is cross-batch**, and that is the deviation the campaign's own rule forbids for clocks. It is legal here
   for one reason and one reason only: every price was solved **within** its own page, from that page's rows against
   that page's baseline, and no `tok/s` from one batch was ever compared with a `tok/s` from another. `x` and `y` are
   ratios of steps inside a page; pooling them pools estimates of one physical quantity, not two machines' rates. Read
   the 7-round projection as a planning number with that asterisk on it, which is what the screen's own "planning
   estimate only" line exists to say.
2. **The falsifier was written before measuring:** *if any pool the campaign can publish lets a licensed rule clear its
   band margin, the campaign's "more rounds would settle it" answer changes and a live re-run is justified.* It did not
   fire. Both published pools exit `2`, one with the rule unlicensed and one with the rule licensed and still 0.073 ms
   short. So this section buys the campaign nothing that costs machine time — and it retires the question rather than
   answering it with another sweep.
3. **Suites.** `python3 -m unittest test_tinytitan_round_budget test_tinytitan_price_pair` → **Ran 63 tests OK** (19
   new, 44 pre-existing); adding `test_quant_perplexity_ab` for the sibling audit → **Ran 71 tests OK**. Only these
   three files import the renamed helpers, so the rename's blast radius is closed by grep and not by assumption. The
   new suite is named in CI (`.github/workflows/ci.yml:992`), so AUD-211 is satisfied; `tools/lint.sh docs` ok (87
   documents, **103/107** python suites registered in CI, was 102/106);
   `PATH="$HOME/.local/bin:$PATH" tools/lint.sh python` ok (ruff 0.16.7, check
   and format clean, 207 scripts parse at the declared 3.13 floor); the CI markdown-link step (`ci.yml:1022-1056`) run
   locally on this tree — **220 local links checked, 0 broken**. `ruff format` was run on the files this change touches
   and nothing else.
4. **Nothing here is a model run.** No server, no process, no fetch: the fixtures are pages already on disk, and
   `/tmp/az209/*.md` are throwaway copies of markdown this repository already published. Machine: M3 MacBook Pro,
   24 GiB, macOS 27.0.1, Xcode 27 / Swift 6.4, HEAD `03c2d03`, on AC power. None of it is evidence that a CI run
   executed the suites.

## 5bg. #217 closes §5w's open seam from one page's own counters: the host has 40.046 ms/token that is not a blocked wait, family 2's 43.006 exceeds even that, and the 20.944 inside the loop is the same ~20 ms §5ay needed from a paired difference (2026-10-17)

**Question.** §5w partitioned a ~217-219 ms/token decode into ~113 ms of GPU busy time and ~102 ms of GPU idle in
named gap pairs, and left one pair open: `shared_expert → moe_phase1_*`, **42.835 ms/token**, called "the open seam"
and "where the next session should start" (`docs/qwen38-decode-12gb-measurements.md:1738`). §5x priced it by
arithmetic — 22.9 µs of host time per command-buffer commit × 6 commits × 48 layers = 6.59 ms/token, 15% of the
family — and closed it by elimination. Elimination is not measurement, and the standing instruction here is not to
rely on old tests. This section measures the same seam on one capture arm, from the counters the shipped build already
prints, with no new instrumentation and no second arm.

### The nesting is a property of the code, not an assumption about it

The three envelope timers are not siblings. Read the scopes, not the footer order:

| term | opened | closed | what it contains |
| --- | --- | --- | --- |
| `body_ms` | `RealForwardRunner+DecodeLayers.swift:46` `let tBodyStart` | `RealForwardRunner+DecodeMoE.swift:472` `totalBodyNanos &+= now - tBodyStart` | every other term below |
| `wait_ms` | `+DecodeLayers.swift:222` `let tWait` | `+DecodeLayers.swift:261-262` | the blocked router readback |
| `io_ms` | `+DecodeMoE.swift:243` `let tIoStart` | `+DecodeMoE.swift:307` | the blocked miss pread, `io_queue_ms`, `io_completion_to_fixup_ms` |

`+DecodeLayers.swift:334` passes `bodyStart: tBodyStart` into the expert stage, so the same reading of the clock that
opens the body is the one that closes it after the io await — which is what makes `io` a *sub*-interval rather than a
term beside it. Therefore

    bookkeeping = body − wait − io

is the only host time inside a layer iteration that is neither blocked on the router nor blocked on the drive, and so
the only host time a bookkeeping change inside the loop could remove.

That is one of the two identities the footer supports, and the first version of this page had only the first. The
other is that a token is longer than the 48 bodies stacked end to end, and the difference is not mystery time — it is
where five of the footer's counters are accumulated. `RealForwardRunner+Decode.swift` runs the token as preamble
(opened `:75`, closed `:156`), embed (`:157`→`:197`), gather (`:201`→`:203`), then `decodeLayerLoop(...)` at `:214`,
then the head (`:276`→`:309`/`:281`). Every one of those four opens and closes **outside** the call that contains the
48 bodies, once per token; the three `loop_*` timers are the same shape one level up, in the generation loop
(`RawCompletion.swift:305,340,362`). So

    off the loop = wall − body

is their envelope, and pricing them against `bookkeeping` — which the first draft did — sums a per-token set into a
per-layer one. The counters classify by scope, read off the sites:

| scope | counters | site |
| --- | --- | --- |
| inside each body | `cb1_ms` | `+DecodeLayers.swift:269` — and it subtracts `waitNanos`, so it is measured beside the wait |
| | `router_readback_ms`, `cache_plan_ms` | `+DecodeMoE.swift:50`, `:67` |
| | `cb2_ms` | `+DecodeMoEFinalize.swift:89` |
| | `rdadvise_ms` | `+Diagnostics.swift:31` |
| once per token, outside | `pre_ms`, `embed_ms`, `gather_ms`, `head_ms`, `head_fused_ms` | `+Decode.swift:156,197,203,309,281` |
| | `loop_sample_ms`, `loop_progress_ms`, `loop_other_ms` | `RawCompletion.swift:305,340,362` |

Adding the two envelopes once, at the end, gives the absolute bound: `host off waits = bookkeeping + (wall − body)` is
every millisecond of the token that is not a blocked wait — the loosest possible host-side explanation for GPU idle.
Anything the footer calls `_ms` that is neither is inside one of the two waits, nested in a term already counted, or a
percentile rather than a total.

### The screen ships, because the identity is the durable part

`benchmark/tinytitan_host_path_split.py` (15 tests in `benchmark/test_tinytitan_host_path_split.py`) reads the
`TinyTitan runner` line, the `TinyTitan gap` lines and the decode footer. It deliberately does **not** parse the layer
trace: `TINYTITAN_LAYER_TRACE` prints only `position < 3 || position % 16 == 0`, which is 33 of 512 positions, so a
trace-derived mean is a sample of a biased subset while the footer is the whole run. Statuses: `0` family 2 exceeds
even the absolute bound, so no host-path change reaches it; `1` host time off the blocked waits reaches family 2, so
an arm is worth pricing — with a different sentence depending on whether that time is inside the loop or only outside
it; `2` the page cannot be read — a required line is absent, or one of the two identities fails. Its verbatim output on
the capture arm:

```
body          : 199.960 ms/token  (the whole layer iteration)
wait          : 105.340 ms/token  (blocked on the router readback, inside body)
io            : 73.676 ms/token  (blocked on the miss pread, inside body)
bookkeeping   : 20.944 ms/token  (body - wait - io: host work in the loop that is neither)
family 2      : 43.006 ms/token  (shared_expert->moe_phase1_hit, shared_expert->moe_phase1_2_routed)
named in body : 18.653 ms/token  (5 counters; 89.1% of the in-body envelope)
wall          : 219.062 ms/token  (one token of decode end to end)
off the loop  : 19.102 ms/token  (wall - body: the token outside the layer bodies)
named off loop: 16.999 ms/token  (8 counters; 89.0% of that envelope)
host off waits: 40.046 ms/token  (bookkeeping + off the loop: every millisecond that is not a blocked wait)
nested in io  : io_queue_ms=0.361, io_completion_to_fixup_ms=1.607 (not added to a split)
nested in pre : pre_release_ms=0.002, pre_pin_ms=0.237, pre_reserve_ms=0.002 (not added to a split)
percentiles   : expert_load_p50_ms=2.000, expert_load_p95_ms=8.000, expert_load_p99_ms=16.000 (not added to a split)
identity      : body = wait + bookkeeping + named in body: the two differ by 10.9% of the envelope
identity      : wall = body + off the loop, named off loop: the two differ by 11.0% of the envelope
VERDICT: a host-path change inside the layer loop can address at most 20.944 ms/token of its 43.006 (49%, 9.6% of the token)
            the absolute bound is worse: 40.046 ms/token is everything that is not a blocked wait, and family 2's 43.006 exceeds even that by 2.960. The rest is queue serialization behind the same bytes family 1 names, and the host is not on it
```

Three things in that block are reported rather than smoothed. The two named sets **under**-cover their envelopes by
almost exactly the same amount — 89.1% and 89.0% — and neither over-counts. That is the screen's own evidence that the
scopes it assigns are the scopes the counters accumulate in: a mis-assigned counter shows up as one envelope over- and
the other under-, not as two clean eighths. The 2.291 ms/token left unnamed inside the body and the 2.103 outside it
are host time the footer does not have a counter for, and they stay unnamed. And `family 2` here is **43.006 ms/token**
against §5w's 42.835, 0.4% apart. That is a counter agreeing across batches, which the campaign's cross-batch rule
permits; no token *rate* from one batch is compared with a rate from another anywhere in this section.

### Result: the seam closes on measurement, and §5x's arithmetic was the right arithmetic

The falsifier was written before measuring: **if this page's own family-2 gap is *not* bounded by its own removable
host time — i.e. if bookkeeping reaches family 2 — then the GPU idles behind live host work, a host-path arm is worth
pricing, and the screen exits `1`.** It did not fire. Bookkeeping is 20.944 ms/token against 43.006: **49%**, and
**9.6% of the token** (0.436 ms of the 0.896 ms per layer). So §5w's "start here" is answered from the same page's
counters: at most half of the seam is host-addressable even if the surgery were free, and the remainder is queue
serialization behind the bytes family 1 already owns — `moe_phase1_hit → moe_phase1_miss_fixup`, 55.447 ms/token here,
which §5w showed is priced by the drive at `bytes ÷ 3.92 GB/s` within 6.4%.

Correcting the scope classification (defect 5 below) moved the host side of that comparison *up*, from 20.944 to
40.046 ms/token, so it could only have made the falsifier easier to fire. It is still 2.960 ms/token short: the seam
fails the loosest test the host can be given, not merely the narrow one it was originally tested against. The honest
statement of the result is therefore the stronger of the two — **no host-path change available on this page can
account for family 2**, because 43.006 ms/token of GPU idle is more than all 40.046 ms/token that is not already a
blocked wait — and the 49% figure is what is addressable from *inside* the layer loop, where a bookkeeping change
would have to live to be cheap.

The largest single bookkeeping item makes the closure sharper than §5x's did. `rdadvise_ms=7.876` is 37.6% of the
whole envelope, and §5m already A/B'd it on this family at this budget: `--rdadvise default` measured **+2.2%** over
`off` because hiding fell from **99.8-99.9% to 84.3-84.9%** when the policy was removed. So the biggest removable term
is not free to remove — deleting it costs the hiding that keeps `io_ms` off the token. Even the *whole* envelope,
taken out at no cost, is 9.6% of the token, and the one part of it ever priced live is negative. That is the closure:
not "the host path is unmeasured" but "the host path is measured, is 20.944 ms/token, and its largest term pays for
itself".

### A third side of §5ay's 20 ms — and what the agreement does not prove

§5ay needed about 20 ms of host bookkeeping to reconcile its paired table, and got it from a *difference*: at 48 slots
family 2 falls to 19.930/20.386 ms/token from 42.835, so "the deeper queue buried about 20 ms of host bookkeeping per
round" — "the same 20 ms measured from opposite sides" (`docs/qwen38-decode-12gb-measurements.md:5914`). This section
measures the **absolute** envelope on a single arm, without any pairing, and gets **20.944**. Three sides — the paired
48-vs-96 difference, §5ay's gap-column arithmetic, and this `body − wait − io` subtraction — land within about a
millisecond of each other.

What the agreement is *not*: 20.944 is the total removable host time, whereas §5ay's ~20 ms was a change in one gap
family between two slot tiers. Those are different quantities that happen to be near-equal, and the campaign has
already been burned once by a numerical agreement read as a mechanistic one (§5af retiring 3.92 GB/s as a *rate*).
The defensible reading is narrower: the envelope bounds §5ay's absorption claim, since a tier difference cannot exceed
the host time available to be absorbed, and 20.5 ≤ 20.944 is consistent. That bound is measured here for the first
time; before this page it was an inference.

### Five defects, and which side was wrong on each

1. **The wall regex assumed the footer's field order.** Written as `decode=<s>s new=<n>tok`, it never matched, because
   the CLI footer prints `new=512tok decode=112.16s`. The symptom was a missing `% of the token` line — the screen
   would have printed a verdict with no denominator. The two fields are now matched independently, and the fixture
   also proves that `TinyTitan kernel total_gpu_ms=... gpu_share_of_decode=59.7%` (log line 1558) does *not* satisfy
   the pattern, since it carries a `decode=` field that is not a duration.
2. **My own per-layer summary mis-read the trace.** An `awk` slice at substr offset 10 instead of 9 dropped the
   leading digit of `body_us` and produced an impossible body(588 µs) < wait(2234 µs). The temptation was to conclude
   the engine's timers disagree. Instead the scopes were read, which is where the nesting table above comes from:
   body ⊇ wait ⊇ io. Corrected, the trace gives per-layer body ≈ 4,588 µs against the footer's 199.960/48 = 4,166 µs,
   and the difference between those is the trace's position sampling, not an engine inconsistency — which is the
   reason the screen reads the footer.
3. **My hand-sum of the first draft's named-terms row was wrong: 24.051 written, 24.047 is what the fixture's digits
   give.** The test moved to the re-derived value and the correction is recorded inside the test's own comment, so a
   future reader cannot re-make it. That row no longer exists — defect 5 replaced it — but the arithmetic lesson did
   not: every sum in the corrected version was computed in a `python3` heredoc from the fixture's own digits, not
   added by hand, which is the third time this campaign has caught itself hand-adding and getting it wrong.
4. **One test name contradicted its assertion.** `..._close_the_identity_within_two_percent` was renamed to
   `..._are_reported_against_the_envelope`. The identity is nothing like 2%, and the assertion never tested a
   threshold; the *name* was the wrong side, so the name moved and no assertion was touched. Defect 5 then renamed it
   a second time, to `test_each_named_set_is_reported_against_its_own_envelope`, and changed its assertion — that one
   was no longer a naming problem but a wrong claim.
5. **The screen classified five counters by footer position instead of by scope, and I found it while checking a
   sentence I had written about the wrong thing.** The first version summed eleven counters into one `named terms`
   row, got 24.047 against a 20.944 envelope, read the 14.8% over-count as *overlap*, and this page asserted in prose
   that `pre_ms`'s window sits inside `cache_plan_ms`'s. Before defending that sentence I went to read the two
   accumulation sites: `pre_ms` closes at `+Decode.swift:156`, which is *before* `decodeLayerLoop` is even called at
   `:214`, while `cache_plan_ms` closes at `+DecodeMoE.swift:67`, inside a body. Neither window contains the other —
   they are in different scopes, once-per-token versus 48-times-per-token. So the over-count was not overlap, it was
   5.394 ms/token of per-token timers (24.047 − 18.653) added into a per-layer envelope, and the doc's mechanism claim
   was false in exactly the way a scope error is false. The fix is the two-envelope classification above; the
   corrected screen reproduces every old number that was right (`body`, `wait`, `io`, `bookkeeping` 20.944, family 2
   43.006) and replaces the 14.8% over-count with two ~11% under-counts. The lesson recorded here is the one the
   campaign keeps re-learning: a mechanism sentence must be read out of the code before it is written down, because a
   plausible mechanism is how a misclassification survives into the published page.

Inventory of test-file changes in this unit: one new file (`test_tinytitan_host_path_split.py`), no pre-existing test
file edited, and inside the new one — 9 tests at the first GREEN, **15** at the second; the two renames above; the
24.047 correction; rows read by `cell()` label instead of by padded column width (which is why three formatter
failures in the first cycle cannot come back); and six tests added for what the first draft had no evidence for —
the off-loop set not being bookkeeping, the two coverage figures, the absolute bound, the footer refusal, the
body-longer-than-wall refusal, and the middle verdict. The two refusal/verdict branches that went in *ahead* of their
tests were then mutation-checked rather than left asserted: deleting the footer refusal makes
`test_a_page_with_no_decode_footer_cannot_be_bounded` fail (with an *error* — `bookkeeping / wall` divides by `None`,
so the refusal also guards a crash), and collapsing the `host_off_waits < fam` test to always-true makes exactly the
two verdict tests fail and no other.

### Sibling-defect sweep of the same shape

The misclassification was found in a screen written this week, so the question the standing instruction asks is what
else reads a counter by its name rather than by where it accumulates. Three neighbours checked. §5w's own ledger
partitions GPU-side intervals, not host totals, and it treats `wait` and `io` as sub-intervals of the body it names,
so it is scope-correct as written. The shipped price-pair screen (`benchmark/tinytitan_price_pair.py`) reads only
paired wall figures and never sums a footer term into an envelope, so it has no scope to get wrong. And the byte
counters cannot enter either split by construction: `_ms_terms` accepts a key only if it ends in `_ms`, so
`rdadvise_mib` and `expert_read_mib` (`ServerModelSession+Diagnostics.swift:339,341`) are filtered out before
classification — which is the one part of this that does not depend on the table above being kept current. The shape is
not systemic here, but it is the reason this screen now prints both envelopes instead of one: a counter added to the
footer will be mis-assigned by the same route, and two ~89% coverages make that visible on the page rather than in a
mechanism claim.

### What this does not say

It does not say the 43.006 − 40.046 = 2.960 ms/token is nothing — it says the host is not on it. Neither does it claim
the bound is tight in the operator's favour: `host off waits` is built from the two *envelopes*, not from the named
counters, so the 2.291 ms/token unnamed inside the body and the 2.103 unnamed outside it are already counted in the
40.046. The bound therefore does not under-count host work by leaving the missing 4.394 out; if anything it
over-counts, because some of that time is the GPU waiting to be fed rather than the host doing something. It does not
rule out a structural change to where the wait happens: a device-side hit path already exists in measurement form
(`decodeExpertExecution == .gpuResidency`, `+DecodeMoE.swift:128-182`), and the reason it cannot become the
authoritative partition is recorded in the code at `+DecodeMoE.swift:175-177` — the GPU classifies against the
residency table as it stood *before* the cache plan adopts prefetched experts, so running the GPU's partition would
re-fetch an expert already resident. Any arm that wants to use it must first give the GPU the adoption, which is a
design change and not a flag. And nothing here is a tok/s claim: this is one arm's own counters against one arm's own
wall, priced for decomposition.

### Deviations, gates and provenance

1. **A protocol deviation cost the first attempt.** The launch line chained `date -u -t $(date +%s)`, which BSD `date`
   rejects; exit 1 short-circuited the `&&` chain and the CLI never ran, so no output files were written. Relaunched
   with `date -u '+...'`. Disclosed because a silent chain failure is exactly the shape that fakes a run.
2. **Binary provenance.** At capture time `find sources -newer .build/release/TinyTitanCLI` named
   `RuntimeConfiguration.swift` — a comment-only change in `03c2d03` — so release was rebuilt (119.59 s) and the run
   then provably matched HEAD. The tree at capture was clean at `7facfe0` except the files this unit touches — two new
   `benchmark/` scripts and this document — none of which is in the binary's dependency closure, so no number here is
   published from a dirty tree.
3. **Machine state at preflight:** on AC power, `memory_pressure -Q` acceptable, swap 6.307 GB, no other model
   process; one model process for the whole capture.
4. **The response hash is not §5bc's.** stdout md5 `1225f6270b9a4425ab6cf0560add9f81` differs from the essay recipe
   earlier sections hashed. This arm used the capture arm's own prompt, so it is a *different recipe*, not a numerics
   change — and the distinction is stated rather than assumed, because a hash difference is the one signal this
   campaign treats as outranking any speed number. No golden baseline was run for this unit: nothing in it touches the
   runtime, and both suites are model-free.
5. **Suites and gates.** Four runs of the suite are on the record, and the counts below are quoted from those runs
   rather than reconstructed. First: the suite did not exist, so it died at `ModuleNotFoundError: No module named
   'tinytitan_host_path_split'`. Second: `F.FF.....` → `Ran 9 tests … FAILED (failures=3)`, and all three were *shape*
   assertions over correct numbers — `'9.6% of the token' not found`, `'family 2    : 43.006 ms/token' not found` —
   because the tests read columns by padded width; that failure mode cannot come back, since rows are now read by
   `cell()` label. Third: after the scope correction and the six new tests, `Ran 13 tests … FAILED (failures=4)`, two
   of which are the ones that could only fail on a misclassified screen —
   `test_each_named_set_is_reported_against_its_own_envelope` (`'89.1%' not found`: the old screen printed one
   `named terms` row and no coverage figure) and `test_a_layer_loop_longer_than_the_token_it_belongs_to_is_a_refusal`
   (`1 != 2`: the old screen had no body-exceeds-wall refusal). Fourth: `Ran 15 tests … FAILED (failures=2)`, the two
   verdict tests (`..._when_only_the_absolute_bound_reaches`, `..._when_family_two_drops_below_bookkeeping`). Final:
   `python3 -m unittest test_tinytitan_host_path_split` → **Ran 15 tests … OK**. The two branches that went in *ahead*
   of their tests had no such proof at the time — the footer test passed on arrival — so they were mutation-checked,
   re-run on copies under `/tmp` so the working file was never touched and nothing needed reverting: deleting the
   footer refusal gives `FAILED (errors=1)` and the one error is
   `test_a_page_with_no_decode_footer_cannot_be_bounded` — an *error*, because `bookkeeping / wall` then divides by
   `None`, so that refusal guards a crash as well as a verdict; collapsing the absolute-bound branch to always-true
   gives `FAILED (failures=2)` and the two are exactly the middle-verdict test and the falsifier test, no others. The
   unmutated copy reports `Ran 15 tests … OK` as the control.
   `tools/lint.sh python` → `ok (ruff 0.16.7, check and format clean, 209 scripts parse under 3.13)`, and the pinned
   `ruff check` / `ruff format --check` on the two new files → `All checks passed!` / `2 files already formatted`.
   `tools/lint.sh docs` **failed first**: `ci.yml names test_tinytitan_host_path_split and no
   benchmark/test_tinytitan_host_path_split.py is tracked` — AUD-211 reading the tracked set, so the suite had to be
   `git add`ed before the gate could see it; after that it reports `ok (87 documents against 18 gates …, 222 ledger
   commit reference(s) resolved; 104/108 python suites registered in CI …)`. The CI markdown-link ruby step, run
   locally over this tree, → `Checked 220 local link(s); 0 broken.` No golden baseline was run: this unit changes no
   runtime code and both suites are model-free.
6. **Not evidence.** The model run happened here and its log is on disk at `/tmp/bg217/err.txt` (1,569 lines, 1,536 of
   them layer-trace lines); nothing here is evidence that a CI run executed a model. The two new suites are model-free
   and CI does run those. HEAD `7facfe0`'s CI run is the campaign's live evidence, and #215's run `38981220287` shows
   **cancelled** because my own push tripped `cancel-in-progress` — reported, not hidden.

## 5bh. #218 puts the missing number on the v4.2 kill: the event arm delivers its stated mechanism in full — 18,384 host waits to zero, the wake bubble to 0.0000 — and loses 14.27 % because it hands the wait to one model-wide timeline that cannot release out of order (2026-10-17)

**Question.** `docs/v4.2-expert-streaming-engine.md:326-327` records "**Not promoted:** immediate submission, event
pread, contiguous pool, GPU classifier, and aging LFU (no end-to-end win under the representative run)", and line 58
keeps the default deferred "because immediate submission was slower in the representative real-model A/B". Neither
line gives a number. Both knobs are still shipped and still flag-selectable
(`sources/TinyTitan/Runtime/Configuration/RuntimeConfigurationTypes.swift:53-81`), and they are the operator's
constant-feeder idea almost word for word: submit the read before the compute needs it, and let the GPU release the
already-encoded compute when the bytes land, instead of waking the host to encode it again. A kill with no published
measurement is the same defect §5ba found on the 2026-09-05 §8 page, so this section measures it.

**Premise, checked before measuring.** `EngineConfiguration` does not carry either flag — they are read off the
environment inside the load path, so the question "does `TinyTitanCLI` honour them at all" was live. The chain is
`Run.swift:117` → `Engine.swift:74` → `ServerModelSession.load` →
`ServerModelSession+Loading.swift:82-83`, which builds the load-time `RuntimeConfiguration` with
`expertIOSynchronization: try RuntimeExpertIOSynchronization.environmentValue()` and
`expertIOSubmission: try RuntimeExpertIOSubmission.environmentValue()`. Confirmed against the machine as well: the
`.event` arms print `io_host_waits=0`, which that path and no other produces.

**Method.** Eight arms, two interleaved rounds, mirrored order, shipped arm first **and** last in each round —
`base, imm, evt, base` then `base, evt, imm, base`. qwen38 4-bit, 384 new tokens, greedy, `--max-context 262144`,
`TINYTITAN_RUNNER_STATS=1 TINYTITAN_KERNEL_STATS=1`, mains power, one model process at a time, no `--ram` flag (the
host derivation on this 24 GB Mac prints `cache=11.88G slots=96`, inside the operator's 12 GiB cap). The falsifier was
written down first: *`.immediate` is a win only if the `moe_phase1_hit → moe_phase1_miss_fixup` interval falls in both
rounds; if only the label moves while the enclosing span grows, the −29 ms/token seen in the smoke was that label, not
a bubble.*

### The clock, against its own bracket

| round | base (first) | imm | evt | base (last) | bracket mean | drift |
| --- | --- | --- | --- | --- | --- | --- |
| r1 | 4.961 | 4.878 | 4.362 | 5.031 | 4.9960 | +1.41 % |
| r2 | 5.028 | 5.078 | 4.271 | 5.121 | 5.0745 | +1.85 % |

Paired against the bracket of its own round: `.immediate` is **−2.36 % then +0.07 %** — the direction flips, so this
page cannot call it a win and does not need to; the kill stands, now with a magnitude and a mechanism. `.event` is
**−12.69 % then −15.83 %**, mean **−14.27 %**, same direction both rounds and seven times the drift band. Neither arm
is close to the +6.51 % / +8.76 % that §5bb's shipped predictive ring earns, and neither is worth a default change.

### The labels are not comparable across arms, and that is the finding

The `.event` run prints **no `moe_phase2_routed` row at all**. Phase 1's fixup and phase 2 are one already-submitted
command under that path, so its trace reads `moe_phase1_hit → moe_phase1_miss_fixup_phase2 → attn_norm_qkv` where the
default reads three rows. A line-for-line A/B would have set the default's 55.200 ms/token against the event arm's
77.566 ms/token — intervals that do not contain the same work. The comparable is the **label-independent span** from
`moe_phase1_hit` to `attn_norm_qkv`, summed over whichever rows the arm happens to print:

| arm | span ms/token | vs round bracket | `io_ms` | Δ | `wait_ms` | Δ | bookkeeping | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 base (mean) | 57.939 | — | 69.671 | — | 93.573 | — | 19.506 | — |
| r1 imm | 60.415 | **+2.476** | 80.176 | +10.505 | 95.498 | +1.926 | 11.894 | −7.612 |
| r1 evt | 78.649 | **+20.710** | 94.482 | +24.811 | 195.452 | +101.880 | −80.899 | −100.405 |
| r2 base (mean) | 56.347 | — | 67.901 | — | 92.613 | — | 19.371 | — |
| r2 imm | 62.706 | **+6.359** | 81.778 | +13.878 | 87.839 | −4.774 | 10.112 | −9.259 |
| r2 evt | 80.129 | **+23.782** | 96.228 | +28.328 | 200.280 | +107.667 | −82.640 | −102.011 |

So the falsifier **fires**. `.immediate` does cut the bubble it names — `io_completion_to_fixup_ms` falls in both
rounds, −14.3 % and −25.2 %, and the `hit → miss_fixup` row fell 55.200 → 35.520 in the smoke — but the span that row
lives inside **grows in both rounds**, because the row immediately after it grows more: `miss_fixup → moe_phase2_routed`
goes 1.614 → 23.217. Immediate submission converts roughly 8-9 ms/token of host bookkeeping into 10-14 ms/token inside
`io_ms`: handing the fixup to the queue earlier does not skip the queue, it joins it behind the demand burst the fixup
was waiting for, and the extra time is charged to the I/O scope instead of the host. That is the whole of it — the
bubble was real, it was 1.5 ms/layer, and it was already off the critical path.

### `.event` gives back exactly what it promised and still loses

Both rounds, identically: `io_completion_to_fixup_ms` = **0.0000**, `io_host_waits` = **0**,
`io_host_waits_avoided` = **18,384** = 48 layers × 383 tokens — every host wait in the run, on every layer of every
token. The v4.2 claim (`docs/v4.2-expert-streaming-engine.md:68-96`, "This removes the successful-I/O task
wake/encode/commit bubble") is not merely true, it is *completely* delivered, and the arm is 14.27 % slower anyway.
The `wait_ms` column says where the loss is: 93.6 → ~198 ms/token, +102 and +108 against the bracket. The wait moved
onto the device; it was not removed.

The reason is in the code, and it is deliberate. `ExpertIOEventCoordinator.swift:18-20` — "Out-of-order operations are
deliberately held until every preceding value is terminal. Advancing a timeline directly from N to N+2 would also
release a GPU wait for N+1, potentially before that batch's bytes were valid" — and `publish()` advances
`event.signaledValue` only through a **contiguous prefix** of terminal values
(`ExpertIOEventCoordinator.swift:85-90`). There is exactly one such timeline per model: `Model.swift:263` constructs a
single `ExpertIOEventCoordinator` and hands the same object to every layer's streamer
(`PreadExpertStreamer.swift:120`), which reserves from it at `PreadExpertStreamer+Plan.swift:184`, and the decoder
pre-encodes `routedCB.encodeWaitForEvent(token.event, value: token.value)` at
`RealForwardRunner+DecodeMoE.swift:374`. A single monotonic counter cannot express "N+1 landed but N has not", so one
slow read holds the releases of every batch issued after it, across all 48 layers. Removing 48 × 1.5 ms of host wake
buys a device wait that cannot pass the slowest in-flight read, and at this byte rate that is a bad trade by 14 %.

The ordering rule is not a defect — it is what makes the path safe, and it is why a shared `MTLSharedEvent` behaves
this way. It is the *granularity* that costs: the safety argument holds per-bucket, so one event per layer would keep
it while a slow read in layer 7 stopped holding layer 12's release. See "What this leaves", below.

### The screen #217 shipped refuses the event arm, and refusing is right

`benchmark/tinytitan_host_path_split.py` run on these captures (exit codes measured without a pipe):

- **base** → exit 1. family 2 36.305 is *below* the absolute bound 37.169, so "a host-path arm is worth pricing — but
  not inside the loop: bookkeeping is only 19.698". r2 agrees (36.362 vs 36.946, bookkeeping 19.824).
- **imm** → exit 1. family 2 35.068 *exceeds* the bound 29.352 by 5.716, so the screen rules the host off it — which is
  precisely what the paired counters then show.
- **evt** → **exit 2, `NOT MEASURED`**: `body=209.035` is smaller than `wait=195.452 + io=94.482`, breaking the nesting
  the split is built on.

That refusal is the first real subject §5bg's guard was written for, and the inequality is not a counter bug. Once the
wait is expressed on the device queue, the same wall interval legitimately belongs to both scopes, so `wait` and `io`
overlap and "body = wait + io + bookkeeping" stops being a partition. The screen was built to stop rather than print
−80.9 ms/token of bookkeeping as if it were a measurement; it did. Do not read the negative column above as a quantity
of host time — read it as the marker that this arm's scopes are concurrent.

### Numerics

All eight arms hash to `c172890df88cdbbfa0627bd320d16614`, identical to §5bb's 384-token output. Both mechanisms are
output-invariant, so nothing on this page is a quality question; these are speed decisions. The contrast is worth
keeping in view: the *genuine* v4.2 rejections are a different class — multi-command staged Metal I/O "produced
different tokens; it is not safe to ship" (`docs/v4.2-expert-streaming-engine.md:319-325`) — and that one would not be
up for re-pricing on a clock at all.

### What this leaves, and what it closes

The feeder idea is not dead, but the cheap version of it is closed. The engine **already** feeds continuously —
`prefetch_issued_per_token=39.30`, `prefetch_adopted_per_token=18.60`, drive duty ≈24 % of the token (§5bb) — and both
shipped knobs that tried to make the *handoff* asynchronous cost more than the bubble they remove. What survives here
is a mechanism, not a guess: **the model-wide timeline, not the device wait, is what makes `.event` lose.** Splitting
that timeline per layer keeps the contiguous-prefix safety argument where it is needed — within a layer, values still
reach the timeline in issue order — while a late read stops gating unrelated releases. It is a small change at one
construction site (`Model.swift:263`) plus the coordinator's per-bucket counters. Registered as the next task; not
shipped, not a default, and it earns nothing here until priced the same way.

**Correction, same night, and it belongs here rather than quietly above: §5bi priced that split and the mechanism claim
in this paragraph is wrong.** `per-layer` delivered the predicted half of its falsifier — `io_host_waits` still 0,
`io_completion_to_fixup_ms` still 0.0000 on all four arms — and left `wait_ms` at 198.990/195.993, i.e. +3.09 then
−1.52 ms/token against `.event`, nowhere near the ~100 ms it had to travel to base's 94. One stall per layer is not
shorter than one shared stall, because each still cannot pass its own layer's slowest in-flight read. **The device wait
is the cost; the timeline granularity was never it.** The −14.27 % above stands; what this paragraph added — the
promising explanation of it — did not survive being measured.

Untouched and still unpriced, named so the next session does not re-find it: `TINYTITAN_DECODE_EXPERT_EXECUTION`
(`hit-fixup` default | `barrier` | `gpu-residency`), and `gpu_classified_*`, which reads 0 in every arm above.

**Provenance.** Machine Mac15,3 Apple M3, 24 GB (25,769,803,776 bytes), macOS 27.0.1, Apple Swift 6.4
(swiftlang-6.4.0.34.1 clang-2100.3.34.1), AC power. Measured artifact is `12ddb80`'s code — `git diff --stat
12ddb80..HEAD` is `docs/qwen38-decode-12gb-measurements.md | 6 +++++-`, documentation only, and the tree was clean for
the whole batch; the binary postdates no source change. Model `models/qwen3.8-flash-next_125B_A6B_4Bit`, tier
`cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G`, `prefetch=1`, no runtime code changed by this unit.

**Exact arm command** (env prefix varied per arm; `${2:+…}` leaves the shipped flag unset for base):

```bash
TINYTITAN_RUNNER_STATS=1 TINYTITAN_KERNEL_STATS=1 \
  [TINYTITAN_EXPERT_IO_SUBMISSION=immediate] [TINYTITAN_EXPERT_IO_SYNC=event] \
  .build/release/TinyTitanCLI --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0 --max-context 262144
```

Every arm exit 0; footer for r1_base1 ends `[stop=maxTokens prefill=10tok/2.99s new=384tok decode=77.41s tok/s=4.961]`.
Per-arm machine state was sampled beside the clock: free memory 77-79 % across all eight, swap used
7,893-8,201 MB — no arm ran under materially different pressure.

**Deviations, all of them.**

1. Three n=32 smokes ran `base → imm → evt` in one order with no control. They are premise and label checks only and
   carry **no** wall conclusion; their `hit → miss_fixup` −29 ms/token reading is exactly what the paired run refuted.
2. Round 1's first arm sits 1.41 % below round 1's last. Deltas are therefore taken against the round's bracket *mean*,
   not against the first arm, which is what keeps `.imm`'s r1 −2.36 % and r2 +0.07 % from being read as a trend.
3. `--ram 12` was not passed. The host derivation on this Mac produces the 11.88 G / 96-slot tier by itself, which is
   within the cap; passing it would have measured the same thing by a different route.
4. No golden baseline: this unit changes no runtime code, and output invariance is proven directly by eight matching
   hashes on the model that matters.
5. `--allow-busy-gpu` was not used, no second model process was started, and nothing was terminated.
6. Nothing here ran in CI and no CI run executed a model. The eight logs are on disk under `/tmp/bh218/` and will not
   survive the machine; the numbers above are transcribed from them.

## 5bi. #219 tests §5bh's one surviving mechanism: 48 timelines instead of one keeps the event arm's counters perfect, does not shorten its wait by a single millisecond, and loses 12.90 % (2026-10-17)

**Question.** §5bh closed with a claim rather than a measurement — that the `.event` arm's loss was the *granularity*
of the model-wide `MTLSharedEvent`, not the device wait itself, because one contiguous-prefix timeline cannot express
"N+1 landed but N has not". If that is right, giving every layer streamer its own timeline should keep the arm's
mechanism (`io_host_waits=0`, `io_completion_to_fixup_ms=0.0000`) and pull `wait_ms` back from ~198 toward base's ~94.
If it is wrong, the split changes nothing and §5bh's closing paragraph was a story. Only the second outcome can close
the idea, so this section buys the split and measures it.

**What shipped.** `TINYTITAN_EXPERT_IO_TIMELINE=shared|per-layer` (`RuntimeConfigurationTypes.swift:89-102`, unset ⇒
`shared`, unknown value ⇒ throws and the load refuses), threaded through `RuntimeConfiguration` to `Model.swift:267-271`,
which builds the timelines through one new factory — `ExpertIOEventCoordinator.makeTimelines(device:numLayers:mode:)`
at `ExpertIOEventCoordinator.swift:45` — and hands layer `L` slot `L` at `Model+LayerStreamers.swift:200`. `shared`
returns the same object in every slot, which is the one-coordinator-per-model topology the shipped engine already ran;
`per-layer` allocates one coordinator each. The sidecar `Model` built for a draft head inherits the mode
(`Model.swift:337`) rather than silently reverting to `shared`. Default is `shared`, nothing is promoted, and the seam
is opt-in only.

**Method.** Eight arms, two interleaved rounds, mirrored order, shipped arm first **and** last — `base, evt, evtpl,
base` then `base, evtpl, evt, base`, so the two event arms swap places across the rounds and neither owns a slot in the
machine's drift. Same model, prompt, 384 greedy tokens, `--max-context 262144`, runner and kernel stats, mains power,
one model process, no `--ram` flag (host derivation gives `cache=11.88G slots=96`, inside the 12 GiB cap). The
falsifier was written down before the first arm: *with `.per-layer`, `io_host_waits` must stay **0** **and** `wait_ms`
must fall from ~198 toward base's ~94; if the counters stay perfect and `wait_ms` does not move, the shared timeline
was not the cost.*

### The clock, against its own bracket

| round | base (first) | evt | evtpl | base (last) | bracket mean | drift |
| --- | --- | --- | --- | --- | --- | --- |
| r1 | 4.925 | 4.371 | 4.308 | 5.051 | 4.9880 | +2.56 % |
| r2 | 4.961 | 4.349 | 4.369 | 4.988 | 4.9745 | +0.54 % |

Paired against the bracket of its own round: `.event` **−12.37 % then −12.57 %** (mean −12.47 %), `.per-layer`
**−13.63 % then −12.17 %** (mean −12.90 %). The comparison that matters is not either of those against base — it is
the two event arms against **each other**, in the same round: `evtpl − evt` is **−0.063 tok/s** in r1 and **+0.020** in
r2. The direction flips, and the mean of −0.022 is 0.49 % of the rate — one fifth of round 1's own +2.56 % drift band.
Splitting the timeline buys nothing measurable.

### The falsifier fired: perfect mechanism, unchanged wait

| arm | `wait_ms` | Δ vs round base | `io_ms` | Δ | `body_ms` | `io_host_waits` | avoided | `c2f_ms` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 base (mean) | 93.736 | — | 70.249 | — | 182.971 | 11,620 | 6,764 | 1.4886 |
| r1 evt | 195.900 | +102.164 | 94.835 | +24.586 | 208.752 | 0 | 18,384 | 0.0000 |
| r1 evtpl | 198.990 | **+105.254** | 95.604 | +25.355 | 212.036 | 0 | 18,384 | 0.0000 |
| r2 base (mean) | 94.363 | — | 69.329 | — | 183.504 | 11,615 | 6,770 | 1.5712 |
| r2 evtpl | 195.993 | **+101.630** | 96.080 | +26.751 | 208.657 | 0 | 18,384 | 0.0000 |
| r2 evt | 197.512 | +103.149 | 96.082 | +26.753 | 210.000 | 0 | 18,384 | 0.0000 |

The first half of the falsifier holds on every event arm in both rounds: 48 timelines release 18,384 host waits to
zero exactly as one timeline did, and the wake bubble is still `0.0000`. The second half fails outright. `wait_ms` for
`.per-layer` is 198.990 and 195.993 — against `.event`'s 195.900 and 197.512 that is **+3.090 then −1.519 ms/token**,
a shuffle inside the counter noise, nowhere near the ~100 ms it had to travel. Per-layer `io_ms` is also unchanged
(+0.769 and −0.002 against its own round's `.event`), so the split did not even move work between the two scopes.

That is the answer, and it overturns §5bh's closing paragraph: **the model-wide contiguous-prefix rule was never the
cost.** The cost is the wait itself. Once the release is a device-side `encodeWaitForEvent`, the decode queue stalls
until the bytes land, and making 48 independent stalls out of one shared stall does not make any of them shorter — each
still cannot pass its own layer's slowest in-flight read. The prefix rule was a real constraint that was not being
hit; §5bh inferred a cause from a mechanism that was merely adjacent to the number.

The feeder counters say the same thing from the other side. Both event arms *do* feed more than the default —
`prefetch_adopted_per_token` is 20.65-20.70 against base's 18.60-18.88 (+1.95/token) and `prefetch_issued_per_token`
39.64-39.68 against 39.32-39.38 — and both are ~12 % slower. Adopting more prefetched experts is worth nothing when
the token then waits on the device for them. That combination — strictly better prefetch, strictly worse clock — is
the strongest form of the §5bb finding that the engine's problem is not feed *volume*.

### The screen refuses both event arms, and this time that is the whole point

`benchmark/tinytitan_host_path_split.py` exit codes, measured without a pipe: **base** `r1_base1` → 0 ("a host-path
change inside the layer loop can address at most 19.341 ms/token of its 38.981"), the other three bases → 1 (family 2
below the absolute bound, bookkeeping only ~18.6-19.3, so price it off the loop). All four event arms — `evt` and
`evtpl`, both rounds → **2, `NOT MEASURED`**: e.g. `body=212.036` is smaller than `wait=198.990 + io=95.604`, breaking
the nesting the split is built on. Same reason §5bh gave for `evt`: once the wait lives on the device, `wait` and `io`
legitimately overlap, so `body = wait + io + bookkeeping` stops being a partition. The negative bookkeeping in the
`evtpl` columns is the marker of concurrent scopes, not a quantity of host time.

### Numerics

All eight arms hash to `c172890df88cdbbfa0627bd320d16614`, identical to §5bb and §5bh's 384-token output. The four
`.per-layer` runs are the first time that code executed at all, and it is output-invariant on arrival, as expected of a
change to *who* the timeline is rather than *what* is read. The shipped path is proved inert from the other side: the
four base arms here now run through the new factory's `shared` branch, and their round means are 93.736 and 94.363
`wait_ms` against §5bh's 93.573 and 92.613 through the old single-coordinator construction — the same token, the same
counters, no new host time. Nothing on this page is a quality question.

### What this closes

Both shipped seams of the asynchronous-handoff idea are now priced and dead: `.event` costs 12.47 % and the 48-way
split of its timeline recovers 0.00 % of it. That retypes §5bh's "what this leaves" from *unpriced promising lever* to
*tested and rejected mechanism*, and it narrows the remaining search: if the wait is the cost, the only feeder work
that can pay is work that removes bytes from under the wait — earlier landing, fewer bytes, or a different place for
the compute — not better bookkeeping about when they land. The `.per-layer` seam stays in the tree as opt-in,
un-promoted, and now with a number attached so nobody pays for it twice. The `TINYTITAN_EXPERT_IO_SYNC=event` path it
rides on is likewise still opt-in and still loses.

Untouched and still unpriced, named so the next session does not re-find it: `TINYTITAN_DECODE_EXPERT_EXECUTION`
(`hit-fixup` default | `barrier` | `gpu-residency`), and `gpu_classified_*`, which reads 0 in every arm above.

**Provenance.** Machine Mac15,3 Apple M3, 24 GB (25,769,803,776 bytes), macOS 27.0.1, Apple Swift 6.4
(swiftlang-6.4.0.34.1 clang-2100.3.34.1), AC power (`pmset -g batt` re-checked in the pipeline immediately before the
batch: "Now drawing from 'AC Power'"). Measured artifact is `c262da7`'s code; the tree was clean from the commit to the
last arm and no source file changed after the binary was built (see deviation 1). Model
`models/qwen3.8-flash-next_125B_A6B_4Bit`, tier `cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G`,
`prefetch=1`. Gates on the committed tree: the full serial package suite, `swift test --no-parallel`, clean — 1,784
tests across seven test bundles (828 + 408 + 157 + 189 + 41 + 63 + 98 in 256 suites), zero failures; the non-passing
lines are the ordinary gated skips (`LibraryContractTests`, `ToolCallingTests`, the real-install and oracle tests, the
pinned Qwen4Exp config test, one tile-boundary kernel test), none of which this change reaches. Within it
`RuntimeConfigurationTests` 10 passed and `ExpertLoadOperationTests` 9 passed; `tools/lint.sh` (all eighteen checks) ok.

**Exact arm command** (env prefix varied per arm; both event flags are unset for base, and `.per-layer` is only
meaningful on the event path):

```bash
TINYTITAN_RUNNER_STATS=1 TINYTITAN_KERNEL_STATS=1 \
  [TINYTITAN_EXPERT_IO_SYNC=event] [TINYTITAN_EXPERT_IO_TIMELINE=per-layer] \
  .build/release/TinyTitanCLI --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0 --max-context 262144
```

Every arm exit 0; footer for r1_base1 ends `[stop=maxTokens prefill=10tok/3.90s new=384tok decode=77.97s tok/s=4.925]`,
for r2_evtpl `[stop=maxTokens prefill=10tok/2.90s new=384tok decode=87.90s tok/s=4.369]`. Per-arm machine state was
sampled beside the clock: free memory 77-80 % across all eight, swap used 7,730.88-8,023.94 MB. The `evtpl` arm drew
its round's **highest** swap in r1 (8,023.94 MB) and its **lowest** in r2 (7,853.25 MB) — the spread is 293 MB, 3.8 % of
the whole range, and it follows the round rather than the arm, so no arm here ran under materially different pressure.

**Deviations, all of them.**

1. The release binary's timestamp is 01:45:21Z and the commit's is 01:50:00Z; the batch started at 01:50:00Z. The
   build therefore precedes its own commit stamp by four and a half minutes, which is the pipeline's order (build →
   tests → lint → commit → batch), not an uncommitted change: no file under `sources/` was touched between the build
   and the commit, so the artifact is `c262da7`'s code.
2. The pipeline's pre-commit clean-suite gate read a 98-test/10-suite run that had been in flight from the previous
   unit (`MemoryBudgetTests`, `TaskMemoryTests` and friends), not a run filtered to the two suites this change touches;
   that gate therefore proved the tree was healthy, not that *these* tests were re-run before committing. They had been
   (10 and 9 passed), and after the batch the full serial package suite was run over the committed tree and is clean, as
   recorded in Provenance. The sequence is: build → touched suites + in-flight suite → lint → commit → batch → full
   suite. Nothing in that order can put an unmeasured binary under a clock, but it is not the order the guide implies.
3. This unit **does** change runtime code, unlike §5bh. It changes no default and no numerics — the `shared` path is
   the pre-existing single-coordinator construction, and eight hashes prove the `per-layer` path emits the same tokens
   — but it is not a measurement-only unit, so it carries tests and a commit rather than only a table.
4. `.per-layer` allocates 48 `MTLSharedEvent`s per model load instead of one. That is not counted anywhere and was not
   measured; it is a memory and allocator cost paid at load, off the decode path, and nothing in these eight arms
   reached a load failure or an allocation warning.
5. `--ram 12` was not passed; the host derivation on this Mac produces the 11.88 G / 96-slot tier by itself, within the
   cap.
6. No golden baseline: `tools/golden-baseline.sh` targets the installed 8-bit Ornith build, which is not what this
   campaign measures, and output invariance here is proven directly by eight matching hashes on the model that matters.
7. `--allow-busy-gpu` was not used, no second model process was started, and nothing was terminated.
8. Nothing here ran in CI and no CI run executed a model. The eight logs are on disk under `/tmp/bi219/` and will not
   survive the machine; the numbers above are transcribed from them.

## 5bj. #220 prices the last unpriced knob on the decode expert path: the shipped hit/fixup split is worth 7.51 %, and moving the partition to the GPU is the first arm that is not slower — by a sixth of what its own first batch said (2026-10-17)

**Question.** §5bi ended by naming the one knob this campaign had never measured, because naming it is cheaper than
letting the next session re-find it: `TINYTITAN_DECODE_EXPERT_EXECUTION` (`hit-fixup` default | `barrier` |
`gpu-residency`). Both non-default modes are worth a measurement for different reasons. `barrier` is documented as
"the former all-experts-after-I/O path as a correctness/performance control for A/B measurements"
(`RuntimeConfigurationTypes.swift:29-34`) — so the whole §5 campaign has been quoting a split it never priced against
the path it replaced. `gpu-residency` moves the hit/miss partition from `DecodeExpertPartition.populate` on the host to
a device-side atomic classification, which is the one change on this path that removes *host work from under the wait*
rather than re-arranging bookkeeping about it — exactly the narrowing §5bi left behind.

**What ships (nothing was written for this section).** The enum and its fail-closed environment read are
`RuntimeConfigurationTypes.swift:29-51` (unset ⇒ `.hitFixup`; unknown value ⇒
`RuntimeConfigurationError.invalidDecodeExpertExecution` and the load refuses). The split is taken only when the mode is
`hitFixup` or `gpuResidency` (`RealForwardRunner+DecodeMoE.swift:125-126`), so `barrier` falls through to commit every
expert after the I/O — there is no separate barrier code to read, which is why it is a control and not a feature. The
GPU arm's residency resources are wired only for that mode (`RealForwardRunner+DecodeLayers.swift:70-71`, from
`ModelExpertIO.swift:162`), the classification is encoded at `RealForwardRunner+DecodeLayers.swift:200`, and the host
then reads the device's lists back under a superset guard (`RealForwardRunner+DecodeMoE.swift:128-174`) that throws if
the GPU ever misses a miss the cache plan expected. That guard is why `gpu_classified_*` read 0 in every prior section:
the mode had never been run.

**Method.** Two batches, one artifact, no source change, mains power, one model process at a time. Batch 1
(`/tmp/bj220/`) is eight arms in mirrored order — r1 `base,barrier,gpu,base`, r2 `base,gpu,barrier,base` — so the
shipped control opens and closes every round and each candidate is measured from two different positions. Batch 2
(`/tmp/bj220b/`) is a confirmation batch written *before* batch 1's magnitude was believed: 3 rounds × 4 arms
(`base, gpu, base, gpu`), 12 arms, no barrier, because a candidate that has to be re-measured is a candidate whose
first number is not yet a number.

**Falsifiers, written first.** *barrier:* if it lands within its round's drift band of base, the shipped phase-1
hit/fixup split buys nothing and every §5 number attributed to it is misattributed. *gpu-residency:*
`gpu_classified_*` must be non-zero — it is, and identically in all eight GPU arms (150 131 / 33 709 / 5 080) — and
host turnaround must fall; if the classifier provably runs and the clock is not up, host partitioning was never on the
critical path, which is what §5bg's screen already predicts.

### The clock

Batch 1, tok/s, with the base bracket's own spread as the drift band:

| round | base1 | barrier | gpu | base2 | base mean | band | barrier Δ | gpu Δ |
|---|---|---|---|---|---|---|---|---|
| r1 | 4.981 | 4.620 | 5.098 | 4.857 | 4.919 | 2.52 % | −6.08 % | **+3.64 %** |
| r2 | 5.093 | 4.533 | 5.211 | 4.863 | 4.978 | 4.62 % | −8.94 % | **+4.68 %** |

Batch 2, same artifact, 02:31–02:48Z:

| round | base1 | gpu1 | base2 | gpu2 | base mean | band | gpu mean Δ |
|---|---|---|---|---|---|---|---|
| c1 | 4.956 | 4.967 | 4.997 | 5.026 | 4.9765 | 0.82 % | **+0.40 %** |
| c2 | 5.013 | 5.158 | 5.260 | 5.267 | 5.1365 | 4.81 % | **+1.48 %** |
| c3 | 5.061 | 5.106 | 5.048 | 5.018 | 5.0545 | 0.26 % | **+0.15 %** |

Five rounds out of five put `gpu-residency` above its own round's base mean, which is the first time in this campaign
that an alternative to a shipped default has been *not slower* in every round it was measured. Under a null of no
effect, five same-sign rounds has probability 1/32 (0.5 × 0.5 × 0.5 × 0.5 × 0.5), so the sign is not chance. Taken
arm-by-arm against each individual base instead of against the round mean, it is 12 positives and 4 negatives out of
sixteen comparisons, and all four negatives are in the confirmation batch — which is the same fact as the previous
paragraph, stated at the resolution the confirmation batch was built to provide.

The magnitude is a different matter, and it does not survive. +3.64 and +4.68 became +0.40, +1.48 and +0.15 — the
confirmation batch's mean is +0.68 % where batch 1's was +4.16 %, and the pooled mean over the five rounds is +2.07 %
with a median of +1.48 %. Nothing about the machine or the arm changed between the two batches except the window, and
the rounds do not rank by drift either (c2's band is 4.81 %, wider than r1's 2.52 %, and its effect is 41 % of
r1's), so this is not "the effect hides inside the drift" — it is "batch 1's two rounds over-stated it, and the
tight-bracket rounds read it at under half a percent." Per the paired-benchmarking rule, an effect that shrinks under
re-pairing was partly drift, and the smaller number is the one to keep.

### The counters say what the clock could not

Per-token wall (= 1000 / the footer's tok/s), against the only counter that replicates. `exposed_ms` is the part of the
read that is *not* hidden behind compute — `(100 − io_hidden_pct)/100 × io_ms`, the quantity §5bi's conclusion said was
the only one worth moving:

| round | base wall | gpu wall | Δ wall | base exposed | gpu exposed | Δ exposed |
|---|---|---|---|---|---|---|
| r1 | 203.29 | 196.16 | −7.14 | 0.133 | 0.068 | −0.065 |
| r2 | 200.88 | 191.81 | −9.08 | 0.135 | 0.069 | −0.067 |
| c1 | 200.94 | 200.14 | −0.80 | 0.190 | 0.095 | −0.094 |
| c2 | 194.69 | 191.85 | −2.84 | 0.155 | 0.076 | −0.080 |
| c3 | 197.84 | 197.55 | −0.29 | 0.195 | 0.103 | −0.092 |
| **mean** | | | **−4.03** | | | **−0.079** |

The counter is stable: −0.065 to −0.094 ms/token across five rounds, all negative, spread 0.030. The clock is not:
−0.29 to −9.08, a factor of 31. 0.079 of 4.03 is **2.0 %** — so the change's demonstrable win is roughly four
hundredths of a percent of a token (0.079 / ~200 ms), and the clock's much larger swing is not the same quantity as the
mechanism. `io_hidden_pct` itself: across all ten base arms it spans 99.65–99.85, across all eight GPU arms 99.78–99.92, and the
two sets separate *within* every round — all sixteen GPU-arm-versus-base-arm comparisons in the same round (one GPU arm
against two bases in batch 1, two against two in batch 2) have the GPU arm above both bases. It is the only counter
in this section that does that.

What did *not* replicate is the attribution batch 1 seemed to offer. `cache_plan_ms` read −0.461 and −0.521 ms/token in
batch 1 — the host planning work the GPU arm removes — and in the confirmation batch it is −0.135, −0.351 and **+0.209**:
c3's GPU arms planned *slower* on the host than its bases. `io_completion_to_fixup_ms` is −0.115, −0.115, −0.110,
−0.093, +0.011: four rounds of a ~0.1 ms/token turnaround gain that is 0.05 % of the token and cannot carry a percent
of clock either. `wait_ms` and `body_ms` are lower in the GPU arm in four and five rounds respectively, but with
magnitudes (−3.5/−4.7 and −6.4/−8.4 in batch 1, −0.7/−1.8/+0.3 and −0.6/−2.7/−0.3 in batch 2) that move with the clock
rather than explaining it — which is the signature of a scope that measures how long the token took, not a quantity of
work removed.

So: the partition does move to the device and it demonstrably does the same job (identical classification counts,
identical output), the exposed read gets reliably a little smaller, and **no counter accounts for more than about 1 % of
the clock gain batch 1 reported, or 2 % of the five-round mean.** The honest claim is that `gpu-residency` is not slower
and is very slightly ahead; the mechanism behind anything larger is not established, and batch 1's own apparent cause
(`cache_plan_ms`) is refuted by batch 2.

### barrier: the control prices the split, and the split is real

| arm | tok/s | wait_ms | body_ms | io_ms | io_hidden_pct | hit_fixup_layers | cache_plan_ms | prefetch adopted/token | read MiB |
|---|---|---|---|---|---|---|---|---|---|
| r1 base (mean) | 4.919 | 95.798 | 185.538 | 69.759 | 99.81 | 11 606 | 4.522 | 18.80 | 75 740 |
| r1 barrier | 4.620 | 114.498 | 198.201 | 66.377 | **0.14** | **0** | 4.262 | **20.66** | **73 856** |
| r2 base (mean) | 4.978 | 93.181 | 183.581 | 71.131 | 99.81 | 11 636 | 4.192 | 18.51 | 76 037 |
| r2 barrier | 4.533 | 116.718 | 202.112 | 67.390 | **0.14** | **0** | 4.585 | **20.57** | **73 945** |

The falsifier did not fire. `barrier` is 6.08 % and 8.94 % slower (mean −7.51 %), and it is slow for a reason the
counters state outright: `io_hidden_pct` goes from 99.81 to 0.14 — the read stops being hidden behind compute almost
entirely — and `wait_ms` gains ~21 ms/token while `io_ms` actually *falls*, by 3.38 and 3.74. The phase-1 hit/fixup
split is therefore worth about 7.5 % of decode, and every §5 number this campaign has attributed to it stands.

And it feeds *better* while losing: 20.57–20.66 prefetched experts adopted per token against base's 18.5–18.9, fewer
bytes read (73.9 GiB against 75.7–76.0), a higher `expert_hit_rate` (0.8495–0.8497 against base's 0.8451–0.8461). That is the
fourth independent instance of the same anti-correlation after §5bb, §5bh and §5bi — better feed, worse clock — and it
is now hard to read as anything else: **the engine's decode rate is not limited by how much the feeder delivers.**

### The screen's verdict on each arm

`benchmark/tinytitan_host_path_split.py`, measured without a pipe. `barrier` → **exit 0** on both rounds:
"a host-path change inside the layer loop can address at most 17.326 ms/token of its 111.378 (16 %, 8.0 % of the
token)" (r2: 18.004 of 113.215). Its family-2 span — the host turnaround the split exists to shorten — is 111–113
ms/token against base's 34–36, which is the same fact `io_hidden_pct=0.14` says from the other side.
`base` and `gpu` → **exit 1** on all four arms screened (batch 1 only — one base and one GPU arm per round; batch 2 ran
no screen): every one has its family 2 (33.200–36.428) below its own absolute bound (34.735–37.092), so a host-path arm
is still worth pricing, but not inside the loop — bookkeeping is only 17.964–19.599 ms/token. The GPU arm's family-2
value (34.887, 33.200) is below its round's base (36.428, 34.351) by 1.54 and 1.15 ms/token, which is the host span the
classifier does take off the path. ~1.3 ms of removed host span against the confirmation rounds' 0.29–2.84 ms wall gain
is at least the right order of magnitude — unlike batch 1's 7.14 and 9.08 ms.

### Numerics

All twenty arms — 8 + 12, three execution modes, two batches — hash to `c172890df88cdbbfa0627bd320d16614`, identical
to §5bb, §5bh and §5bi's 384-token output. `barrier` recomputes the hit experts it did not overlap, so a differing hash
there would have been a correctness finding outranking every number above; it does not differ. `gpu-residency` runs a
different partition *authority* — the device decides, the host only checks the superset — and produces the same tokens,
which is the strongest statement this page can make that the two partitions agree. *Amended by §5bn the same day: "the
device decides" is false, so this sentence's premise goes with it. `RealForwardRunner+DecodeMoE.swift:130-182` reads the
GPU's lists back, sorts them, checks the superset, adds them into three counters — and then calls the* same
`DecodeExpertPartition.populate(topK:missIndices:plan.misses:…)` *the default branch calls, under its own comment
"Execute the CPU partition regardless. It is the authority, and it is the only one that reflects adoption"
(:175-177). There is never a second executed partition to agree with; the hash equality instead proves what §5bn's
52-arm counter table proves — that the added classification changed no byte of the run.* Note that `expert_hits` does vary
arm-to-arm (157 195–157 455) while the output does not: the routing is fixed, which experts happened to be resident
when needed is not, so no two arms share a byte-exact I/O state. That is a reason the clock is noisy, not a reason to
distrust the hash.

### What this closes, and what it leaves

`TINYTITAN_DECODE_EXPERT_EXECUTION` is now fully priced and is the last knob on the decode expert path that was not.
`barrier` is retired as a live option — it is worth its documented weight (−7.5 %) and nothing else. `gpu-residency` is
the first non-default arm in this campaign measured *not slower* than the shipped default, with a replicating but tiny
mechanism (−0.079 ms/token of exposed read, ≈0.04 % of a token) and a clock that will not support a promotion decision.
*Amended by §5bn: "not slower" and "replicating" do not survive two further batches — 24 more paired arms put the same
knob between −0.30 % and +2.12 % depending on whether one contaminated arm is withdrawn, inside control bands of
3.2–13.5 %, and the −0.079 ms/token mechanism is 0.04 % of a token, an order inside that noise. The arm is now "not
measurably different from the default", which is what this sentence's own hedge already said.*
It stays opt-in and un-promoted: the bar here is a number that survives re-pairing, and +4 % did not. Promoting it
would also put a new fail-closed throw (`ModelError.internalInconsistency`, when the GPU's miss list ever fails to
contain the plan's) on the default path of a shipping engine, and it allocates per-layer residency buffers whose memory
cost this section did not measure — on a 24 GB host under a 12 GiB cap, an unmeasured allocation is not a free win.
*Amended by §5bn: measured now, and the marginal cost is zero bytes. `RealForwardRunner+BuildScratch.swift:104-126`
(seven classification buffers, 248 B of payload) and `PreadExpertStreamer.swift:239-248` (the residency table,
512 × 16 B = 8 KiB) are built with no `decodeExpertExecution` branch anywhere in the path, so every shipped run already
pays for them whether or not the knob is on. The fail-closed throw is the caveat that survives, not the memory.*

What it leaves, named so it is not re-found: the one surviving reading of all of §5ba–§5bj is that decode is bounded by
the *token's own critical path*, not by the feed. Nothing tried here — better synchronisation (event −12.5 %,
per-layer −12.9 %), better prefetch (ring depth, horizon, per-layer), better partition *readback* (+0.04 % of a token's
counters, not its clock — §5bn re-pairs that clock in 24 further arms and cannot price it either way) or worse
(I/O overlap −7.5 %) — has moved the clock by an amount that both replicated and was large. The gap between what the
device ledger says a token should take and what it takes has been measured from four sides now and priced from none.

**Provenance.** Machine Mac15,3 Apple M3, 24 GB (25,769,803,776 bytes), macOS 27.0.1, Apple Swift 6.4
(swiftlang-6.4.0.34.1 clang-2100.3.34.1), AC power (`pmset -g batt` re-checked immediately before batch 1: "Now drawing
from 'AC Power'", 100 % charged). Measured artifact is `c262da7`'s code, built 01:45Z; the tree was clean from
`b43a19c` through both batches and **no source file was modified for this section**, so every arm ran the same binary
that §5bi's arms did (`.build/release/TinyTitanCLI` mtime unchanged across all twenty runs). Model
`models/qwen3.8-flash-next_125B_A6B_4Bit`, tier `cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G`,
`prefetch=1`. Gates: none were re-run, because nothing but this document changed — see deviation 8. Per-arm machine
state sampled beside the clock: batch 1 free 78–80 %, swap used 7,874.88–8,079.56 MB (205 MB spread, and the fastest
arm of each round is not the arm with the least swap); batch 2 free 77–80 %, swap 7,824.94–8,181.19 MB. No arm ran
under materially different pressure from its own round.

**Exact arm commands.** Batch 1 and batch 2 differ only in the `TINYTITAN_DECODE_EXPERT_EXECUTION` prefix (unset for
the shipped `hit-fixup`) and in the output directory:

```bash
TINYTITAN_RUNNER_STATS=1 TINYTITAN_KERNEL_STATS=1 \
  [TINYTITAN_DECODE_EXPERT_EXECUTION=barrier|gpu-residency] \
  .build/release/TinyTitanCLI --model models/qwen3.8-flash-next_125B_A6B_4Bit \
  --prompt "Write a detailed essay about the history of computing." \
  --max-new 384 --temperature 0 --max-context 262144
```

All twenty arms exit 0. Two footers, one per batch: batch 1 `r1_gpu` ends
`[stop=maxTokens prefill=10tok/3.02s new=384tok decode=75.33s tok/s=5.098]`, batch 2 `c3_gpu2` ends
`[stop=maxTokens prefill=10tok/2.90s new=384tok decode=76.53s tok/s=5.018]`. Both premise smokes (24 tokens, one arm per
mode) ran before the batches and exited 0, which is what established that `gpu-residency` executes at all.

**Deviations, all of them.**

1. Two batches, twenty arms, and the magnitude is read across them. That is the point of writing the second batch, but
   it means the pooled +2.07 % mean is not one batch's number; the raw clocks are never compared across batches, only
   within-round paired deltas, and the 1/32 sign calculation treats the five round-level deltas as independent draws,
   which two of them arguably are not (batch 1's two rounds share a window).
2. `barrier` is priced from two rounds in one batch and never confirmed. Its sign is not in doubt — the read's exposed
   fraction goes from 0.19 % of `io_ms` in base (0.13 ms/token) to 99.86 % in `barrier` (66.3–67.3 ms/token), a
   deterministic 526-fold change, not a drift artefact — but its −7.51 % magnitude is n=2 and should be read as "about
   seven and a half percent", not to a decimal.
3. In batch 2 the GPU arms sit at fixed positions (2nd and 4th) while batch 1 measured them 3rd and 2nd. Position was
   controlled for in batch 1 and given up in batch 2 in exchange for two GPU samples per round. Against the base arm
   that ran immediately before it, six of the seven GPU arms that have one are faster (+0.13 % to +2.89 %) and the
   seventh, c3_gpu2, is −0.59 %. The eighth arm, batch 1's r1 `gpu`, ran third directly behind `barrier`, so no base
   precedes it; against the base two positions back it is +2.35 %.
4. The residency buffers' memory cost is not measured, and the superset guard's throw has never been observed to fire
   in any of these twenty arms (all exit 0), so it is an untested-in-production failure mode that promotion would put
   on the default path.
5. This unit changes no runtime code — the section is a measurement over the shipped tree, so no test was added and no
   suite was re-run. `swift test --no-parallel`, the 18 `tools/lint.sh` gates and CI were all verified on this exact
   tree in §5bi (`b43a19c`); what is verified *for this change* is `tools/lint.sh docs` and the markdown link check.
6. `--ram 12` was not passed; the host derivation on this Mac produces the 11.88 G / 96-slot tier by itself, within the
   operator's cap.
7. No golden baseline: `tools/golden-baseline.sh` targets the installed 8-bit Ornith build, which is not what this
   campaign measures, and output invariance here is proven directly by twenty matching hashes on the model that matters.
8. `--allow-busy-gpu` was not used, no second model process was started, and nothing was terminated. Nothing here ran
   in CI and no CI run executed a model. The logs are under `/tmp/bj220/` and `/tmp/bj220b/` and will not survive the
   machine; every number above is transcribed from them.

## 5bk. #221 tests the one premise §5e closed by assertion — that nothing in the engine controls GPU clock state — and a stream of dispatches too small to matter buys 5.88 % of pure device-execution time in both rounds, which is the first lever this campaign has found that does not touch bytes (2026-10-17)

### The seam

§5e measured the head GEMV at 9.42 ms back-to-back and 12.73 ms with 50 ms
between launches, attributed the difference to "GPU-state (DVFS/power between
launches)", and closed the lever in one clause: *"nothing in the engine controls
GPU clock state, so this is a measured bound, not an actionable lever."* The
4 ms figure was the **head's** and the bound it implied for the token was
+1.8 %. Neither half of that closing sentence was a measurement. The premise is
an assertion about a thing nobody tried, and the arithmetic priced only one
kernel of the many that live inside the same idle cadence — §5w's ledger says
113.00 ms/token of GPU busy spread over ~48 000 dispatches per 512 tokens, all
of them launched into the same 101.98 ms of per-token device idle.

The falsifier written down before measuring: **if the penalty is device state
and a cheap stream can hold that state, a concurrent stream of near-empty
dispatches must make the measured kernel's `gpuEndTime − gpuStartTime` shorter,
not longer.** If instead the stream only competes for execution resources, the
same quantity must grow. A result that moves neither has not tested the premise.

### The instrument

`benchmark`-free: this is a bench arm, and it is the only thing in `sources/`
that moved. `sources/TinyTitanBench/GPUKeepWarmStream.swift` (new, 138 lines)
owns a **second `MTLCommandQueue`** and a background thread that encodes
`affine_quant_gemv_simd` over an eight-row weight slice — one threadgroup, 256
threads, 20 KiB of weights — commits it, waits, and sleeps.
`TINYTITAN_BENCH_HEAD_KEEPWARM_US` sets the nominal period, bounded
100…20 000 µs; the arm prints `keepwarm_dispatches` and
`keepwarm_probe_us` so the record carries the *achieved* rate, not the requested
one. `TINYTITAN_BENCH_HEAD_CADENCE_MS` and the new flag now parse through one
`envIntFlag`, and a value that is not an integer **refuses** (exit 133, four
refusals verified: keep-warm without cadence, `abc` for either knob, 999999 µs
out of range) instead of reading as "off" — the AUD-307 shape, in the one place
this campaign still had it. The cadence arm also refuses `iterations < 2`, since
it discards the first sample and divides by `iterations − 1`, and prints median,
min and max as well as the mean.

The measured quantity is on-device: `gpuEndTime − gpuStartTime` of the *measured*
launch. Host submission latency is outside it by construction, so this arm cannot
be a launch-path effect.

### The numbers

One arm of `head_affine8` at the qwen38 shape (248 320 × 2560, 8-bit, 635 MB of
weights in ~676 MB of buffers), 40 iterations, 50 ms cadence, controls first **and** last in
each round, treatments mirrored between them. Medians of 39 samples, µs/token-
equivalent of one launch:

| arm | nominal period | dispatches issued | round 1 | round 2 | Δ vs that round's control mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` (position 1) | — | 0 | 10 407.0 | 10 602.7 | — |
| `base_again` (position 5) | — | 0 | 10 950.1 | 10 907.0 | — |
| `kw200` | 200 µs | 3 897 / 3 874 | **10 029.0** | **10 145.0** | **−650 / −610 (−6.08 % / −5.67 %)** |
| `kw1000` | 1000 µs | 1 134 / 1 156 | 10 453.5 | 10 450.8 | −225 / −304 (−2.11 % / −2.83 %) |
| `kw5000` | 5000 µs | 328 / 325 | 10 672.6 | 11 195.7 | −6 / +441 (−0.06 % / +4.10 %) |

The two controls of a round differ by 543.1 µs (round 1) and 304.3 µs (round 2),
so the drift inside a twelve-second round is 2.8–5.2 % and no single-arm number
from this page is worth anything. Against that, `kw200` is **the fastest of all
five arms in both rounds**, and it beats *both* controls of *both* rounds; its
two paired deltas differ by 40 µs. `kw1000` is negative in both rounds.
`kw5000` splits sign. Mean over rounds: **−630 µs (−5.88 %) at 200 µs,
−265 µs (−2.47 %) at 1000 µs, +217 µs (+2.0 %) at 5000 µs** — a dose
response with the densest arm strongest, which is what holding a decaying state
looks like and what a warm-up ramp does not produce: `kw5000` sat at position 2
in round 2 and was the *slowest* arm of that round, 441 µs over the round's
own leading control.

Three secondary readings, all from the printed counters. **(1) The nominal
period is not the achieved one**: 3 897 dispatches inside a ~2.42 s window is
one every 616 µs, not every 200 µs, because each iteration encodes, commits,
waits and sleeps; the achieved rate is what the table carries. **(2) The stream
is not free and won anyway**: at 24.6 µs per dispatch the dense arm burns
~96 ms of device time in a ~2.4 s window, ~4 % of the device, and the measured
kernel still got 5.9 % faster — contention would have moved that number the
other way. **(3) The ambient state is visible in the calibration probe** (one
synchronous spin taken before the stream starts): 47.3 µs in the first arm of
the session and 15.7–24.8 µs in every later one, a 2–3× swing on an identical
20 KiB dispatch, which is the same device-state effect seen from the other end.

### What it is worth, and what it is not

The back-to-back arm run in the same window as these ten measured
9.172 ms/launch (200 iterations, 73.6 GB/s), so today's cadence penalty is
+1.24 to +1.78 ms, **not** §5e's +3.31 ms; the campaign's rule that qwen38
clocks are never compared across batches applies to this bench too. `kw200`
recovers 0.63 ms of that ~1.5 ms — about **40 % of the penalty that is on the
table today**, and it leaves 0.86–0.97 ms still above steady state. Transferred to
§5w's device ledger at the measured *percentage* (−5.88 % of 113.00 ms of GPU
busy = −6.64 ms/token, 217.34 → 210.70) that is **+3.2 % decode**, which would
be the largest single effect this campaign has measured and is precisely why it
is not claimed: the 5.88 % was read on one 635 MB streaming GEMV, and the roles
in that 113 ms differ in length by three orders of magnitude, so there is no
arithmetic route from a kernel's clock penalty to a token's. The number is a
**reason to run the live arm**, and the live arm is the only thing that can
price it. `--ram 12`, 96 slots, `TINYTITAN_GPU_KEEPWARM_US` opt-in, base and
keep-warm mirrored, output hashed.

### Deviations

1. **This is a bench arm, not an engine arm.** Nothing under `sources/TinyTitan`
   moved, so no decode token changed and no runtime default did. The result is a
   premise falsified in one direction, not a throughput gain.
2. **No unit test exists for the instrument.** `TinyTitanBench` is an executable
   target and the suite cannot import it; `tests/` has no `TinyTitanBench`
   directory and adding a library target to hold one knob would be a larger
   change than the probe. What *is* verified is the four refusals and one
   steady-state arm, each run with the exit code recorded, and the fact that the
   reproduction lands within 3 % of §5e's published back-to-back figure
   (9.172 against 9.42 ms). Disclosed rather than worked around.
3. **n=2 rounds, and `kw200` ran at position 3 in both.** The mirroring reversed
   the three treatments, which put the dense arm in the middle twice. Round 2's
   position-2 arm contradicts a monotone ramp, but the position was not fully
   controlled and the next batch must place `kw200` at position 2 or 4.
4. **The spread of a single arm's samples is not a Gaussian I measured**: each
   arm's min-to-max range is 4.4–7.2 ms wide, seven to eleven times the paired
   delta being read off it. The medians are what the deltas use; the ranges are
   printed so a reader can see how much of each arm is unexplained.
5. **Gates run on this exact tree**: `tools/lint.sh format` (ok, 618 Swift files),
   `swiftlint` (0.65.1, `--strict` clean), `sendable`, `force-cast`, `func-length`,
   `file-length` (largest 499), `unbounded-read`, `arch-path`, `stdout-clean`,
   `docs` (ok, 87 documents against 18 gates, 222 ledger commit references resolved)
   and CI's `Check Markdown links` step run verbatim from `.github/workflows/ci.yml`
   (220 local links, 0 broken, exit 0). `swift build -c release` clean. The
   `sendable` gate **failed on the first run** and was right: the
   `unchecked-invariant:` note was written below the class declaration, and the
   gate requires it in the comment block above — the note moved, the gate and the
   build were re-run. No suite was re-run because no runtime file changed.
6. **Mains power, no model process, nothing terminated**; the machine free-RAM
   sample at start was 85 %. No model was loaded and no bytes were read from the
   install by this section, so no golden baseline and no `--ram` decision applies.
7. The logs are `/tmp/bk221/` and do not survive the machine; every figure above
   is transcribed from the printed arm lines.

## 5bl. #222 runs §5bk's keep-warm lever inside the engine, and twenty live arms say what the bench could not: the output is bit-identical at 35 000 dispatches per request, the effect is +0.48 % where its own rounds allow 4 %, and the dose the bench measured best was never actually run (2026-10-17)

**Question.** §5bk measured a −5.88 % cut in the *measured launch's own* `gpuEndTime − gpuStartTime`
when a second `MTLCommandQueue` fed the device one tiny dispatch through the gaps, at an achieved
616 µs period. It closed by saying the bench number is a reason to run the live arm and that the live
arm is the only thing that can price it. This is that arm: the same queue, the same kernel, the same
8 × 2560 × 8-bit spin, wired into the single generation orchestrator so that the CLI and the server
both get it, behind `TINYTITAN_GPU_KEEPWARM_US`.

**What shipped.** `RuntimeGPUKeepWarm` in `sources/TinyTitan/Runtime/Configuration/RuntimeConfigurationTypes.swift`
and `GPUKeepWarmRunner` in `sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift`, hooked at
`sources/TinyTitanLib/ServerModelSession+Generation.swift:24-31` right after the slot is acquired and
stopped by `defer`. Unset means inert; a non-integer or anything outside 100…20000 µs refuses the
request rather than quietly running a control that reads as the arm. Commit `248f795`, handover park
note `8d46b15`. The default is unchanged, as required.

**Method.** Twenty arms, one artifact, clean tree at `248f795`, mains power, one model process,
qwen38 4-bit with **no `--ram` flag** — which on this 24 GiB Mac derives the shipped
`cache=11.88G slots=96` tier, printed by every arm's load line — 384 tokens greedy at
`--max-context 262144`.
Batch 1 (`/tmp/kw222/b1/`, 03:57–04:07Z) is the dose axis mirrored — r1
`base, kw@200us, kw@2000us, base`, r2 `base, kw@2000us, kw@200us, base` — with the shipped control
first *and* last in both rounds, so each candidate is read from two different positions. Batch 2 (`/tmp/kw222/b2/`, 04:07–04:22Z) is the confirmation batch written before batch 1's
magnitude was believed: three rounds of `base, kw, base, kw`, so each candidate arm is measured from
two positions inside one round and the pair straddles the two controls.

**Falsifier, written first** (in #222): *a keep-warm arm that does not beat both controls of its round
in at least two rounds is drift.*

### The dose is not the ask, and this is the finding the bench could not give

The runner prints its **achieved** period at stop. Asked 200 µs it achieved 2036–2049 µs across six
arms (34 332–38 490 dispatches per request); asked 2000 µs it achieved 11 817–12 797 µs (5 645–6 425
dispatches). `Thread.sleep` on this macOS quantises far above the request, so **§5bk's best dose — an
achieved 616 µs — was never run in the engine.** The live arm sits where §5bk's dose curve measured
−2.47 % (1 ms) and +2.0 % (5 ms), i.e. on its flat region, and at 2000 µs the ask it ran at six times
§5bk's worst point. Any verdict here is a verdict on a weak keep-warm, not on the mechanism.

### The clock

Batch 1, against each round's own base mean, with the base-vs-base spread as the drift band:

| round | base1 | kw200 | kw2000 | base2 | base mean | band | kw200 Δ | kw2000 Δ | pair Δ |
|---|---|---|---|---|---|---|---|---|---|
| r1 | 4.935 | 5.063 | 5.260 | 5.344 | 5.1395 | 7.96 % | −1.49 % | +2.34 % | +0.43 % |
| r2 | 5.418 | 5.567 | 5.540 | 5.237 | 5.3275 | 3.40 % | +4.50 % | +3.99 % | **+4.24 %** |

Batch 2, the pre-registered confirmation:

| round | base1 | kw | base2 | kw2 | base mean | band | kw Δ | kw2 Δ | pair Δ |
|---|---|---|---|---|---|---|---|---|---|
| c1 | 5.403 | 5.720 | 5.629 | 5.473 | 5.516 | 4.10 % | +3.70 % | −0.78 % | +1.46 % |
| c2 | 5.543 | 5.601 | 5.698 | 5.688 | 5.6205 | 2.76 % | −0.35 % | +1.20 % | +0.43 % |
| c3 | 5.587 | 5.543 | 5.645 | 5.640 | 5.616 | 1.03 % | −1.30 % | +0.43 % | −0.44 % |

The falsifier's letter is met: three arms (r2 `kw2000`, r2 `kw200`, c1 `kw`) beat both controls of
their round, across two rounds. The magnitude is not. Pairing each round's *arm pair* against its
*control pair* gives +0.43, +4.24, +1.46, +0.43, −0.44 — median **+0.43 %**, mean +1.22 % — and the
confirmation batch alone gives **+0.48 %** with a split sign (2 of 3 positive, which under a null has
probability 0.5). Across all five rounds, 4 of 5 pair deltas are positive; that is p = 0.1875 under a
null of no effect, which is not a finding.

The band is the argument. **In four of the five rounds the pair delta is smaller than that round's own
base-vs-base spread** — +1.46 % inside a 4.10 % band, +0.43 % inside 2.76 %, −0.44 % inside 1.03 %,
+0.43 % inside 7.96 %. Only r2's +4.24 % escapes its 3.40 % band, and it is one round in a five-round
series whose other four land inside. The shipped-drift reading is therefore the honest one: this arm
moves the clock by less than the machine moves it between two identical base arms in the same round.

The session also warmed up as a whole: the ten base arms ran 4.935, 5.344, 5.418, 5.237, 5.403, 5.629,
5.543, 5.698, 5.587, 5.645 tok/s — first to last **+14.4 %** — and `base2 > base1` in four of five
rounds, so an arm placed at position 3 inherits upward drift. That is exactly why batch 2 places its
two keep-warm arms at positions 2 *and* 4 and averages the pair: its +0.48 % is the position-balanced
number, and it is the smallest one.

### What the run does prove

**Output invariance, twenty for twenty.** Every one of the 20 arms — both doses, both positions, both
batches — produced md5 `c172890d` on the response text. A second command queue issuing ~35 000
dispatches per request, on its own timeline, changes nothing the model says. Whatever else this lever
is, it is not a numerics risk, and that is the property §8-style rejections usually cannot offer.

### Deviations, all of them

1. **The best dose was never tested** (above). A sub-millisecond period needs `mach_wait_until`, not
   `Thread.sleep`; the knob as shipped cannot ask for what §5bk actually measured.
2. **No device counters were transcribed.** The manifest carries `tok_s`, `md5`, `dispatches`, free-RAM
   and swap per arm, but not the runner-stats gap counters, so this batch neither confirms nor denies
   §5bk's mechanism at engine scale — it only prices its wall-clock consequence. That is a protocol
   gap against paired-benchmarking rule 5, and it means a follow-up must capture the footer counters.
3. **Machine pressure was high all batch**: swap used 7 778–8 200 MiB throughout, free 77–82 %. All 20
   arms share it, so it does not bias the pairing, but it is not the quietest window this campaign has
   measured in and the bands would likely be tighter on a colder machine.
4. **The two batch-1 arms share one bracket**, so their deltas are not independent observations.
5. **Tree provenance**: the batch ran on `248f795`, which was committed and *unpushed* at measurement
   time (CI 38999848021 / CodeQL 38999848468 were in flight on `36273da`); the tree was clean, so the
   no-dirty-tree rule holds, and the named commit is the artifact.
6. **Gates on the shipped tree** (measured before the batch, at `248f795`): `format` (ok, 619 Swift
   files, after one `[AddLines]` drift in the new file was fixed and the release build re-run clean at
   76.19 s), `swiftlint --strict`, `sendable` (380 files, 0 undocumented), `force-cast`, `func-length`
   (14 baselined, 0 new), `file-length` (largest 499), `unbounded-read`, `arch-path`, `stdout-clean`
   (4 targets in `TinyTitanLib`'s closure, 215 files, no stdout write), `docs` (exit 0), CI's markdown
   link step verbatim (220 links, 0 broken). `RuntimeConfigurationTests` exit 0, 11 tests in 1 suite.
   The RED was a compile failure (`cannot find 'RuntimeGPUKeepWarm' in scope`), not a failing
   assertion, and the first GREEN nearly went unnoticed because a trailing `echo` masked `swift test`'s
   exit code in a pipeline — the verdict was read from the log, then re-run with the exit code taken
   directly.
7. **`--ram 12` was not passed, and `TinyTitanCLI` cannot take it** — `--ram` is the launcher's flag and
   `--ram-budget` the server's. The driver's own header comment says "at --ram 12", so the batch was
   described by an intention rather than by its command line; what every arm actually printed at load is
   `TinyTitan ram profile cache=11.88G slots=96 … (cache budget, not a process target)`, the host
   derivation on this 24 GiB Mac, which is the shipped tier and inside the operator's cap. A re-run that
   tried to honour the header would fail on the unknown flag rather than silently measure a different
   tier, which is why this is a wording defect and not a comparability one. Same fact §5bh, §5bi and
   §5bj each record as their own deviation.

### What it leaves

The keep-warm lever stays unadopted: the shipped default is unchanged, the knob stays opt-in, and the
number to carry forward is **+0.48 %**, not +4.24 %. The single unpriced thing left on this line is a
true sub-millisecond spinner — `mach_wait_until` at the 616 µs §5bk achieved, with the footer gap
counters captured — and the reading here says the expected payoff is under the noise floor, so that
run needs a reason beyond this batch. What §5w's ledger actually said is that 101.98 ms of a 217.34 ms
token is GPU idle in named gaps; §5bk said the device executes those gaps' launches faster when kept
warm; §5bl says feeding them at 2 ms does not buy a user-visible token. The remaining candidate is not
a second queue at all — it is the 52.8 ms `phase1_hit → phase1_miss_fixup` gap, which is bytes, not
clock state.

## 5bm. #223 reads the premise every closure on that gap took on faith — that a miss's 2.7689 MB is all bytes the kernel needs — and the installed layout says 99.852 % of it is, so "fewer bytes per miss" closes at the format, and the 4 096 B that is left is the alignment the split depends on (2026-10-17)

§5w closed *faster bytes* (the burst delivers at 97 % of the wall; §5af's verdict is 3.3–3.4 GB/s and
"no process count, burst split or address set moves the wall by more than the control's own noise") and
named *earlier bytes* and *fewer bytes* as what remained. Both of those readings inherit one unexamined
fact. §5w established that `expert_read_mib / expert_misses` is **2.7689 MB in all five arms, exactly
`expertStride`** — which proves the engine reads a whole stride per miss. It does not prove the stride is
all payload. A miss could be costing bytes nobody consumes: interior padding between the packed tensors, a
per-expert header, or a rounding-up of the last slice. That is the cheapest remaining lever in this
campaign — one number, no model — and it had never been read.

**Method.** Offline, model-free, no process started and no clock taken. Parsed
`models/qwen3.8-flash-next_125B_A6B_4Bit/packed_experts/layout.json` (54 046 742 B) in full — **all 24 576
experts**, 48 layers × 512, not a sample — against the manifest's `expertStride`, and re-derived the
reader's range arithmetic from the tree rather than trusting the summary that prompted the parse.

**What one stride is.** `expertStride = 2 768 896 = 169 × 16 384` pages, and `layer_NN.bin` is exactly
512 × that. Nine roles per expert, tiling from offset 0:

| role | dtype | shape | bytes | offset |
| --- | --- | --- | ---: | ---: |
| `gate` | U32, 4-bit | [640, 2560] | 819 200 | 0 |
| `gate_scales`, `gate_biases` | BF16 | [640, 40] | 51 200 each | 819 200, 870 400 |
| `up` | U32, 4-bit | [640, 2560] | 819 200 | 921 600 |
| `up_scales`, `up_biases` | BF16 | [640, 40] | 51 200 each | 1 740 800, 1 792 000 |
| `down` | U32, 4-bit | [2560, 640] | 819 200 | 1 843 200 |
| `down_scales`, `down_biases` | BF16 | [2560, 10] | 51 200 each | 2 662 400, 2 713 600 |

3 × 819 200 + 6 × 51 200 = **2 764 800 B of payload**. Scanned over every expert: distinct payload totals
`{2 764 800}`, distinct paddings `{4 096}`, **interior holes 0**, `size != stride` 0, `offset !=
expert × stride` 0, roles-per-expert `{9}`. The blob has no header, no metadata prefix (that lives
out-of-band in `layout.json`) and no gap between tensors. Payload is **99.852 %** of the stride; the
unconsumed part is one trailing 4 096 B page-round-up.

**What that is worth.** At §5w's 73.51 demand misses/token the engine reads 203.54 MB per token and
consumes 203.24 MB: **0.301 MB/token of padding**, which at the wall (3.35 GB/s) is **0.090 ms of the
52.803 ms gap** — 0.17 % of the line, 0.04 % of the token. §5bl's +0.48 % was at least inside a
measurable band; this is two orders below one.

**What it closes.** *Fewer bytes per miss* is closed at the format level, not by a benchmark. And the
4 096 B is not recoverable in the way it looks: `offset == expert × stride` — true in 24 576 of 24 576
cases — is what lets a reader address a sub-range at all, and the split that §5c shipped is a sub-range
read. `phase1Bytes` is the manifest's `down` offset (`Model+LayerStreamers.swift:142-153`), the front
fetch is `[0, 1 843 200)` and the tail `[1 843 200, stride)`, i.e. **1 843 200 + 925 696 = 2 768 896**:
the tail *includes* the padding. Cutting it would save 0.09 ms/token and break the "the assembled slot is
the same bytes a whole-region fetch writes" property that `splitMissesResidentWithIdenticalBytes` pins
(`tests/TinyTitan/Infrastructure/Streaming/PreadExpertStreamerTests+Reading.swift:123`). Recorded as
measured-and-refused.

**What it does not close, and one reading it sharpens.** The 52.8 ms line still has two live directions —
earlier bytes (bounded by landing slots, `free=0.99`, and by the RAM tier) and the second and third gap
pairs (26.766 and 16.070 ms/token). And this screen says something about the split that §5c's own
description invites a reader to over-read: if the whole stride predicts the wait (51.93 ms at 3.92,
ratio 1.017) but the *front range alone* predicts only 34.6 ms, then the gap is not "the device waits for
gate+up" — it is the drive being busy for the whole delivery regardless of which part lands first. That is
the signature of a drive-bound pipeline, and it is consistent with the early fixup being in force. The
front/tail split cannot buy the third of the gap a "down is only needed by phase 2" reading would promise.

**Deviations, all of them.** (1) Layout arithmetic only: no read was issued, no process started, no
timing taken, so nothing here is a level and no qwen38 clock is compared to any other. (2) The
bytes-per-token figures reuse §5w's counters (73.51 misses, 203.55 MB/token, captured on `8c61f14`);
the verdict is a *ratio* — 4 096 / 2 768 896 — so it is insensitive to that provenance, and the ms
conversion is labelled with both walls (3.35 and 3.92 GB/s) rather than picking the flattering one.
(3) The scan covers the installed qwen3.8 4-bit layout. Another family has its own stride and needs its
own arithmetic; nothing here claims a format-wide property. (4) `expertStride` was read from the
manifest, not derived; the payload total was derived from `layout.json`'s per-role `size` fields, and
`layout.json` was also the source of the `down` offset that the reader calls `phase1Bytes` — the two
independent routes to 1 843 200 agree. (5) The claim "the tail includes the padding" follows from
`stride − phase1Bytes` in `PreadExpertStreamer+IO.swift:166-180` and the whole-stride fallback in
`expert_io.c:88-93`, both read this session; the residency comment there ("Residency still publishes
only after the tail lands") is why the padding reaches the buffer at all.

## 5bn. #224 puts the campaign's two non-negative levers in one process: 52 arms say the combination buys nothing, and the same 52 say `gpu-residency` executed work byte-for-byte identical to the shipped default (2026-10-17)

**Question.** §5bj priced `TINYTITAN_DECODE_EXPERT_EXECUTION=gpu-residency` at +2.07 % pooled and §5bl priced
`TINYTITAN_GPU_KEEPWARM_US` at +0.48 % position-balanced. Those are the only two arms in §5ba–§5bm that did not measure
slower than the shipped default, and the operator's standing ask — *"maybe we find a combination which will improve
decode tok/s"* — names the one cell neither section tested: both levers live in one process. Nothing else in this
campaign has ever run two non-default knobs together.

**Design.** Batch 1 (`/tmp/c224/b1/`, 06:52–07:29Z) is 4 rounds × 7 arms: `base, X, base, Y, base, Z, base` with
(X, Y, Z) rotating over (kw, gpu, both) by one position per round, so every candidate is bracketed by two shipped
controls inside its own round and no candidate is always first — 28 arms. Batch 2 (`/tmp/c224/b2/`, 07:34:38–08:13:49Z)
is the confirmation batch, written before batch 1's magnitudes were believed: 6 rounds × 4 arms = 24, odd rounds
`base, gpu, base, gpu` (Q1: does `gpu-residency` replicate §5bj?) and even rounds `base, kw, base, both` (Q2: does
adding residency to keep-warm buy anything?). Both batches ran one artifact (`.build/release/TinyTitanCLI` sha256
`e8535a85…e86cb2b6`, rebuilt at `b6da2d7` and re-hashed before arm 1, which is how the parked binary's
mtime-vs-commit loose end got closed), a clean tree, mains power, one model process, qwen38 4-bit with no `--ram`
flag — `TinyTitanCLI` still cannot take one — so every arm loaded the shipped `cache=11.88G slots=96` tier, 384 tokens
greedy at `--max-context 262144`. **52 arms, and no number in this section is compared across the two batches.**

**Falsifiers, written first.** (1) *combination:* `both` is a finding only if its median bracket delta exceeds the
median base-to-base band **and** is positive in at least 3 of 4 rounds. (2) *one lever or two:* if `both` ≤ max(kw, gpu),
the two arms are one mechanism and the combination family closes.

### Batch 1: the clock, and the round that ate it

| round | kw Δ | gpu Δ | both Δ | both−kw | both−gpu | round mean of the three |
|---|---|---|---|---|---|---|
| r1 | +5.40 | +0.14 | +6.98 | +1.58 | +6.84 | +4.17 |
| r2 | −0.63 | +0.05 | +1.12 | +1.76 | +1.07 | +0.18 |
| r3 | **+44.92** | −0.78 | **+41.74** | −3.18 | +42.53 | +28.63 |
| r4 | −2.15 | −3.54 | −3.95 | −1.81 | −0.41 | −3.21 |

**r3 is withdrawn, and it is the finding that matters most of this batch.** `r3_base2` finished at 2.319 tok/s against
its round mates' 5.567/5.372/5.505. Its *work* counters are its neighbours' work: 31 293 evictions and 22 986 reloads —
identical in all 52 arms of both batches — `io_host_waits=11 618`, ring `free=0.99`, and the same output text. Its
*host-side durations* tripled: `wait_ms` 231.572 against ~80, `body_ms` 393.107 against ~164, `cache_plan_ms` 12.184
against 3.4, `router_readback_ms` 0.0818 against 0.012, `cb1_ms` 19.163 against 3.6, `embed_ms` 2.322 against 0.017,
`io_hidden_pct` 97.54 against 99.95. Same bytes, same routing, three times the wall in the operations that only wait:
that is this host thread being descheduled, not a knob. Two of r3's three brackets contain it and they would have
reported **+41 %** and **+45 %** — an order above anything §5ba–§5bm ever measured. Withdrawn, which is what the
falsifier was written for.

With r3's poisoned brackets out, and the 15 surviving controls spanning 4.849–5.567 tok/s (sd 0.220 = 4.1 % of the
mean, 14.81 % end to end):

| arm | clean bracket deltas | mean | median |
|---|---|---|---|
| kw | +5.40, −0.63, −2.15 | +0.87 % | −0.63 % |
| gpu | +0.14, +0.05, −0.78, −3.54 | −1.03 % | −0.36 % |
| both | +6.98, +1.12, −3.95 | +1.38 % | +1.12 % |

Mid-batch I read the additivity durations' sign backwards and reported rate and duration as contradicting each other.
They do not: `body_ms` excess −1.16/−1.29/−1.17 and `wait_ms` −0.70/−0.91/−2.24 against a rate excess of
+1.44/+1.70/+1.74 is agreement in 3/3 rounds (`both` saves more than the sum of its parts). The correction is the kind
of thing the counters exist to catch — and the super-additivity is ~1.2 % of a token's body time measured against
per-round `gpu` deltas that flip sign (+0.14, +0.05, −0.78, −3.54), so the prediction it is being tested against is
itself noise. It is not a lever.

### Batch 2: written for confirmation, delivered by a machine that had stopped cooperating

| round | type | base1 | base2 | band | arm 2 | arm 4 | paired verdict |
|---|---|---|---|---|---|---|---|
| r1 | A | 4.277 | 4.882 | 13.2 % | gpu 4.998 | gpu 5.122 | gpu **+10.49 %** |
| r2 | B | 4.747 | 4.979 | 4.8 % | kw 5.161 (+6.13) | both 4.644 (−4.50) | both−kw **−10.63 %** |
| r3 | A | 4.743 | 4.144 | 13.5 % | gpu 4.812 | gpu 4.339 | gpu **+2.97 %** |
| r4 | B | 3.809 | 3.964 | 4.0 % | kw 3.816 (−1.81) | both 3.824 (−1.61) | both−kw **+0.21 %** |
| r5 | A | 3.821 | 3.944 | 3.2 % | gpu 4.097 | gpu **2.783** | gpu **−11.40 %** as run |
| r6 | B | 4.108 | 3.956 | 3.8 % | kw 4.500 (+11.61) | both 4.059 (+0.67) | both−kw **−10.94 %** |

Deltas are against the round's own control mean; the candidate percentages in parentheses are each candidate against
that same mean, and the `paired verdict` column is the difference of those two percentages — the estimator batch 1's
`both−kw` column uses. These two type-B rows were first printed as the direct arm ratio (`4.644/5.161 − 1 = −10.02`,
`4.059/4.500 − 1 = −9.80`); the correction puts all six contrasts on one estimator, and deviation 9 shows the verdict
is the same under either. One arm is withdrawn on the same mechanical screen used for batch 1's r3 — *body_ms* over
1.20× the mean of its three round mates while its work stays inside the all-arm ranges: **`r5_p4`** at `body_ms`
331.603 against
its mates' 240.315/223.157/232.441, with `expert_misses` 28 391, `expert_read_mib` 74 970.0, 31 293 evictions, 22 986
reloads, `io_host_waits` 11 535 and the same output text as always, while `io_queue_ms` went 0.678 → 2.241,
`expert_load_p99_ms` 8 → 16 ms and `io_hidden_pct` fell to 98.98. No other arm in 52 crosses the bar.

That withdrawal is also the honest answer to Q1, and it is why Q1 is not answered. Scored as run, the seven paired
`gpu` observations across both batches (+0.14, +0.05, −0.78, −3.54, +10.49, +2.97, −11.40) mean **−0.30 %**, 4/7
positive. Scored with `r5_p4` out and the round's surviving arm kept, they mean **+2.12 %** — within 0.05 points of
§5bj's published +2.07 %. **One contaminated arm moves the verdict by the entire size of the effect.** Batch 2's
type-A control bands (13.2 %, 13.5 %) are two to four times the largest gain this campaign has ever credibly measured,
so #224 neither replicates §5bj nor refutes it, and §5bj's own hedge ("a clock that will not support a promotion
decision") is now the load-bearing part of it.

Machine context, sampled per arm and recorded because it decides how much of the above is readable: swap climbed
8 682 → 10 778 MB through batch 2 against batch 1's 8 460 → 9 820 MB, system free held at 76–79 %, and batch 2's
controls ran 3.809–4.979 tok/s where batch 1's ran 4.849–5.567. The keep-warm dose is a third dose again: achieved
period 1 326–1 568 µs with 58 353–70 481 dispatches/request, i.e. every batch 2 keep-warm arm ran the *coarse* dose
§5bl could not schedule deliberately.

### The verdict the two falsifiers actually return

**(1) The combination is not a finding, in either batch.** Batch 1's `both` on the three usable rounds is
+6.98/+1.12/−3.95 — 2 of 3 positive, mean +1.38 %. Batch 2's `both` against its round's control mean is
−4.50/−1.61/+0.67 — 1 of 3 positive, median −1.61 % against a median band of 4.0 %. Both fail.

**(2) The two arms are one mechanism, and the family closes as a no.** The drift-free form of the question is `both`
minus `kw` in the same round, and all six contrasts are: +1.58, +1.76, −1.81 (batch 1) and −10.63, +0.21, −10.94
(batch 2) — **mean −3.31 %, best positive +1.76 %, against control bands of 3.2–11.4 %**. That mean is not an artifact
of the estimator: scored as the direct arm ratio the same six are −2.75, +0.62, −0.44, −10.02, +0.21, −9.80, a mean of
**−3.70 %** whose best positive is **+0.62 %**, so on both readings every contrast sits inside its own band and neither
reading makes the pair look additive. The counters say the same
thing with a cleaner sign in batch 2's three type-B rounds: `wait_ms` +10.18 % and `body_ms` +7.65 % (3/3 positive),
so the stacked arm spends *more* host wait and *more* body time per token than keep-warm alone, every round it ran.
`both` vs `gpu` looked better in batch 1 (+2.50 %) but that is keep-warm's own margin, not a synergy. **Adding
`gpu-residency` to keep-warm buys nothing that keep-warm did not already buy** — which is the operator's question
answered: the combination exists, it is bit-identical, and it does not compound.

### What the mode executes: 52 arms of counters, and the source that explains them

Grouping all 52 arms by what was switched on gives the campaign's cleanest statement about `gpu-residency`:

| counter | base (28 arms) | kw (7) | gpu (10) | both (7) |
|---|---|---|---|---|
| `expert_evictions` | 31 293 | 31 293 | 31 293 | 31 293 |
| `expert_reloads` | 22 986 | 22 986 | 22 986 | 22 986 |
| ring `begins` | 18 001 | 18 001 | 18 001 | 18 001 |
| `expert_misses` mean (range) | 28 726 (28 259–29 047) | 28 948 (28 424–29 383) | 28 646 (28 391–28 963) | 28 834 (28 328–29 127) |
| `expert_read_mib` mean | 75 854 | 76 440 | 75 644 | 76 139 |
| `hit_fixup_layers` = `io_host_waits` mean | 11 617 | 11 669 | 11 600 | 11 638 |
| `prefetch_adopted_per_token` mean | 18.69 | 18.11 | 18.89 | 18.40 |

Every kind's range sits inside the *base arms' own* spread, and the three counters that cannot move unless the
assignment moves are equal to the unit in all 52. The mode changes no executed work — not one streamed byte, not one
fixup, not one adopted prefetch. The source says why, and §5bj did not read it closely enough:
`RealForwardRunner+DecodeMoE.swift:125-129` enters the residency branch only for `.hitFixup` or `.gpuResidency`;
inside, the GPU's lists are read back (`:130-145`), sorted (`:152-153`), checked for the superset property
(`:167-171`) and summed into the three `gpu_classified_*` counters (`:173-175`) — and then
`:175-182`'s own comment takes over: *"Execute the CPU partition regardless. It is the authority, and it is the only one
that reflects adoption; running the GPU's partition would re-fetch an expert already resident."* Both branches end in
the *same* `DecodeExpertPartition.populate(topK:missIndices:plan.misses:…)` call with the *same* arguments. The device
classifies; the host decides. The one executed delta is `RealForwardRunner+DecodeLayers.swift:69-71`, which encodes
`moe_classify_expert_residency` (`:195-208`) **only** when `decodeExpertExecution == .gpuResidency`: one 1-lane
dispatch per MoE layer per token (`moe.metal:132` — `if (lane != 0) return;`, then a serial loop over `top_k` = 10
positions at `:134`), 48/token → 18 432/request, riding an existing commit+wait. That is keep-warm's mechanism
(§5bk, §5bl) at roughly a third of its density (keep-warm ran 58 353–70 481 dispatches/request in this batch's own
arms), which is the reason two knobs that poke the same queue do not compound.

**Marginal memory: zero bytes.** §5bj refused to promote the arm partly because "it allocates per-layer residency
buffers whose memory cost this section did not measure". They are not its allocations. The seven classification
buffers (`residencyHitCount`, `…HitPositions`, `…MissCount`, `…MissPositions`, `…MissExperts`, `…ResolvedSlots`,
`…ResolvedGenerations`) are built at `RealForwardRunner+BuildScratch.swift:104-126` and the residency table at
`PreadExpertStreamer.swift:239-248`, in code with no `decodeExpertExecution` branch anywhere in the path — 248 B of
payload and 512 × 16 B = 8 KiB respectively, already paid by every shipped run whether the knob is on or not. Turning
it on allocates nothing more. The caveat that survives is not memory but reach: the fail-closed
`ModelError.internalInconsistency` throw at `RealForwardRunner+DecodeMoE.swift:167-171` sits inside the
`decodeExpertExecution == .gpuResidency` guard, so it is unreachable unless the knob is on — which is exactly why §5bj's
"on the default path" objection is about *promoting* the mode, not about shipping it opt-in. On the shipped default the
branch at `:183-188` populates the partition and cannot throw.

### Deviations, all of them

1. **Batch 2 was not a good batch and is not quoted as one.** Its bands are 2–4× the effect; one arm withdrawn; Q1
   reported as a range over a single withdrawal decision (+2.12 % / −0.30 %) instead of a number. The pre-registered
   falsifiers, not the pooled means, are what the verdict rests on.
2. **Batch 2's type-A rounds carry two controls for two arms**, so those arms are scored against the round's control
   pair rather than a strict flanking bracket, as §5bl's batch 2 was. Batch 1's 7-arm rounds are properly bracketed.
3. **The keep-warm dose is not one dose, now across three batches.** Achieved period was 2 028–2 042 µs
   (34 860–35 977 dispatches/request) in batch 1's r1–r3 arms, 1 313–1 382 µs (54 779–59 440) in batch 1's r3_p4 and
   both r4 arms, and 1 326–1 568 µs (58 353–70 481) in all six batch 2 keep-warm arms. `Thread.sleep`'s quantisation
   moved with the session and the swap level, so "kw" is at least three different doses across the 52 arms and §5bl's
   deviation 1 (the best dose was never runnable) still stands.
4. **The clock is withdrawn across batches, not just across rounds.** Batch 1 controls 4.849–5.567, batch 2 controls
   3.809–4.979: no figure in either table is comparable to the other's, and none is compared here.
5. **Counters were transcribed** (`wait_ms`, `body_ms`, `io_ms`, `cache_plan_ms`, `io_host_waits`,
   `router_readback_ms`, `io_queue_ms`, achieved period, dispatches, `gpu_classified_*`, and the eight work counters
   grouped above), which closes §5bl's deviation 2.
6. `--ram 12` is again absent from every command line because `TinyTitanCLI` takes neither `--ram` nor `--ram-budget`;
   every arm printed the shipped host derivation (`cache=11.88G slots=96 resident_floor=3.72G estimate=15.60G`,
   `budget=12288MiB`). Same wording defect §5bh, §5bi, §5bj and §5bl each record.
7. **One stale comment found in the path and deliberately not edited.** `sources/TinyTitan/Metal/MoE/moe.metal:117`
   says "Only eight positions are touched"; qwen38's `topKExperts` is 10 (`ModelTypes.swift:282`), and the kernel is
   correct because it loops to the passed `top_k` (`moe.metal:134`). It is filed, not fixed, because that file is
   compiled into the shipped artifact and editing it mid-campaign would change the binary every later arm in §5ba–§5bn
   is measured against. A one-word doc fix belongs on its own commit after the campaign, not inside a measurement.
8. **Batch 2's type-B rounds did not rotate their candidates.** `kw` ran at position 2 and `both` at position 4 in all
   three rounds, so whatever a fixed slot in the round is worth on this machine lands entirely inside that column.
   Batch 1's usable `kw`/`both` pairs sit at different positions in different rounds (p2/p6, p6/p4, p2/p6), so no single
   position effect can hide across its three contrasts. Read batch 2's −10.63 and −10.94 as "stacking did not help, and
   the size of the drop is this design's blind spot", not as a measured cost.
9. **Two batch-2 numbers changed estimator after the first form went in.** `r2` and `r6` were printed as the direct arm
   ratio and now carry the difference of the round's two bracketed deltas, so all six contrasts share one estimator:
   the mean moves from −3.01 % to **−3.31 %**, the best positive stays +1.76 %, and scoring all six as the direct ratio
   gives −3.70 % with a best positive of +0.62 %. Every one of the three readings is negative and inside the bands, so
   no verdict here depends on the choice; the record is here because a published number changed.

**Provenance.** Machine Mac15,3 Apple M3, 24 GB (25,769,803,776 bytes), macOS 27.0.1, Apple Swift 6.4, AC power
(`pmset -g batt`: "Now drawing from 'AC Power'"), `lipo -archs` → `arm64`. Tree clean at
`b6da2d7395c415c90406aebe663926b83f2e2c83` for both batches; artifact sha256 `e8535a85…e86cb2b6`, rebuilt at that
commit and re-hashed before arm 1; binary mtime 06:49:20Z (local 08:49:20, five minutes before batch 1's first arm).
Model
`models/qwen3.8-flash-next_125B_A6B_4Bit`, tier `cache=11.88G slots=96`, `prefetch=1`. One model process at a time, no
`--allow-busy-gpu`, nothing else started in the window. **Gates: none re-run for the batches** — no source file was
modified for this section, only this document and §5bj's four markers. All 52 arms exited 0 and every one hashed to
`c172890df88cdbbfa0627bd320d16614`, the §5bb/§5bh/§5bi/§5bj/§5bl 384-token greedy output, including all seven arms
with both knobs live.

### What it leaves

That batch's controls became the project's published qwen38 levels: the wiki's Benchmarks page withdrew its
2026-08-30 `6.82` row and carries §5bn's and §5r's arms as a protocol of their own (median **5.40 tok/s** over
the 15 surviving controls; wiki `c2591e7`, head `5e13783`; §5s, §5t and §5bd each carry the dated amendment).
That is a *use* of the control arms, not a second verdict — the combination family is still unadopted here.

The combination family closes unadopted: the shipped default is unchanged, both knobs stay opt-in, and the number to
carry is **"+0.48 % from keep-warm alone, and stacking residency on it measured −3.31 %"** — not `both`'s own +1.38 %.
`gpu-residency` is now re-measured twice more and its clock has never been narrower than the bands that contain it, so
the campaign's position on it is §5bj's own: opt-in, un-promoted, and no longer worth a fifth batch. What is left on
this line is what §5bl already named — the 52.8 ms `phase1_hit → phase1_miss_fixup` gap — and §5bm closed its bytes at
the format. The earlier-bytes and the 26.8/16.1 ms gap pairs remain, and both are bounded by slots and the RAM tier,
not by a knob.

## 5bo. The read-ahead advice is a measured no-op, and the control was already there (2026-10-17)

Asked to add a server `--rdadvise` switch and re-test predictor adoption. The first half
needed no code: the policy is already reachable from the server through
`TINYTITAN_RDADVISE_POLICY` (`off | bounded | adaptive | default`, parsed by
`RDAdvicePolicyMode.parse` at `ServerModelSession+Loading.swift:192-195`). This record exists
so the next session does not re-derive it — an earlier attempt used `TINYTITAN_RDADVISE`, saw
the counter unchanged, and concluded the lever was unreachable.

**Method.** 128 greedy tokens, one fresh server per arm on port 8099, rate from the engine's
own `decode_tok_s` footer, arms interleaved because the baseline drifts: six `default` runs
spanned **4.41–5.29 tok/s** (12 %), which is the detection floor for anything below ~5 %.

| policy | `rdadvise_ms` | `rdadvise_calls` | tok/s (paired means) |
| --- | ---: | ---: | ---: |
| `default` | 6.6–7.9 | 9 041–9 064 | 4.83 (6 runs) |
| `off` | **0.000** | **0** | 4.81 (6 runs; one 4.080 outlier) |
| `bounded` | 4.3–4.4 | 6 136–6 430 | 5.08 (2 runs) |

**Verdict: no effect.** The advice genuinely costs ~7 ms of host time per token and removing
it entirely changes the rate by less than the drift — the token is I/O-bound and absorbs the
saving, exactly as §5w's partition predicts (43 ms of host against a 53 ms byte wait). A
server-side `--rdadvise` flag would be churn for a no-op; the real gap is that the env is
undocumented in the server's `--help`.

**The predictor half was not re-run, and should not be.** The space is already measured in
`qwen38-prefetch-predictor-study.md`: L+2 lookahead adopted 19.0 reads/token (the reads did
arrive) and still lost 3.7 %; the mixed-horizon ring lost 9.7 %; temporal predictors score
0.000 precision because the previous token's experts are always resident. Every extra
speculative read is issued into the same saturated device, so its queue cost cancels the bytes
it removes. Re-implementing one would re-run a refuted patch.

## 5bp. Cross-request overlap is memory-gated on a 24 GB host, not flag-gated (2026-10-17)

Asked whether more overlap could still buy a percentage. The intra-token answer is §5w and
§5bo (the byte wait is `bytes ÷ 3.92 GB/s`, and removing host work does not move the rate).
The untested half was **cross-request** overlap, and it does not fit.

`--max-concurrent-sequences` is accepted at any power of two, but `BatchedMemoryBudget`
clamps it to what the machine can hold and says so at load:

| cache tier | per-slot | batch budget | outcome |
| --- | ---: | ---: | --- |
| default 96 slots (`cache=11.88G`) | 3 375 MiB | **0 MiB** | `serving 1 at once` |
| `--ram-budget 8G` (`cache=3.96G`) | 3 375 MiB | 4 096 MiB | still 1 |
| `--ram-budget 4G` (`cache=0.99G`, 8 slots) | 3 375 MiB | ≥ 6 750 MiB | **width 2 loads, then every request fails** `status=500 error=expert cache cannot place` |

So the third row is the load-bearing one: a width can pass the load-time budget and still be
unservable, because two sequences' expert unions do not fit an 8-slot cache. Batching this
model needs a host where `cache + N × 3.3 GB` fits — 48–64 GB, not 24 — and it is not a flag
change.

Recorded for honesty: a probe printed `single(8 slots, 64 tok) 10.67 tok/s`; that request had
returned the 500 above, so the figure is an error path's wall clock and not a measurement.
