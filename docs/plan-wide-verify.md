# Wide verify with prompt-copied drafts: Qwen3.8 decode on the 24 GB M3

Status: **gate 2 passed on the dedicated path** (2026-11-12). The existing
chunked prefill fails it (W=8 at 4.62-7.61 scalar tokens), but a dedicated
W-row verify -- the MTP pair schedule widened, `verifyGreedyRows` -- costs
**3.53 scalar tokens at W=8**, measured interleaved with the chunked path in one
process (see "Dedicated-path probe"). The same run shows the blocker: only 58%
of the pass's rows agree with scalar decode, so task 3 (byte identity) is the
next work. Gate 1 is unmeasured (no transcripts on this host).

## Goal and projection

Raise Qwen3.8-Flash-Next 4-bit decode on the MacBook Pro M3 24 GB from ~4.7
tok/s toward 10 tok/s on edit-heavy git-session work, with no change to the
emitted text (AGENTS.md: a speed change may not cost quality).

Why this lever and not another, on this machine alone:

- Expert cache: capacity-bound at 96 slots; a perfect workload profile removes
  only 10-15% of misses in the trace replay, and repeating content removes none.
- SSD: saturated at 3.0-3.3 GB/s; batching tokens cuts I/O per token 8-15%.
- CPU and ANE on weights: already measured as losses or unreachable
  (`docs/cpu-coexecution-plan.md`, `docs/v4.5-ane-prefill.md`).

That leaves GPU work per token, which a multi-row pass amortises: one read of
the dense weights serves every row, and distinct experts per row fall from 10 to
5.3 at W=8 and 4.2 at W=16 on copied text.

Projection (**not measured**), from a pass cost of ~370 ms + 55 ms/row fitted to
the two-row MTP pass (480 ms) and the 200-token prefill, against a 211 ms token:

| share of tokens in accepted drafts | W=8 | W=16 |
| --- | ---: | ---: |
| 50% | 6.4 tok/s | 6.9 tok/s |
| 70% | 7.5 tok/s | 8.5 tok/s |
| 85% | 8.4 tok/s | 10.2 tok/s |
| inside a fully copied span | 9.9 tok/s | 12.8 tok/s |

Plain prose stays near 5 tok/s. W=16 needs extra work: about 20% of layer passes
exceed the 96 slots and would have to be split.

Caveat on the goal itself: coverage depends on the agent's edit format.
Whole-file rewrites raise coverage and tok/s but also emit more tokens for the
same change, so **task time** is measured beside tok/s.

## The prior verdicts this must answer

`docs/qwen38-mtp-investigation-2026-11-09.md` and
`docs/qwen38-mtp-and-overlap-verdict.md` closed two-row MTP at 480 ms/pass
(2.22x a token): 164 ms union expert fetch, 233 ms serial two-row GPU work, 83 ms
turnaround. Wide verify differs in two ways that the gates test rather than
assume: copied tokens route to overlapping experts (the union grows slowly with
W), and one pass carries up to W-1 accepted tokens instead of at most one.

## Tasks

| # | Task | State |
| --- | --- | --- |
| 1 | Draft coverage replay (gate 1, stop under ~40% at min-run 8) | instrument built (`benchmark/draft_coverage_replay.py`); **no transcripts on this machine** -- run when real git-session transcripts exist |
| 2 | Pass-cost curve (gate 2, stop if a W=8 pass > ~4.5 scalar tokens) | chunked path: stop (4.62-7.61); **dedicated path: pass (3.53)**, interleaved run below |
| 3 | Byte-identical W-row verify kernels | **next**: the dedicated pass agrees with scalar decode on 58% of rows at W=8 |
| 4 | Seeded-sampling verify | acceptance rule built and tested (`DraftAcceptance`); wiring lands with task 3 |
| 5 | Draft source and controller | built and tested (`PromptLookupDrafter`, `WideVerifyController`) |
| 6 | GPU-side fetch planning (~20 ms/layer turnaround) | last; small |

### Gate 2 instrument: `TinyTitanBench passcost <install> [rounds] [promptFile]`

Loads through `ServerModelSession.load` (the server's runner, slots and head),
encodes an edit-shaped prompt, then:

1. decodes a greedy reference continuation scalar; the per-token times after a
   32-token warm-up are the scalar baseline (this phase runs no passes, so no
   pass warms a scalar token's experts);
2. replays it and at each round's boundary prefills the next W reference tokens
   as one pass through the production chunked path, times it, restores the state
   captured before it (`captureInferenceState` / `restoreInferenceState`), and
   advances 16 tokens by scalar decode, checking each against the reference.

One width per round, and the pass precedes the scalar decode of its own tokens,
so a pass never finds its own experts cached. Output: per-width median ms, the
ratio to a scalar token, the linear fit, the gate verdict, the last-row greedy
token against the reference (the byte-identity question as a rate), and a JSON
record under `.build/pass-cost/`.

### Gate 2 result (2026-11-12)

Qwen3.8-Flash-Next 4-bit (`qwen38flash`, revision `478474da`), MacBook Pro M3
24 GB, production runner (96 expert slots, 11.88 GB cache budget), 1,461-token
edit prompt, six passes per width. The M3 has no MPP tensor ops, so the prefill
projections ran the fallback kernels. Raw record:
`.build/pass-cost/passcost-20261112T110630Z.json`.

| W | passes (ms) | median | scalar tokens | last row = scalar |
| ---: | --- | ---: | ---: | ---: |
| 2 | 2432, 2219, 1286, 1065, 2551, 2207 | 2212.9 | 7.54 | 6/6 |
| 4 | 1146, 1897, 2211, 2221, 1235, 1088 | 1566.2 | 5.34 | 4/6 |
| 8 | 2146, 2322, 2026, 3276, 3394, 1857 | 2233.9 | **7.61** | 3/6 |
| 16 | 3766, 3394, 4377, 4519, 3818, 3123 | 3791.9 | 12.92 | 5/6 |

Scalar token: 293.4 ms median, 300.0 ms mean (400 samples, no passes in that
phase). Fit: **1,352 ms + 150.7 ms/row**, against the projection's 370 + 55.

Verdict by the plan's own line: **stop**. Even the fastest W=8 pass (1,857 ms)
is 6.3 scalar tokens. At the fit, a pass beats scalar decode only at per-token
acceptance of about 0.97 or more, and certain acceptance at W=16 is at best
~1.25x (16 tokens in ~3.76 s). `WideVerifyCostModel.measuredChunkedPrefill` is
now the controller's default, so it drafts only inside spans it has been
copying perfectly.

Two further findings:

- **The existing path is not identical to scalar decode.** The last row's
  greedy token differed from the scalar reference in 6 of 24 passes, on code
  being copied. That is task 3's identity gap, measured: the GDN recurrence is
  exact (pinned in `GDNKernelTests`), so it lies in the projections, attention,
  routed MoE or head.
- **The pass base, not the per-row cost, is the wall.** A two-row pass here
  costs 1.1-2.6 s, while the purpose-built two-row MTP verify on the same model
  measured 480 ms (`docs/qwen38-mtp-investigation-2026-11-09.md`). The generic
  chunked prefill carries roughly a second of fixed cost per call that a
  dedicated verify does not.

**Open question for the operator.** The stop line was defined on "the existing
edited-turn resume", and that path fails it clearly. Whether a purpose-built
W-row verify (task 3's design) would pass is not measured: its base would start
from the MTP pair's 480 ms, not 1,352 ms, but the same investigation found that
pair still 2.22x a token and bounded by the expert union and serial row work.
Building task 3 to find out is multi-day kernel work. The options are (a) close
the plan here, or (b) fund a narrower probe first -- time the existing
`verifyGreedyPair` schedule generalised to W rows (no identity work) to measure
the dedicated-path base and per-row cost before committing to task 3.

### Dedicated-path probe (2026-11-12): gate 2 passes

The open question above was answered by option (b). `verifyGreedyRows` runs W
rows through one backbone traversal with the routed experts of all rows planned
as one union against the pinned decode slot cache
(`encodeRoutedMoEVerifyRows`, the pair schedule with the row count as a
parameter), and a head shared with the pair verify (`encodeVerifyHeadPair`,
walked two rows at a time: W/2 lm_head reads, an upper bound). The union must
fit the cache, so W <= slots / topK = 9 at 96 slots.

`TinyTitanBench passcost <install> 6 both` alternates the chunked and dedicated
paths round by round in one process, so both share one scalar baseline and one
machine state. Idle M3, 96 slots, 1,461-token edit prompt; scalar token **299.2
ms** median (592 samples). Raw record:
`.build/pass-cost/passcost-20261112T135906Z.json`.

| W | chunked ms | scalar tokens | dedicated ms | scalar tokens | dedicated head ms | dedicated rows = scalar |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 773.0 | 2.58 | **464.0** | **1.55** | 19.1 | 8/12 |
| 4 | 1027.5 | 3.43 | **669.2** | **2.24** | 34.4 | 15/24 |
| 8 | 1382.9 | 4.62 | **1057.5** | **3.53** | 68.3 | 28/48 |

Fits: chunked **827 ms + 135.7 ms/row**, dedicated **259 ms + 103.4 ms/row**.
The dedicated path clears the 4.5 line at W=8 (3.53). At that cost a W=8 pass
breaks even at a per-token acceptance of about 0.75 and, fully accepted, emits
8 tokens in 1.06 s (~7.6 tok/s against 3.34 scalar).

Two things this does **not** show yet:

- **Identity.** Only 28 of 48 rows at W=8 (58%) predicted the token scalar
  decode produced, and the chunked path's last row matched 2 of 6. Wrong rows
  reject correct drafts (acceptance falls) and, worse, change the emitted text,
  which AGENTS.md forbids. The GDN recurrence is exact; the remaining suspects
  are the dense projections (prefill fallback kernels on this M3, no MPP),
  attention, and the head order. This is task 3 and the next work item.
- **The chunked numbers moved between runs** (W=8: 2,234 ms alone this
  morning, 1,383 ms interleaved here), so a single-path run on this machine is
  not a reliable instrument; the interleaved `both` mode is the one to quote.

`WideVerifyCostModel.measuredDedicatedVerify` holds this fit. The controller's
default stays `measuredChunkedPrefill` until the dedicated path is a decode path
that passes the identity gate.

### Identity bisect (2026-11-12): where the pass leaves scalar decode

Instrument: `TinyTitanBench verifyprobe <install> 8` decodes 8 tokens scalar at
position 1,493 of the edit prompt with activation dumps on, restores, verifies
the same inputs as one `verifyGreedyRows` pass with dumps on, and
`benchmark/verify_identity_bisect.py .build/verify-probe 1493 8` compares each
row with scalar decode: the residual after every layer (`L{L}_rows_after`
against `L{L+1}_entry`) and, for layers 0-3, the attention input and output,
post-attention residual, MoE input and MoE output. The dump window is now
`activationDumpFirstPosition`/`activationDumpPositionCount` on the runner, so a
probe can sit anywhere in a context.

Result (7 of 8 rows' argmax agreed; row 0 flipped at a near-tie, logits 0.5
apart): the drift is numeric and it has three sources, in order of size.

| stage | pass vs scalar, max abs | reading |
| --- | ---: | --- |
| layer 0 attention: input, output, post-attention residual | **0** (rows 0-2) | the attention path is exact |
| layer 0 MoE output | 3e-5 to 5e-4, every row | source 1: the routed experts already run decode's phase-1/2 kernels, so the shared expert's multi-row prefill block is the suspect |
| layer 1 attention input | **0.8-1.4** (rows 0-1) from a 3e-5 residual | source 2, dominant: between layer 0's output and layer 1's attention -- where the layer-1 PLE n-gram block is injected |
| layer 0 attention input, rows 3-7 | ~1e-3 (rows 0-2: 0) | source 3: the multi-row input mixer or norm diverges from row 3 |

Everything after is propagation (final residual 0.1-0.4, logits up to 0.67).

**Source 2 was a state-restore bug, not drift, and it is fixed.** The PLE
n-gram hash reads the last `ngramSize` (3) tokens from `pleContext`, which
lives on the runner and was not part of `InferenceAuxiliaryState`: after any
restore, the first two tokens hashed whatever the runner had last processed.
That is why rows 0-1 differed by 0.8-1.4 at layer 1 while row 2 (whose context
is the pass's own tokens) differed by 0.001. It is not specific to speculation:
every prompt-cache and interior-checkpoint resume of a Qwen3.8 install went
through the same restore. `InferenceAuxiliaryState` is now version 2 and carries
the context; a version-1 payload (including disk-cached entries) is refused on
restore, so the caller re-prefills rather than hashing the wrong rows. Re-probed
on the model: layer-1 attention input now 0.001-0.002 in every row, and all 8
rows' argmax agree with scalar decode (7 of 8 before).
Next, in that order: the shared expert in the decode GEMV's order, then the
multi-row projections (hyper-connection gates, attention/GDN, PLE key/value:
`prefill_dequant_affine_qmm_f16_block` runs one thread per output over k,
where decode's GEMV reduces across 32 SIMD lanes); each is done when the bisect
reports 0 for its stage.

### Task 3 design: W rows of one sequence, byte-identical to scalar decode

The pass must produce, for every row, the logits scalar decode would. Two
existing results bound the design:

- the fused batched decode (`docs/batched-decode-phase2c-2026-10-21.md`, T34) is
  bit-identical at widths 2 and 4 once the multi-row GEMV keeps the single-row
  accumulation order (`dequant_int4_gemv2_simd`); the prefill tile kernels do
  not, which is why the pair verify diverged on one prose prompt;
- the pass has rows of **one** sequence: row i's attention sees rows < i of the
  same pass, and row i's GDN state follows row i-1's.

So the verify reuses the decode kernels' per-row order with weights read once:
multi-row GEMV for every dense projection, causal attention over the cache plus
the pass's own earlier rows, the routed MoE over the union of the rows' experts,
the GDN recurrence stepped row by row, and one W-row lm_head read.

**GDN rollback to an arbitrary accepted row.** The pair verify snapshots GDN
state after row 0. With W rows the accepted count is known only after the head,
and a per-row state snapshot would cost ~72 MB per row (2 MB/layer) -- about
1 GB at W=16. Instead: snapshot once before the pass, keep each GDN layer's
per-row recurrence inputs (post-conv q/k/v, a, b: ~16 KB/row/layer, ~10 MB at
W=16), and on partial acceptance restore the snapshot and re-step the recurrence
over the k accepted rows. That is state arithmetic only, no weight reads; the
conv tail is rebuilt from the same retained pre-conv rows. KV rows of full
attention layers rewind by cursor; PLE's rolling window by
`rewindPLE(acceptedRows:passRows:)`.

### Task 4: sampling under the default presence penalty

Speculation engaged only under `isPureGreedy`, and the default presence penalty
of 1.5 switched it off. `DraftAcceptance.resolve` samples row i with the history
scalar decode would have had (committed history plus drafts 0..<i) at position
boundary + i, and stops at the first sample that differs from its draft. With a
fixed `seed` the sampler's seed is a function of position alone
(`Sampler.seedFor`) and the presence table folds multi-token appends and re-seeds
on a shrink, so the emitted text is byte-identical to scalar decode given
identical row logits. Without a seed each emitted token is still the target's
own sample given an accepted prefix, so the distribution is the target's.
Requests with a structured-output constraint stay scalar: the grammar advances
per committed token.

### Task 5: draft source and controller

`PromptLookupDrafter` is the replay tool's `NgramIndex` in Swift (longest n in
8...2, most recent prior occurrence, may run into its own suffix), cross-checked
against a naive restatement on random streams. `WideVerifyController` estimates
per-position acceptance per matched n (geometric model, forgetting at 0.98 per
pass) and picks the draft length with the best expected tokens per millisecond
under `WideVerifyCostModel`; it drafts only when that beats a scalar token. The
cost model's defaults are this plan's projection and are replaced by the gate-2
fit.
