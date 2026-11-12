# Wide verify with prompt-copied drafts: Qwen3.8 decode on the 24 GB M3

Status: in progress (2026-11-12). Gates 1 and 2 are not yet measured; the
model-free pieces (draft source, controller, sampler acceptance) are built and
tested, and the gate-2 instrument is built and waiting on the 4-bit install.

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
| 2 | Pass-cost curve (gate 2, stop if a W=8 pass > ~4.5 scalar tokens) | instrument built (`TinyTitanBench passcost`); waits on the 4-bit install |
| 3 | Byte-identical W-row verify kernels | design below; starts after gate 2 |
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
