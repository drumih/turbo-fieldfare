# Design: layer-major prefill for Qwen3.8-Flash-Next

**Status: removed 2026-09-22. It was implemented behind
`TINYTITAN_PREFILL_LAYER_MAJOR=1` (default off) and measured (2026-09-21) with and
without slot concentration. The expert-read reduction is real and larger than this
design predicted, but the restructure is 2.7x slower in wall clock, its output
diverges from chunk-major, and chunk-major already runs prefill at 90% GPU
occupancy — so the whole line could win at most ~10% even if it were free. Closed
by measurement, it was then deleted from the engine rather than left as a losing
switch: the flag, its residual-per-chunk plumbing and the concentrated-slot
override are gone, and chunk-major is the only schedule.** The sections below are
the design as written; the measurements follow the header.

## Measured, 2026-09-21: the reads fall 92%, the wall rises 2.7x

Both halves exist (un-concentrated at 8136cf6, concentration at 4add9dd), so the
pair could finally be measured. Conditions: 24 GiB M3, macOS 27.0, commit
2538d43, a 9,000-token prompt (three 4,096-token chunks), 32 new tokens,
temperature 0, server path, prefetch at its now-shipped depth 1.

| | chunk-major (shipped) | layer-major + 512 slots |
| --- | ---: | ---: |
| prefill | **379.7 s** | 1030.5 s (**2.7x**) |
| expert reads | 166,119 MiB (162 GiB) | **12,577 MiB (12.3 GiB)** |
| expert hit rate | 0.149 | **0.700** |
| evictions / reloads | 60,738 / 41,029 | 2,219 / 569 |
| GPU busy (sum of roles) | 343.3 s | 346.7 s |
| occupancy | 90.0% | 33.5% |
| kernel roles | — | every role within 2% of chunk-major |

The read reduction is not the ~39% this design predicted but **92%**: with a
layer's whole expert set resident, the band reads that layer's needed experts
once instead of re-streaming them per chunk. The mechanism works as argued. It
also does not matter:

**The GPU does identical work in both arms and the wall is 2.7x longer**, so the
cost is CPU-side time between layer passes — not I/O, not kernels. The kernel
gaps name it: `prefill_moe_reduce -> prefill_gdn_router` totals **663.7 s** over
the 143 layer-chunk boundaries (4.64 s each, against 7.9 s in total for
chunk-major). That is the cost of building a layer's cache outside any overlap:
the previous layer's streamer was dropped (the helper was removed in TT-050), then the next is opened — 512 slots allocated, size-checked, SHA-256 verified and wired — in front
of a GPU with nothing else to run. Chunk-major never pays it because its
per-layer caches open once, during the first chunk, where the open overlaps GPU
work; its only cost is then 74 ms per boundary.

### Correction: on this engine the ordering is what saves the reads

The concentration was added because layer-major alone measured no read reduction
(8136cf6: 113.4 -> 113.8 GiB). That no longer reproduces. Layer-major at the
*shipped* 96 slots, with no concentration at all, read **14,270 MiB** against the
concentrated arm's 12,577 MiB — both against chunk-major's 166,119 MiB, measured
on the same prompt and machine. Each run's 32-token decode portion accounts for
roughly 10 GiB of those totals, so the prefill reads are ~152 GiB for chunk-major
against ~2-4 GiB for either layer-major arm: the reordering alone is what keeps a
layer's working set in its slots across the band's chunks. That makes the
concentration a ~1.7 GiB refinement rather than the enabling half, and it means
the 4add9dd premise should be re-derived, not inherited, by any retry.

### The whole line can win at most ~10% of prefill

Worth stating before anyone builds the pooled retry: on this prompt chunk-major
prefill runs at **90.0% GPU occupancy** (`busy_ms=343,279` of `span_ms=381,500`),
so the exposed expert I/O layer-major could remove is 38 s of 381 s. Even a
version with a free restructure and zero reads could not beat 343 s, i.e. about
-10%. The design's "~15-20% of prefill" came from a 2026-09-05 profile doc where
non-GPU time was 21% of a 651 s 10k prefill; the QSA work has since cut the GPU
half, so the same read saving is worth proportionally less. The flag stays off,
and the cached-pool retry is not worth building for this.

## Open item: the identity check fails on this prompt

On the 9,000-token prompt both layer-major arms returned different greedy text
from chunk-major (`32630aaf…` against `b574afad…`, diverging at the first token;
one run per arm, and the profile echo is identical apart from the flags). This is
a divergence in prefill logits, not a truncation artifact, and not baseline
nondeterminism: a repeat of the chunk-major arm reproduced `32630aaf…` byte for
byte.

Three candidates have been eliminated or narrowed by measurement:

- **Baseline nondeterminism** — eliminated by that repeat.
- **The prefetch ring** — eliminated: layer-major with
  `TINYTITAN_PREDICTIVE_PREFETCH=0` produced the same text (`b574afad…`) and the
  same prefill time (1034.5 against 1030.5 s). The ring is entered only from
  `RealForwardRunner+Decode`, so it never runs during prefill.
- **The 512-slot concentration** — eliminated by the discriminator: layer-major
  with `TINYTITAN_PREFILL_LAYER_SLOTS=96` reproduces `b574afad…` exactly, at
  1042.5 s and 14,270 MiB.

What the two layer-major arms share, and chunk-major does not, is residency: a
layer's expert set stays in its slots across the band's chunks (see the
*Correction* section above — ~2-4 GiB of prefill reads against 152 GiB). The text
therefore follows **which experts are resident**, which points at a numerical
difference between the hit and miss prefill paths rather than at the visit order
or the slot count. That would be a property of the shipped machinery, not of
layer-major alone; with the line bounded at ~10% of prefill it is not worth
another 20-minute run to apportion the last part. It does mean any future revival
must prove byte-identical output under a residency-matched configuration, not
merely at the shipped cache size.

## The problem, measured

Prefill's expert cache is inert. A chunk routes essentially every expert in a
layer against 96 slots, so:

| 8k prompt, chunk 4,096 | |
| --- | ---: |
| expert hit rate | **0.4%** |
| expert bytes read | **111 GiB** |
| whole expert corpus | 68 GiB |
| `io_hidden_pct` | 0.00 |

Each chunk re-streams what the previous one evicted, so cost scales with the
**chunk count**. Raising the chunk 2,048 -> 4,096 cut reads 167.5 -> 111.0 GiB
and prefill 506.4 -> 450.5 s, which confirms the mechanism. 4,096 is the
largest allowed chunk and scratch scales with it, so that lever is spent.

Layer-major takes the expert-visit count to **one per layer** regardless of
prompt length: read each layer's experts once, use them for every token.

## Why this is smaller than it first appears

Two things I assumed were blockers are not.

**The KV cache is already position-addressed.** `kSlot(layer:position:)` and
`vSlot(layer:position:)` take explicit positions, and prefill attention gets
`kvValidCount: startPosition + t` computed from the chunk rather than read from
the cursor. `KVCacheManager.position` is bookkeeping -- validation and where
decode resumes -- not a structural constraint. Writes already happen ahead of
the cursor within a chunk today; a band is the same pattern at larger scale.

**Only the residual has to persist between layers.** `hidden` is
`chunkTokens * hiddenSize * residualStreams` (2,560 x 4 = 10,240 per token,
20 KB at fp16). Everything else in `PrefillChunkScratch` -- `normed`, `q`,
`attn`, the GDN temporaries, `routedX`, `h1` -- is per-layer working space,
lives entirely inside one layer's pass over one chunk, and can stay
chunk-sized and be reused.

That is the difference between ~110 KB/token (all scratch) and 20 KB/token
(residual only): a 8k-token band costs **168 MB**, not 900 MB.

## The shape

```
for band in bands(prompt):              # bounded by residual memory
    for layer in 0..<numLayers:         # experts read ONCE per layer per band
        for chunk in band.chunks:       # order preserved for recurrent state
            run layer on chunk
    kv.advance(by: band.tokenCount)
```

- **Band size** from a residual budget: `band = budget / (residualWidth * 2)`.
  At a 256 MB budget that is ~12.8k tokens, so most coding prompts are one
  band and get the full effect; longer prompts degrade gracefully to today's
  behaviour rather than failing.
- **Chunks within a band stay in order**, so GDN's recurrent state, both conv
  histories and the PLE latch see the same token sequence they do now.
- **QSA selection** per (layer, chunk) still sees only preceding positions of
  its own layer, because chunks are processed in order within the layer.
- **KV** is written positionally per (layer, chunk) exactly as today; only the
  `advance` moves to the end of the band.

## Expected gain

Reads scale with band count rather than chunk count. An 8k prompt at chunk
4,096 is 3 chunks in 1 band: **111 GiB -> ~68 GiB, about -39%**. Expert I/O is
roughly half of prefill, so **~15-20% of prefill**, growing with prompt length
because the chunk count grows and the band count does not.

## What has to change

1. `PrefillChunkScratch`: split the residual out of the per-chunk layout and
   size it per band. Every `scratch.hidden` access becomes band-relative.
2. `executePrefillChunk` (261 lines): invert the loops, and take the layer as a
   parameter rather than iterating internally.
3. `prefillChunkState` / `markCommitted`: currently one commit per chunk; it
   becomes one per band.
4. `kv.advance`: once per band.
5. `reserve` / `validateRange`: must tolerate writes across a whole band ahead
   of the cursor.

## Test plan

The goldens **cannot gate this**. Their prompts are single-chunk and never
leave QSA's dense-exact window, so they would pass whatever it broke. Same
trap as the QSA selection work.

1. **Output identity, long prompt.** A multi-chunk, multi-layer prompt
   (>= 3,306 words, as used for the chunk change) must produce byte-identical
   text with layer-major on and off. Put it behind `TINYTITAN_PREFILL_LAYER_MAJOR`
   so both orders exist on one build.
2. **Recurrent state specifically.** GDN state and the conv histories are the
   parts most likely to break under reordering and least likely to show up in
   a short comparison. Compare `TINYTITAN_ACT_DUMP` activations at the last token
   of each chunk boundary against the chunk-major run.
3. **Both goldens**, to catch anything that leaked into the single-chunk path.
4. **Then** the 8k A/B for the number, reporting `expert_read_mib` alongside
   prefill so the mechanism is visible and not just the outcome.

## Risks

- **Highest-risk change in this runtime.** Recurrent state across a changed
  visit order, with no automated gate.
- **Decode blast radius** is small but non-zero: `reserve`/`validateRange` are
  shared, and the chunk change already showed that touching KV sizing costs
  ~3% of decode.
- **The prize is real but not transformative.** ~15-20% of prefill, against a
  10k prompt that currently costs 450 s. Worth doing, not worth rushing.

## Recommendation

Do it behind an environment flag, default off, and flip the default only after
test 1 and test 2 pass. That keeps a working prefill path throughout, which is
the property the QSA work relied on and the reason those three changes could be
landed one at a time.
