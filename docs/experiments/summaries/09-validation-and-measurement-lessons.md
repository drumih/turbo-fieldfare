# Validation and measurement lessons

[Previous: Sampling, tokenization, and output](08-sampling-tokenization-and-output.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md)

Several conclusions changed because the original measurement answered the
wrong question. The project now separates exact lossless gates from numerical
quality gates, profiler attribution from clean throughput, and training traces
from representative holdouts.

| Current rule | Purpose |
| --- | --- |
| Exact identity for claimed-lossless changes | Detect unintended math or storage changes |
| Distributional quality for reordered floating-point work | Avoid false rejection near ties |
| Interleaved production A/B plus isolated attribution | Separate mechanism from outcome |
| Short, medium, long, and holdout rows | Detect scaling and generalization failures |

## Correctness and quality gates

<a id="meth-01"></a>
### METH-01: Numerical reordering needs a quality oracle

- **Hypothesis:** Exact cross-process tokens or raw logits could validate any
  candidate.
- **Variants tested:** That oracle was applied to MLX attention,
  staged MPP INT4 prefill, and batched routed MoE.
- **Evidence:** Each candidate
  had a strong speed signal but crossed known near ties under reordered
  floating-point work. Reference-relative delta-NLL, top-k agreement, and
  endpoint gates later passed.
- **What changed the conclusion:** The validation
  contract, not the candidates, was wrong.
- **Final disposition:** Methodology
  corrected; three false rejections reversed.
- **Lesson:** Numerical equivalence
  is a distributional claim, not a bit-identity claim.

<a id="meth-02"></a>
### METH-02: Lossless changes still require exact identity

- **Hypothesis:** Once quality gates replaced universal exact comparison, exact
  checks might be unnecessary.
- **Variants tested:** Cache policy, packed-load
  width, FP16 ring storage, and lossless fusion rollbacks.
- **Evidence:** These
  changes claim identical model math and therefore passed identical bits or
  tokens.
- **What changed the conclusion:** Nothing; the gate depends on the
  claim.
- **Final disposition:** Methodology retained.
- **Lesson:** Use the strictest gate appropriate to the candidate's contract,
  not one universal tolerance.

<a id="meth-03"></a>
### METH-03: Production-shaped offsets

- **Hypothesis:** An offset-zero parity buffer represented the standalone INT4
  path.
- **Variants tested:** Aligned test buffers and live resident tensors.
- **Evidence:** `uint` loads passed at offset zero but failed when BF16 planes
  left packed weights only 2-byte aligned.
- **What changed the conclusion:** The
  real layout invalidated the test's alignment assumption.
- **Final disposition:**
  Real-shape offset tests became required.
- **Lesson:** Include live offset and
  stride contracts in kernel fixtures.

<a id="meth-04"></a>
### METH-04: Threadgroup versus grid synchronization

- **Hypothesis:** A local barrier fix could stand in for a complete ownership and
  scope proof.
- **Variants tested:** The production shader inventory was audited
  across threadgroup scratch, disjoint device partials, and queue-ordered
  dispatches.
- **Evidence:** The only proven bug was a reused FP16 prefill
  reduction bank whose old cycle lacked a reader-to-next-writer edge. Other
  cross-threadgroup data relied on disjoint ownership and queue order, not a
  threadgroup barrier.
- **What changed the conclusion:** Synchronization scope
  became an explicit part of the proof.
- **Final disposition:** Methodology and correctness repair retained; the
  concrete failure is recorded in [KV-14](05-attention-and-kv-cache.md#kv-14).
- **Lesson:** Name the producer, consumer, memory
  region, ownership, and synchronization domain before choosing a barrier.

## Measurement discipline

<a id="meth-05"></a>
### METH-05: Profiler throughput is diagnostic

- **Hypothesis:** A profiled run's tok/s could serve as a production comparison.
- **Variants tested:** A clean production row, an attribution profile, and a
  production-behavior timeline row.
- **Evidence:** The clean row measured 6.144
  tok/s. Profile instrumentation disabled the normal command-buffer pipeline and
  measured 4.228 tok/s; the timeline retained production behavior and measured
  6.337 tok/s.
- **What changed the conclusion:** Clean, profile, and timeline rows answer
  different questions: production throughput, detailed attribution, and
  production-behavior timing. They are not interchangeable throughput samples.
- **Final disposition:** Methodology retained.
- **Lesson:** Use profiler
  spans to choose a target and clean A/B rows to decide promotion.

<a id="meth-06"></a>
### METH-06: First-run and thermal state

- **Hypothesis:** One control/candidate pair could establish a small shader win.
- **Variants tested:** Single-order runs, interleaved warm rounds, and balanced
  M5 thermal sequences.
- **Evidence:** Apparent 15-100% swings, including apparent 2x wins on the first
  run, disappeared under interleaving.
- **What changed the conclusion:**
  Order and thermal state explained the signal.
- **Final disposition:** Balanced
  interleaved measurement adopted.
- **Lesson:** Small GPU deltas require an order
  that cannot assign warm-up or throttling to one variant.

<a id="meth-07"></a>
### METH-07: Mechanism counts are not outcomes

- **Hypothesis:** Fewer allocations, dispatches, or command buffers implied a
  faster path.
- **Variants tested:** Argument-buffer rings and merged command
  buffers.
- **Evidence:** The ring cut 21,217 allocations to two but slowed long
  prefill about 9%. The merge cut submissions but slowed decode from 4.013 to
  3.707 tok/s.
- **What changed the conclusion:** Both mechanism counts improved
  while their end-to-end gates regressed.
- **Final disposition:** End-to-end time
  remains the promotion gate.
- **Lesson:** Mechanism counters explain results; they do not replace results.

<a id="meth-08"></a>
### METH-08: Training traces need holdouts

- **Hypothesis:** A policy optimized on one or several related traces would
  generalize.
- **Variants tested:** Single-trace packed layout and trace-trained
  heterogeneous cache allocation.
- **Evidence:** The layout won natural-text
  replay and failed near 4K; the allocation lost on its held-out trace.
- **What changed the conclusion:** Workload-specific locality did not transfer.
- **Final disposition:** Representative holdouts required; see
  [CACHE-04](03-expert-cache-prediction-and-layout.md#cache-04) and
  [CACHE-08](03-expert-cache-prediction-and-layout.md#cache-08).
- **Lesson:** Treat
  layout and cache policies like learned models.

<a id="meth-09"></a>
### METH-09: Greedy repetition-loop artifact

- **Hypothesis:** The long fixture's hit-rate decline revealed a general cache
  defect.
- **Variants tested:** Offline token-period and route-period analysis of
  the existing trace.
- **Evidence:** Late output locked into a period-44 loop with
  94-100% rolling ID match. One cycle touched a median 62 distinct experts per
  layer versus 16 slots. The apparent hit-rate decline from 66.6% to 54.2% was
  cyclic thrash from repeated text.
- **What changed the conclusion:** The fixture
  was an artificial routing workload.
- **Final disposition:** Long-decode
  methodology corrected.
- **Lesson:** Label repetition onset before treating a
  greedy long run as representative generation.

## Boundaries that were not failed experiments

- ANE/Core ML offload was excluded by the platform and architecture decision;
  no failed ANE performance candidate exists.
- Requantizing weights below INT4 was cancelled by the quality floor; no 3-bit
  or 2-bit weight runtime was built.
- Draft-model speculative decoding and n-gram verification stopped at limited
  scope analysis; they were not complete runtime candidates.
- Cache-conditional routing remains a quality-changing hypothesis. Production
  routing was never cache-conditional.
- APFS preallocation remains unexecuted.
- The iPhone port is deferred; all performance results here are Mac results.

## False rejections and attribution corrections

Three experiment conclusions are genuine reversed quality rejections:

1. [MLX full-attention geometry](05-attention-and-kv-cache.md#kv-06) had a
   genuine speed signal and later shipped as v2 after the quality oracle was
   repaired.
2. [Staged affine MPP INT4 prefill](06-prefill.md#pf-12) shipped after a
   distributional quality gate replaced an exact raw-logit comparison.
3. [Batched routed MoE prefill](06-prefill.md#pf-15) shipped after the same
   class of validation error was corrected.

QKV epilogue fusion is a separate attribution warning. An early row was noisy;
a later joint rollback supported the targeted fusion stack but did not isolate
the epilogue's share.

[Previous: Sampling, tokenization, and output](08-sampling-tokenization-and-output.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md)
