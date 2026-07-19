# Decode MoE, INT4, and router experiments

[Previous: Model installation and expert I/O](01-model-install-and-expert-io.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Expert cache, prediction, and layout](03-expert-cache-prediction-and-layout.md)

Decode performance improved most when the runtime exposed independent rows,
overlapped dense work with expert reads, and shortened the affine INT4
instruction chain. Several cooperative or adaptive variants lost because they
reduced parallelism or optimized the wrong cache state.

| Current result | Disposition |
| --- | --- |
| Persistent multi-TG routed MoE with hit-first execution | Production |
| Four-group INT4 paths and safe `ushort` packed loads | Production |
| Multi-TG router GEMV with scalar final selector | Production |
| Cooperative, paired-row, adaptive, and progressive variants | Rejected |

## Dispatch, cache, and overlap

<a id="dec-01"></a>
### DEC-01: One SIMDgroup per output row

- **Hypothesis:** Assigning one SIMDgroup to each output row would expose enough
  parallel work for decode GEMV.
- **Variants tested:** The baseline scalar layout
  and row-parallel INT4/INT8 kernels with a wider MoE threadgroup.
- **Evidence:**
  The bundled stage rose from 2.131 to 2.188 tok/s. The source attributes the
  roughly 9 ms cb2 movement to the wider MoE threadgroup, so it does not isolate
  the standalone GEMV contribution.
- **What changed the conclusion:** Later
  kernels retained row-cooperative SIMD GEMV.
- **Final disposition:** Production
  structure; the original speed row is bundled attribution.
- **Lesson:** Record a
  multi-change stage as a bundle unless a later gate isolates each contribution.

<a id="dec-02"></a>
### DEC-02: SIMD-cooperative MoE inner GEMV

- **Hypothesis:** Several lanes cooperating on each selected expert row would
  accelerate affine decode.
- **Variants tested:** Cooperative per-expert work
  against independent row streams.
- **Evidence:** The candidate was about 2x
  slower; cb2 grew from roughly 230 to 527 ms after 256 row streams collapsed to
  eight.
- **What changed the conclusion:** The dispatch exposed the lost
  parallelism directly.
- **Final disposition:** Rejected.
- **Lesson:** Cooperation
  can cost more than it saves when it serializes independent rows.

<a id="dec-03"></a>
### DEC-03: Persistent multi-threadgroup MoE

- **Hypothesis:** Persistent row workers could remove serialized expert
  execution.
- **Variants tested:** The prior dispatch and a multi-threadgroup
  persistent kernel.
- **Evidence:** cb2 fell from about 239 to 60 ms; decode rose
  from 2.188 to 3.313 tok/s.
- **What changed the conclusion:** Nothing.
- **Final disposition:** Production.
- **Lesson:** Dispatch structure produced a larger
  gain than the later instruction-level refinements.

<a id="dec-04"></a>
### DEC-04: Initial 16-slot expert cache

- **Hypothesis:** Reusing recently routed experts from existing buffers would
  avoid demand reads.
- **Variants tested:** No cache and a 16-slot per-layer LRU.
- **Evidence:** I/O fell from about 166 to 88 ms; decode rose from 3.313 to 4.261
  tok/s.
- **What changed the conclusion:** The cache stayed, while the replacement
  policy later changed to LFU.
- **Final disposition:** Production capacity;
  superseded policy.
- **Lesson:** Small bounded expert reuse captures meaningful
  locality without resident-model growth. The later policy and capacity work
  appears in the [expert-cache summary](03-expert-cache-prediction-and-layout.md).

<a id="dec-05"></a>
### DEC-05: Shared-MLP and I/O overlap

- **Hypothesis:** The dense shared MLP depends on cb1 output but not on routed
  expert bytes, so it can run while misses are read.
- **Variants tested:** Serial
  shared/routed work and a shared command buffer committed before asynchronous
  fetch.
- **Evidence:** cb2 wall time fell from 62 to 48 ms; decode rose from
  4.404 to 4.736 tok/s.
- **What changed the conclusion:** Nothing.
- **Final disposition:** Production.
- **Lesson:** A dependency graph can reveal safe work
  that leaves the I/O critical path.

## INT4 and router kernels

<a id="dec-06"></a>
### DEC-06: MoE INT4 vectorization

- **Hypothesis:** Four-group blocking, `half4` activation reads, and wider packed
  loads would shorten the affine INT4 dependency chain.
- **Variants tested:** The
  scalar group loop and a four-group blocked inner GEMV.
- **Evidence:** Routed GPU
  time fell from 36.5 to 31.3 ms; cb2 fell from 48 to 42 ms; the bench reached
  4.957 tok/s.
- **What changed the conclusion:** Later analysis showed that
  blocking and instruction-level parallelism mattered more than load width.
- **Final disposition:** Production.
- **Lesson:** “Vectorization” here means a
  shorter, more independent instruction stream, not merely a larger integer
  load.

<a id="dec-07"></a>
### DEC-07: Standalone INT4 vectorization

- **Hypothesis:** The MoE four-group technique should also accelerate Q/K/V/O
  and the tied LM-head GEMV.
- **Variants tested:** Byte loads, 32-bit `uint` loads,
  and alignment-safe 16-bit `ushort` loads with `half4` activations and
  four-group blocking.
- **Evidence:** Offset-zero tests passed for `uint`, but live
  resident tensors placed packed weights at 2-byte-aligned offsets and produced
  garbage. The corrected `ushort` path cut LM-head GPU time from about 21.5 to
  16 ms and moved the 16-slot bench from about 4.76 to 5.07-5.38 tok/s.
- **What changed the conclusion:** Real-model offsets disproved the initial
  alignment assumption. Load-width-only movement was small; blocking and ILP
  created most of the gain.
- **Final disposition:** Corrected path in production.
- **Lesson:** Test the real offset contract, not only an aligned buffer base.

<a id="dec-08"></a>
### DEC-08: Multi-threadgroup router GEMV

- **Hypothesis:** The 128-expert router GEMV should run across threadgroups while
  the small global top-k remains separate.
- **Variants tested:** One threadgroup
  for GEMV plus selection, and a multi-TG GEMV followed by a one-TG selector.
- **Evidence:** cb1 fell from about 50 to 44 ms with identical token IDs.
- **What changed the conclusion:** Nothing.
- **Final disposition:** Production.
- **Lesson:** Parallelize the large independent projection and keep the small
  global reduction simple.

<a id="dec-09"></a>
### DEC-09: Parallel K=8 selector

- **Hypothesis:** Parallelizing top-k selection would complement the router GEMV
  split.
- **Variants tested:** Parallel and scalar final selectors.
- **Evidence:**
  The parallel selector took 36.94 microseconds; the scalar selector took 18.86.
- **What changed the conclusion:** Coordination dominated the tiny reduction.
- **Final disposition:** Rejected.
- **Lesson:** The optimal geometry can change
  abruptly at a stage boundary.

<a id="dec-10"></a>
### DEC-10: Gate/up fusion

- **Hypothesis:** Gate and up projections can share activation loads and
  per-group activation sums.
- **Variants tested:** Separate projections, fused projection, and 4, 8,
  and 16 rows per threadgroup.
- **Evidence:** Paired decode measured 5.266 tok/s for the separate control and
  5.378 tok/s for the fused candidate; eight rows per threadgroup beat four
  and sixteen.
- **What changed the conclusion:** Nothing.
- **Final disposition:** Production at the measured
  geometry.
- **Lesson:** Shared-input fusion still needs a geometry sweep.

<a id="dec-11"></a>
### DEC-11: Phase 2 plus reduction fusion

- **Hypothesis:** Fusing down projection with expert reduction would eliminate
  partial-output traffic.
- **Variants tested:** Separate phase 2 and reduction
  against one fused kernel.
- **Evidence:** The isolated path saved about 19
  microseconds and removed the `yPartial` round trip.
- **What changed the conclusion:** Nothing.
- **Final disposition:** Production.
- **Lesson:** A small
  kernel gain becomes more useful when it also removes a dispatch and buffer.

## Rejected geometry refinements

<a id="dec-12"></a>
### DEC-12: Phase-1 paired-row variants

- **Hypothesis:** Each lane could compute two output rows and amortize affine
  work.
- **Variants tested:** `f2`, widened-load `f2`, and miss-subset forms.
- **Evidence:** Some isolated rows looked promising, but production-shaped pairs
  took about 207-212 microseconds versus 194-196 for the control. Miss-subset
  forms also lost in the common four-to-seven-hit state.
- **What changed the conclusion:** Production hit/miss geometry reversed the small synthetic signal.
- **Final disposition:** Rejected.
- **Lesson:** Weight geometry by the runtime's
  state distribution.

<a id="dec-13"></a>
### DEC-13: Phase-1 `u16` packed loads

- **Hypothesis:** Safe 16-bit packed loads could reduce load count without
  changing row geometry.
- **Variants tested:** Byte and `ushort` loads.
- **Evidence:** Median decode moved from 5.961 to 5.973 tok/s; routed GPU time
  fell from about 30.65 to 30.30 ms.
- **What changed the conclusion:** Nothing,
  but the gain remained small.
- **Final disposition:** Production.
- **Lesson:**
  This load-width refinement is distinct from the larger four-group INT4 win.

<a id="dec-14"></a>
### DEC-14: Phase-2 d2, d2p, and d4

- **Hypothesis:** Computing two or four down rows together could preserve vector
  work and share setup.
- **Variants tested:** d2, paired d2, and d4.
- **Evidence:** d2 showed a tiny isolated win, and d2p improved some buckets.
  Repeated end-to-end rows remained mixed. A fresh profile placed phase 2 at
  5.70 ms/token, while the previously measured 0.6-1.1 ms/token improvement
  implied only 0.3-0.6% of the diagnostic step, below run-to-run variation.
- **What changed the conclusion:** Current whole-step share made the attainable
  gain small.
- **Final disposition:** Rejected as default.
- **Lesson:** Reprofile a
  candidate after earlier optimizations shrink its target.

<a id="dec-15"></a>
### DEC-15: Affine group-sum reuse

- **Hypothesis:** Reusing activation sums across affine groups would remove
  repeated work.
- **Variants tested:** Phase-1 x-sum and phase-2 activation-sum
  forms.
- **Evidence:** A development microbenchmark improved from 454.05 to 420.53
  microseconds. In the production wrapper, the default took 309.98 microseconds
  and x-sum took 348.83.
- **What changed the conclusion:** The production wrapper
  reversed the helper result; this gate did not isolate why.
- **Final disposition:**
  Rejected.
- **Lesson:** Measure the optimized helper inside its actual wrapper.

<a id="dec-16"></a>
### DEC-16: Adaptive MoE geometry

- **Hypothesis:** Cache-hit count could select the fastest geometry for each
  token.
- **Variants tested:** Profiled choices across hit states.
- **Evidence:** Qualitative screening found one candidate favored in a six-hit
  synthetic state, while short and long real-decode rows remained mixed.
- **What changed the conclusion:** One state-specific winner did not
  form a reliable policy.
- **Final disposition:** Rejected.
- **Lesson:** Adaptive
  dispatch needs a stable decision surface, not an isolated winning point.

<a id="dec-17"></a>
### DEC-17: Progressive expert execution

- **Hypothesis:** Launching ready experts in chunks would overlap their GPU work
  with later reads.
- **Variants tested:** Chunk sizes one, two, and four plus a
  corrected event-order implementation.
- **Evidence:** After the event-order fix, the 1K/256 candidate diverged from
  the control and regressed from 4.799 to 4.648 tok/s. Earlier chunk-two
  evidence came from a short row; longer rows were nearly flat.
- **What changed the conclusion:** Early
  evidence depended on an unsafe or unrepresentative schedule.
- **Final disposition:** Rejected and runtime-disabled.
- **Lesson:** Validate synchronization
  before trusting speed from overlapping work.

## Hit-first execution

<a id="dec-18"></a>
### DEC-18: Hit-first phase-1 split

- **Hypothesis:** Cached experts can execute while missing experts arrive, with a
  second phase for misses only.
- **Variants tested:** The default hit-first split
  and a rollback with identical forced token IDs.
- **Evidence:** The rollback
  produced 4.518 tok/s; hit-first produced 5.169, a 14.4% advantage.
- **What changed the conclusion:** Nothing; the forced-ID gate removed route divergence
  as a confounder.
- **Final disposition:** Production.
- **Lesson:** Coarse hit/miss
  overlap succeeded where progressive multi-chunk scheduling failed.

[Previous: Model installation and expert I/O](01-model-install-and-expert-io.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Expert cache, prediction, and layout](03-expert-cache-prediction-and-layout.md)
