# Prefill experiments

[Previous: Attention and KV cache](05-attention-and-kv-cache.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Fusions, head, and orchestration](07-fusions-head-and-orchestration.md)

Production prefill uses 128-token chunks, projection-specific GEMV/QMM
selection, bounded staged affine MPP for INT4 projections, and batched routed
MoE. Several candidates improved one shape, allocation count, or host but
failed the M2 long-row gate.

| Current result | Disposition |
| --- | --- |
| Chunk 128, staged affine MPP, and batched routed MoE | Production |
| Shared/fetch overlap v3 | Rejected and removed |
| Shared INT8 QMM, deeper lookahead, and argument-buffer rings | Rejected on M2 |

## Shape and scheduling

<a id="pf-01"></a>
### PF-01: Mixed QMM and GEMV dispatch

- **Hypothesis:** Matrix kernels help only where token count and projection shape
  amortize setup.
- **Variants tested:** QMM and repeated GEMV by projection at `T = 32`.
- **Evidence:** QMM was slower for full Q and neutral for SWA Q; KV/O and
  other projection families benefited.
- **What changed the conclusion:** The
  shape matrix replaced a single global rule.
- **Final disposition:** Production
  policy.
- **Lesson:** Select prefill kernels by projection and shape.

<a id="pf-02"></a>
### PF-02: Chunk size 128

- **Hypothesis:** Larger chunks would amortize per-chunk work without breaking
  the 8 GB budget.
- **Variants tested:** 32 and 128 tokens per chunk on M2.
- **Evidence:** Prefill fell from 15.80 to 9.34 s at 121 tokens, 49.80 to 26.61 s
  at 527, and 92.89 to 52.35 s at 1017.
- **What changed the conclusion:** Nothing;
  memory and correctness gates passed.
- **Final disposition:** Production.
- **Lesson:** A configuration sweep can unlock more than a new kernel.

<a id="pf-03"></a>
### PF-03: Deeper tile lookahead

- **Hypothesis:** Two pending tiles and four experts per tile would hide more
  expert reads.
- **Variants tested:** Production depth one/eight experts and depth
  two/four experts.
- **Evidence:** The candidate regressed 527-token prefill from
  26.61 to 33.38 s and 1017 from 52.35 to 73.88 s because tile dispatches
  doubled.
- **What changed the conclusion:** Scheduling cost exceeded hidden I/O.
- **Final disposition:** Rejected.
- **Lesson:** More lookahead is useful only when
  it does not multiply dispatch work.

<a id="pf-04"></a>
### PF-04: SIMD-cooperative routed MoE

- **Hypothesis:** SIMD cooperation would accelerate grouped routed prefill.
- **Variants tested:** Row/expert-parallel and SIMD-cooperative kernels.
- **Evidence:** Kernel time regressed from 1.410 to 4.619 ms.
- **What changed the conclusion:** Cooperation removed useful row and expert
  parallelism.
- **Final disposition:** Rejected and removed.
- **Lesson:** The decode
  parallelism lesson also applies to prefill's grouped expert work.

<a id="pf-05"></a>
### PF-05: Shared-expert INT8 QMM v2

- **Hypothesis:** The dense shared expert should use QMM once token count exceeds
  the `M = 1` regime.
- **Variants tested:** Legacy GEMV, full QMM v2, and a phase-1
  hybrid.
- **Evidence:** QMM was faster for `M >= 16` and slower at `M = 1`. Medium rows
  improved modestly, but the final 1017-token M2 row regressed from 45.59 to
  52.01 s. The hybrid beat legacy but not full v2.
- **What changed the conclusion:**
  Long-row M2 behavior broke the shape-local promotion case.
- **Final disposition:** Rejected as production.
- **Lesson:** A mixed-length policy needs
  representative short, medium, and long gates.

## Allocation and overlap

<a id="pf-06"></a>
### PF-06: Routed metadata reduction

- **Hypothesis:** One sorted-pair buffer could replace three routed metadata
  buffers.
- **Variants tested:** Three-buffer and one-buffer grouping.
- **Evidence:** Allocations fell from 90 to 30 in the small case and 360 to 120
  in the larger case.
- **What changed the conclusion:** No throughput claim was
  needed; the change improved bounded allocation and clarity.
- **Final disposition:** Production; bounded-allocation cleanup.
- **Lesson:** Record allocation improvements
  without inventing a speed result.

<a id="pf-07"></a>
### PF-07: Argument-buffer reuse v1

- **Hypothesis:** Reusing bindings across streamed tiles would reduce encoder
  work.
- **Variants tested:** Per-tile allocation and a first reuse design.
- **Evidence:** The candidate failed its runtime gate and raised lifetime
  concerns.
- **What changed the conclusion:** Allocation reduction did not improve
  the production schedule.
- **Final disposition:** Rejected and removed.
- **Lesson:** GPU resource reuse requires both lifetime proof and wall-time value.

<a id="pf-08"></a>
### PF-08: Argument-buffer ring v2 and v3

- **Hypothesis:** A small ring could eliminate most argument-buffer allocations.
- **Variants tested:** Per-dispatch allocation and two ring designs on M2 and M5.
- **Evidence:** Allocations fell from 21,217 to two, but M2 1,017-token prefill
  and TTFT regressed about 9%. On M5, v3 improved prefill and TTFT by only
  0.2-0.3%, below the 2% action gate, and one 527-token repeat reversed
  direction slightly.
- **What changed the conclusion:** Allocation count moved dramatically while wall time did not.
- **Final disposition:** Rejected and removed.
- **Lesson:** Mechanism counters are
  diagnostics, not acceptance metrics.

<a id="pf-09"></a>
### PF-09: Shared/fetch overlap on M2

- **Hypothesis:** Decode's shared-MLP/I/O overlap would transfer to chunked
  prefill.
- **Variants tested:** Serial and overlapping shared expert/fetch paths.
- **Evidence:** The candidate improved 121 and 527 tokens by about 1% and 5.3%,
  then materially regressed 1017 tokens.
- **What changed the conclusion:** The
  long M2 row reversed the short and medium gains.
- **Final disposition:**
  Rejected on production M2.
- **Lesson:** A successful decode schedule may not
  transfer to long prefill.

<a id="pf-10"></a>
### PF-10: Shared/fetch overlap v3 on M5

- **Hypothesis:** M5 scheduling and tensor paths might change the overlap
  trade-off.
- **Variants tested:** Balanced, thermally controlled control/candidate rows
  at 527 and 1017 tokens.
- **Evidence:** The candidate improved about 2.2-2.3% at
  both lengths.
- **What changed the conclusion:** The balanced M2 revalidation was
  order-sensitive and materially regressed long prefill. The user judged the
  2.2-2.3% M5 gain too small to justify a separate scheduler branch.
- **Final disposition:** Rejected and removed.
- **Lesson:** A small host-specific win may cost more complexity than it
  returns.

## Installation and startup policy

<a id="pf-11"></a>
### PF-11: Trusted installation receipt

- **Hypothesis:** A trusted receipt could skip model SHA verification and reduce
  time to first token.
- **Variants tested:** Full hash and receipt-trusted load.
- **Evidence:** The receipt avoided 6.741 s of hashing in one 527-token
  attribution row. Clean medians improved 37.95% at 121 tokens and 11.52% at
  527, but regressed 38.65% at 1,017 because full hashing had warmed expert-file
  pages.
- **What changed the conclusion:** The switch changed
  integrity semantics and cache state.
- **Final disposition:** Conditional integrity policy, not a production
  kernel-speed claim.
- **Lesson:** Separate trust policy from
  compute performance.

## Production compute paths and gates

<a id="pf-12"></a>
### PF-12: Staged affine MPP INT4

- **Hypothesis:** Expand only a 32x64 affine INT4 tile into 4 KiB threadgroup
  memory, then use MPP matmul without a full expanded buffer.
- **Variants tested:**
  Packed QMM and bounded staged MPP across projection shapes and quality
  endpoints.
- **Evidence:** Weighted M128 work improved about 73.8%; stable
  512-token prefill improved 11.421% and TTFT 10.947%. A 36-endpoint quality gate
  passed.
- **What changed the conclusion:** An initial 0.09375 logit delta had
  caused a false rejection; distributional quality showed acceptable or improved
  behavior.
- **Final disposition:** Production; reversed rejection.
- **Lesson:**
  Bounded dequantization can unlock matrix hardware without violating the model
  memory rule. See [METH-01](09-validation-and-measurement-lessons.md#meth-01).

<a id="pf-13"></a>
### PF-13: Direct shader-local UInt4

- **Hypothesis:** A direct packed UInt4 path could avoid staged half tiles.
- **Variants tested:** Retired packed QMM, direct UInt4, and current staged MPP.
- **Evidence:** At M128, direct UInt4 beat the retired path by 68.46% but lost to
  staged MPP by 20.40%; at M32 it lost to staged MPP by 38.61%.
- **What changed the conclusion:** The current winner, not the retired baseline, set the gate.
- **Final disposition:** Rejected.
- **Lesson:** Compare new work with today's best
  path.

<a id="pf-14"></a>
### PF-14: QMM threadgroup reuse

- **Hypothesis:** Reusing activation tiles across output rows would accelerate
  the remaining packed QMM families.
- **Variants tested:** Current and reuse
  kernels by projection.
- **Evidence:** Individual families improved 3.2-9.7%
  with exact output, but fresh attribution left only about 0.41% whole-prefill
  opportunity.
- **What changed the conclusion:** Earlier promotions had shrunk the
  target.
- **Final disposition:** Rejected after re-attribution.
- **Lesson:** Local
  speedup times current share determines value.

<a id="pf-15"></a>
### PF-15: Batched routed MoE

- **Hypothesis:** Grouping same-expert token rows would amortize affine setup
  within the existing bounded scratch budget.
- **Variants tested:** Pair and
  batched routes with 495,616 bytes of scratch, isolated kernels, balanced M2
  rows, and quality endpoints.
- **Evidence:** Isolated time fell from 4943.704 to
  3415.312 microseconds, a 30.91% gain. Balanced 121/527 prefill improved about
  2.0-2.2%; quality matched the reference gate.
- **What changed the conclusion:**
  The first removal used invalid exact cross-process output comparison. A later
  same-process/distributional gate reversed it.
- **Final disposition:** Production;
  reversed rejection.
- **Lesson:** Batching can improve streamed MoE without
  expanding model-scale scratch. See
  [METH-01](09-validation-and-measurement-lessons.md#meth-01).

<a id="pf-16"></a>
### PF-16: Long prefill endpoint gate

- **Hypothesis:** Chunked production prefill would retain quality and bounded RSS
  beyond the short fixtures.
- **Variants tested:** Sixteen endpoints through 3707
  tokens against the scalar reference.
- **Evidence:** Aggregate delta-NLL was
  +0.002588; top-1 matched 16/16. Chunked RSS was 888.3 MiB versus 1517.9 MiB for
  scalar, with no model-scale heap staging.
- **What changed the conclusion:** A single sensitive 1,024-token endpoint
  outlier did not persist at neighboring lengths.
- **Final disposition:** Production validation.
- **Lesson:** Long endpoint sweeps
  separate a local tie from a systematic quality change.

[Previous: Attention and KV cache](05-attention-and-kv-cache.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Fusions, head, and orchestration](07-fusions-head-and-orchestration.md)
