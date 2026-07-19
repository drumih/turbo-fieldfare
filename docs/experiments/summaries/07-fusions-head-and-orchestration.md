# Fusions, head, and orchestration experiments

[Previous: Prefill](06-prefill.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Sampling, tokenization, and output](08-sampling-tokenization-and-output.md)

Targeted fusions and dependency-aware I/O overlap survived. Larger wrapper
fusions, extra queues, cross-token event chains, and resource reuse often lost
end-to-end even when they removed dispatches or allocations.

| Current result | Disposition |
| --- | --- |
| QKV epilogue, layer tail, QKV GEMV, and single-wait pipeline | Production |
| Row Fusion D greedy head | Production |
| Monolithic, chunk, tiled, wide, and late-wait head variants | Rejected |
| Extra queue, encode-ahead, merge, and reuse candidates | Rejected |

## Targeted fusions

<a id="orch-01"></a>
### ORCH-01: QKV epilogue fusion

- **Hypothesis:** Per-head norm, RoPE, and KV write could share one projection
  epilogue.
- **Variants tested:** Separate operations and the fused epilogue.
- **Evidence:** The first full run was noisy. A later joint rollback showed the
  enabled QKV-epilogue/layer-tail stack ahead by about 0.076 tok/s and lower by
  2.9 ms/token. That row supports the stack but does not isolate the
  epilogue's share.
- **What changed the conclusion:** Better paired measurement recovered a
  small stack-level signal.
- **Final disposition:** Production; exact attribution
  remains bundled.
- **Lesson:** Do not assign a joint rollback's gain to one
  member of the bundle.

<a id="orch-02"></a>
### ORCH-02: Layer-tail fusion

- **Hypothesis:** Residual, scale, and the following norm could share one kernel.
- **Variants tested:** Separate tail operations and the fused tail.
- **Evidence:** Parity held. In the historical paired attribution row, the
  fused path measured 2.693 tok/s and 371.3 ms/token, versus 2.056 tok/s and
  486.3 ms/token for the legacy chain. The absolute row predates the current
  stack and is not a current baseline.
- **What changed the conclusion:** Nothing.
- **Final disposition:** Production.
- **Lesson:** Small
  targeted fusions work when their data lifetime and dependency boundary align.

<a id="orch-03"></a>
### ORCH-03: Monolithic post-attention/pre-FFN fusion

- **Hypothesis:** Combining most of the block middle would remove more launches
  and intermediates.
- **Variants tested:** A broad monolithic wrapper and the
  legacy path.
- **Evidence:** The candidate produced 1.811 tok/s versus 2.756 for
  the control. Later, narrower post-attention and shared phase-1 fusions
  succeeded.
- **What changed the conclusion:** The failure identified a bad
  fusion boundary, not a universal fusion no-go.
- **Final disposition:** Rejected.
- **Lesson:** Fuse operations with shared data and compatible geometry, not every
  adjacent operation.

<a id="orch-04"></a>
### ORCH-04: Fused QKV GEMV

- **Hypothesis:** Q, K, and V projections could share activation reads.
- **Variants tested:** Separate and fused projection dispatch.
- **Evidence:** A repeated 256-token rollback gate measured the fused path at
  6.215 tok/s and the unfused control at 6.110 tok/s.
- **What changed the conclusion:** The initial short real-decode gate regressed
  despite a synthetic win; later repeated rows and a post-promotion rollback
  favored fusion.
- **Final disposition:** Production; reversed rejection.
- **Lesson:** A modest fusion belongs in the default only after a repeated
  whole-step gate.

<a id="orch-05"></a>
### ORCH-05: LM-head Fusion D family

- **Hypothesis:** Final norm, INT4 head, softcap, and selection could collapse
  into a faster chain.
- **Variants tested:** Chunk, row, tiled, wide, and late-wait
  designs.
- **Evidence:** The chunk version improved a local stage but lost about
  0.9% end-to-end. The row design later improved the repeated short median by
  1.0% and the long pair by 2.8%, removed CPU argmax, and became the default
  greedy head. Tiling cut head GPU time from 14.2 to 13.1 ms while decode fell
  from 5.962 to 5.926 tok/s; late wait was worse.
- **What changed the conclusion:**
  The row redesign passed whole-step gates that the other variants failed.
- **Final disposition:** Row Fusion D is production; the other variants were
  rejected.
- **Lesson:** Judge each design in a family separately. Sampled-head fusion is
  covered separately in [OUT-02](08-sampling-tokenization-and-output.md#out-02).

## Shader and head variants

<a id="orch-06"></a>
### ORCH-06: Shader-only sweep

- **Hypothesis:** Packed width, paired rows, remainder specialization, and local
  attention mappings could yield low-risk speedups.
- **Variants tested:** Scale/bias SIMD broadcast, threadgroup-staged
  activations, `ushort`/`uint` packed loads, metadata AoS/AoSoA, two rows per
  SIMDgroup, INT8 paired/multi-row loads, down-row padding, paired-row phase 1,
  SIMD-per-position attention, tiled attention, full-GQA attention, and an
  attention chunk sweep.
- **Evidence:** No variant earned promotion. INT8 already reached
  about 114 GB/s. Apparent first-run 2x wins vanished under interleaved warm
  rounds.
- **What changed the conclusion:** Thermal and first-run state explained
  the large early signals.
- **Final disposition:** Rejected sweep.
- **Lesson:**
  Interleave warm controls before believing a small shader delta.

<a id="orch-07"></a>
### ORCH-07: RoPE frequency tables

- **Hypothesis:** Precomputed frequencies would remove repeated trigonometric
  work.
- **Variants tested:** Current computation and table lookup.
- **Evidence:**
  The table path was neutral or slower.
- **What changed the conclusion:** Extra loads and indexing offset the saved
  arithmetic.
- **Final disposition:** Rejected.
- **Lesson:** Replacing arithmetic with memory traffic is not automatically a
  win on a GPU.

<a id="orch-08"></a>
### ORCH-08: LM-head width and tile sweep

- **Hypothesis:** Wider threadgroups and different vocab tiling would improve the
  large tied head.
- **Variants tested:** Several widths and tile layouts.
- **Evidence:** Isolated movement was small and no repeatable end-to-end gain
  appeared.
- **What changed the conclusion:** The system path absorbed or reversed
  local changes.
- **Final disposition:** Rejected.
- **Lesson:** Stop sweeping a
  family when end-to-end sensitivity stays below noise.

## Scheduling and synchronization

<a id="orch-09"></a>
### ORCH-09: Single-wait I/O pipeline

- **Hypothesis:** Coarser waits and a pipelined hit/miss schedule would reduce
  per-layer serialization.
- **Variants tested:** Production pipeline and rollback
  forms.
- **Evidence:** Production measured 6.416 tok/s. The full pipeline rollback
  measured 5.545 tok/s, and the hit-split-only rollback measured 5.926 tok/s.
- **What changed the conclusion:** Rollback
  isolated the value with the current stack.
- **Final disposition:** Production.
- **Lesson:** Synchronization removal helps when it preserves useful overlap and
  valid dependencies.

<a id="orch-10"></a>
### ORCH-10: Second Metal queue

- **Hypothesis:** Shared and routed GPU branches could run independently on two
  queues.
- **Variants tested:** Single and secondary queue schedules.
- **Evidence:** The single-queue control measured 5.592 tok/s; the
  secondary-queue candidate measured 5.059 tok/s.
- **What changed the conclusion:** Added synchronization
  and resource contention offset queue-level parallelism.
- **Final disposition:**
  Rejected.
- **Lesson:** Multiple queues do not create independent hardware.

<a id="orch-11"></a>
### ORCH-11: O3 cross-token chain

- **Hypothesis:** Carrying events across token boundaries would reduce CPU waits.
- **Variants tested:** Default orchestration and the O3 event chain on short and
  long rows.
- **Evidence:** The short median favored O3, 6.524 versus 6.385 tok/s;
  the long gate favored rollback, 6.756 versus 6.737.
- **What changed the conclusion:** Context-dependent mixed results did not justify the added state.
- **Final disposition:** Rejected and removed.
- **Lesson:** Cross-token complexity
  needs a stable gain across the supported context range.

<a id="orch-12"></a>
### ORCH-12: CB1 encode-ahead

- **Hypothesis:** Encoding the next token's cb1 while the current token completed
  would hide CPU work.
- **Variants tested:** Default and encode-ahead schedules.
- **Evidence:** The sustained 1K/256 candidate diverged from the control at
  emitted index 51 and regressed from the control's 4.801 tok/s to 4.613. An
  earlier tiny smoke had moved from 6.032 to 6.098 tok/s.
- **What changed the conclusion:** The sustained row reversed the smoke result.
- **Final disposition:** Rejected.
- **Lesson:** Use a sustained row for pipeline overlap decisions.

<a id="orch-13"></a>
### ORCH-13: Decode argument-buffer reuse

- **Hypothesis:** Recycling MoE bindings would reduce per-layer allocation and
  encode work.
- **Variants tested:** Fresh and reused argument buffers.
- **Evidence:** The reuse path had an unsafe lifetime and divergent output.
  Decode also fell from 4.754 to 3.998 tok/s, and prefill rose 9.6%.
- **What changed the conclusion:** Lifetime and speed both failed.
- **Final disposition:** Rejected
  and removed.
- **Lesson:** Resource reuse is invalid until GPU ownership and
  completion are explicit.

<a id="orch-14"></a>
### ORCH-14: All-hit fast path

- **Hypothesis:** Tokens with all experts cached could skip synchronization.
- **Variants tested:** General and all-hit-specialized paths across 351 states.
- **Evidence:** Total relevant synchronization was only 2.761 ms; decode remained
  neutral, 3.746 versus 3.736 tok/s.
- **What changed the conclusion:** The measured
  target was too small.
- **Final disposition:** Rejected.
- **Lesson:** Size the
  overhead before specializing it.

<a id="orch-15"></a>
### ORCH-15: Merge shared and hit command buffers

- **Hypothesis:** One command buffer would halve submissions without changing
  work.
- **Variants tested:** Separate and merged schedules.
- **Evidence:**
  Submissions fell, but decode regressed from 4.013 to 3.707 tok/s.
- **What changed the conclusion:** Submission reduction neither improved speed nor preserved
  output; the sequence diverged at index 36.
- **Final disposition:** Rejected.
- **Lesson:** Submission count is not a performance result.

[Previous: Prefill](06-prefill.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Sampling, tokenization, and output](08-sampling-tokenization-and-output.md)
