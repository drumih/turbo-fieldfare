# RDADVISE experiments

[Previous: Expert cache, prediction, and layout](03-expert-cache-prediction-and-layout.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Attention and KV cache](05-attention-and-kv-cache.md)

Darwin's `F_RDADVISE` tells the kernel which file ranges an application expects
to read. It produced a repeatable short-run win and was briefly a default
candidate. Longer and differently scheduled runs exposed severe stalls and no
single policy that stayed safe across contexts. Production keeps it off.

| Current result | Disposition |
| --- | --- |
| Static miss-only advice | Rejected as a default |
| Capped policy | Rejected |
| Adaptive and asynchronous policies | Rejected as defaults; opt-in retained |
| Production profile | RDADVISE off |

## Discovery and repeat

<a id="rad-01"></a>
### RAD-01: Initial miss-only real-decode signal

- **Hypothesis:** Advising the kernel about upcoming miss ranges would start
  page-cache work before demand reads.
- **Variants tested:** Real decode with
  miss-only advice off and on.
- **Evidence:** The first row showed a plausible I/O
  and token-rate improvement.
- **What changed the conclusion:** One row could not
  establish a default.
- **Final disposition:** Promising conditional result.
- **Lesson:** System advice needs repeats and context sweeps.

<a id="rad-02"></a>
### RAD-02: Repeated short-run default gate

- **Hypothesis:** The initial result would survive paired same-session repeats.
- **Variants tested:** Default miss-only advice against no advice.
- **Evidence:**
  Median decode rose from 5.176 to 5.449 tok/s, about 5.3%; I/O fell from 87.4 to
  72.2 ms.
- **What changed the conclusion:** Later long-context tests found unsafe
  stalls.
- **Final disposition:** Temporarily promoted, later removed.
- **Lesson:**
  The early win was real for its state, but its state was too narrow.

## Policy variants

<a id="rad-03"></a>
### RAD-03: Advice-call cap

- **Hypothesis:** Limiting advice calls would preserve prefetch value while
  reducing overhead.
- **Variants tested:** Unbounded and capped miss-only advice.
- **Evidence:** The cap reduced advisory work but increased demand-read I/O.
- **What changed the conclusion:** Call count was not the controlling metric.
- **Final disposition:** Rejected.
- **Lesson:** Fewer system calls can still
  produce worse page-cache timing.

<a id="rad-04"></a>
### RAD-04: Advice inside the I/O path

- **Hypothesis:** Moving advice into the planned fetch path might remove serial
  decode-thread advice time without losing its I/O benefit.
- **Variants tested:**
  Plan-time advice and I/O-path advice.
- **Evidence:** Decode fell from 6.432 to 6.093 tok/s; I/O rose from 71.3 to
  84.8 ms.
- **What changed the conclusion:** Placement increased interference
  with the demand path.
- **Final disposition:** Rejected.
- **Lesson:** Advice timing
  is part of the I/O schedule, not a free hint.

<a id="rad-05"></a>
### RAD-05: Long-context policy gate

- **Hypothesis:** The short-run default would remain safe as context and miss
  state changed.
- **Variants tested:** Unbounded advice, bounded advice, and no
  advice across context lengths.
- **Evidence:** At 1,024 context tokens, unbounded advice
  fell to 2.334 tok/s and spent 86.39 ms in advice calls; bounded advice reached
  4.966 tok/s. The bounded policy then regressed the 512-token context.
- **What changed the conclusion:** No static cap covered both states.
- **Final disposition:** The bounded static cap was rejected for default promotion at
  this gate; later runtime policy moved production to RDADVISE off.
- **Lesson:** A policy must
  survive the context transition where its mechanism changes.

<a id="rad-06"></a>
### RAD-06: Adaptive policy

- **Hypothesis:** Miss count, byte count, slow-call thresholds, and cooldown could
  select advice only in favorable states.
- **Variants tested:** Several adaptive threshold combinations across repeated
  1,536-token gates, followed by a fresh policy that capped the total advised
  ranges at 512 MiB.
- **Evidence:** One 1,536-token row improved from
  4.634 to 4.756 tok/s, but the 512 MiB candidate later fell from 4.823 to 3.397.
- **What changed the conclusion:** State-specific wins failed to generalize.
- **Final disposition:**
  Rejected for default promotion; adaptive remains opt-in.
- **Lesson:** Adaptation adds value only when the observed signals
  predict the safe state reliably.

## Final conditional result

<a id="rad-07"></a>
### RAD-07: Asynchronous advice overlap

- **Hypothesis:** Moving advice off the demand critical path could keep its page
  warming while hiding call latency.
- **Variants tested:** Synchronous RDADVISE
  and asynchronous-overlap RDADVISE on a 1K/256 decode.
- **Evidence:** The candidate improved
  from 4.863 to 5.300 tok/s. Its first rejection also relied on an exact-ID oracle
  later judged inappropriate near known ties.
- **What changed the conclusion:** The later
  [quality-gate audit](09-validation-and-measurement-lessons.md#meth-01)
  corrected the exact-ID reasoning. The project still found no stable
  selection rule, then froze new M2 performance work without reopening this
  candidate.
- **Final disposition:** Conditional positive result; production remains off.
- **Lesson:** The accurate conclusion is “no stable production policy,” not
  “every RDADVISE variant was slow.”

[Previous: Expert cache, prediction, and layout](03-expert-cache-prediction-and-layout.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Attention and KV cache](05-attention-and-kv-cache.md)
