# Expert cache, prediction, and layout experiments

[Previous: Decode, MoE, INT4, and router](02-decode-moe-int4-and-router.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: RDADVISE](04-rdadvise.md)

The production runtime keeps 16 expert slots per layer and uses LFU
replacement. More memory improves hit rate, but trace-trained allocation,
prediction, and disk-layout policies failed representative holdouts. The
[resource split](../../SYSTEM_DESIGN.md#resource-split) explains the slot
capacity and memory cost.

| Current result | Disposition |
| --- | --- |
| Uniform 16-slot LFU | Production |
| 24/32 slots | Conditional +memory experiments |
| Predictor, Markov, read sorting, and single-trace layout | Rejected |
| Fresh preallocated expert artifact | Unexecuted hypothesis |

## Cache policy and capacity

<a id="cache-01"></a>
### CACHE-01: LRU to LFU

- **Hypothesis:** Frequency would retain recurring experts better than recency
  under a fixed 16-slot budget.
- **Variants tested:** LRU and LFU on canonical and
  long decode rows.
- **Evidence:** Canonical decode rose from 5.476 to 5.631 tok/s
  and I/O fell from 72.6 to 64.8 ms. Long-run I/O fell from 110.5 to 97.9 ms.
- **What changed the conclusion:** Nothing.
- **Final disposition:** Production.
- **Lesson:** Replacement policy captured more reuse without spending memory.

<a id="cache-02"></a>
### CACHE-02: Windowed LFU

- **Hypothesis:** Aging frequency counts would adapt faster when routing changed.
- **Variants tested:** Monotonic LFU and a 64-token window.
- **Evidence:** The
  window reduced misses slightly, but two real-decode prompts were neutral or
  mixed. Belady's offline-optimal policy, which knows all future requests,
  remained about 8.4 percentage points above LFU on the France prompt and 10.8
  percentage points above it on the corrected two-prompt corpus.
- **What changed the conclusion:** Better miss counts did not reduce the measured
  step.
- **Final disposition:** Rejected.
- **Lesson:** Cache-policy simulation needs
  an end-to-end I/O gate.

<a id="cache-03"></a>
### CACHE-03: 24 and 32 expert slots

- **Hypothesis:** More resident experts would buy enough I/O reduction to justify
  the memory.
- **Variants tested:** 16, 24, and 32 slots per layer.
- **Evidence:**
  Relative to the 16-slot control, trace replay gave 24 slots another 12.6-14.9
  hit-rate points for 769 MiB and 32 slots another 20.9-23.7 points for 1.54
  GiB. A live 32-slot row fell to 1.578 tok/s as I/O reached 318.3 ms/token and
  pipeline wait reached 238.3 ms/token on the 8 GB Mac.
- **What changed the conclusion:** The hit-rate win remained real,
  but it violated the memory objective for the default.
- **Final disposition:**
  Conditional +memory experiment; production stays at 16.
- **Lesson:** Report
  memory-funded speed separately from memory-free optimization.

<a id="cache-04"></a>
### CACHE-04: Fixed-total heterogeneous allocation

- **Hypothesis:** Layers should receive different slot counts under the same
  480-slot total.
- **Variants tested:** Uniform `16 x 30` allocation and an exact dynamic
  program that distributed the same 480 total slots across layers to minimize
  misses on related training traces.
- **Evidence:** The optimized allocation lost on the
  held-out near-4K trace.
- **What changed the conclusion:** Holdout routing did not
  match the training traces.
- **Final disposition:** Rejected.
- **Lesson:** A
  trace-trained cache layout needs representative holdouts before runtime work.

## Prediction and disk layout

<a id="cache-05"></a>
### CACHE-05: Cross-layer expert predictor

- **Hypothesis:** One layer's expert choices could predict the next layer's
  misses.
- **Variants tested:** Cross-layer set overlap and direct copied-expert
  prediction.
- **Evidence:** Mean Jaccard was 0.039; copied predictions hit only
  7%.
- **What changed the conclusion:** The offline signal failed the threshold
  for a runtime candidate.
- **Final disposition:** Rejected before implementation.
- **Lesson:** Measure predictability before adding speculative I/O.

<a id="cache-06"></a>
### CACHE-06: Markov warming and prefetch

- **Hypothesis:** Recent expert transitions could predict future demand.
- **Variants tested:** Early Markov warming and later demand-adjacent prefetch.
- **Evidence:** The first version was flat or negative. Later rows collapsed from
  values such as 5.413 to 2.256 tok/s and changed output.
- **What changed the conclusion:** Background reads contended with foreground NVMe and memory
  traffic; the later implementation also reused the foreground slot pool and was
  not correctness-safe.
- **Final disposition:** Rejected.
- **Lesson:** Prefetch accuracy alone
  cannot justify traffic that delays the demand path.

<a id="cache-07"></a>
### CACHE-07: File-offset read ordering

- **Hypothesis:** Sorting misses by file offset would improve storage locality.
- **Variants tested:** Route order and physical-offset order.
- **Evidence:** The
  first row improved from 6.012 to 6.117 tok/s. Repeats reversed the result:
  6.099 to 6.067 tok/s, with I/O rising from 75.0 to 76.7 ms.
- **What changed the conclusion:** The first win did not repeat.
- **Final disposition:**
  Rejected.
- **Lesson:** Interleave repeated I/O-policy comparisons.

<a id="cache-08"></a>
### CACHE-08: Packed expert layout

- **Hypothesis:** Reordering experts from a natural-text trace and combining
  adjacent reads would reduce calls and seek distance.
- **Variants tested:** The
  identity layout and one trace-specialized layout with grouped reads.
- **Evidence:** Natural-text replay improved 3.61% with about 49.7 MB extra RSS,
  using the unit recorded by the source artifact.
  The near-4K gate reversed it: decode fell from 4.111 to 3.449 tok/s and prefill
  rose from 168.85 to 312.48 seconds total.
- **What changed the conclusion:** The
  training-like trace did not generalize.
- **Final disposition:** Rejected.
- **Lesson:** On-disk order is a workload policy and needs multi-workload gates.

<a id="cache-09"></a>
### CACHE-09: APFS extent preallocation

- **Hypothesis:** A freshly preallocated expert artifact might reduce file
  fragmentation.
- **Variants tested:** Existing-file extent inspection only.
- **Evidence:** The files were fragmented, but no clean preallocated artifact
  received an end-to-end gate.
- **What changed the conclusion:** No runtime
  conclusion was reached.
- **Final disposition:** Unexecuted hypothesis.
- **Lesson:**
  A plausible filesystem mechanism is neither a win nor a rejection until the
  artifact is built and measured.

[Previous: Decode, MoE, INT4, and router](02-decode-moe-int4-and-router.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: RDADVISE](04-rdadvise.md)
