# Sampling, tokenization, and output experiments

[Previous: Fusions, head, and orchestration](07-fusions-head-and-orchestration.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Validation and measurement lessons](09-validation-and-measurement-lessons.md)

The main sampling win came from replacing a repeated-vocabulary algorithm with
one pass. Tokenizer caching produced a second large support-path win. Further
fusion and allocation work had mixed end-to-end value.

| Current result | Disposition |
| --- | --- |
| One-pass sampling with Top-P/Top-K | Production |
| Fused Gumbel head | Conditional; default off |
| Tokenizer process cache | Production; support path |
| Bounded detokenizer tail | Production; allocation hygiene |

## Sampling and head

<a id="out-01"></a>
### OUT-01: One-pass Gumbel sampling

- **Hypothesis:** One Gumbel-max pass over the vocabulary would replace repeated
  top-k extraction.
- **Variants tested:** The old repeated extraction, greedy
  selection, and one-pass sampled selection.
- **Evidence:** The old path took
  about 2.651 s/token and produced 0.377 tok/s. One-pass sampling restored about
  5.86-5.89 tok/s, near greedy throughput.
- **What changed the conclusion:** The product path later added
  full-distribution Top-P, Top-K, then temperature. The default Top-K `64`
  case uses a specialized 1,024-to-64 reduction measured at 0.733 ms/sample.
- **Final disposition:** Production sampling path, including the specialized
  Top-64 truncation route.
- **Lesson:** Removing
  an avoidable algorithmic factor can dwarf kernel tuning.

<a id="out-02"></a>
### OUT-02: Fused Gumbel head

- **Hypothesis:** Sampling inside the LM-head reduction would avoid standalone
  logits and another pass.
- **Variants tested:** Fused sampling and standalone logits
  plus one-pass sampling, both using plain seeded temperature sampling.
- **Evidence:** The isolated candidate looked competitive, but end-to-end sampled
  decode produced 5.124 tok/s versus 5.192 tok/s for the standalone control.
- **What changed the conclusion:** Removing an
  intermediate did not shorten the whole pipeline.
- **Final disposition:** Conditional; default off. This is the sampling half of
  the wider fusion result in [ORCH-05](07-fusions-head-and-orchestration.md#orch-05).
- **Lesson:** Fusion needs a whole-step win even
  when it removes a visible buffer.

## Tokenization and output

<a id="out-03"></a>
### OUT-03: Tokenizer process cache

- **Hypothesis:** Reusing tokenizer state would avoid repeated model and table
  loading across tests and requests.
- **Variants tested:** Per-use load and a
  process cache.
- **Evidence:** The tokenizer suite body fell from 29.002 to 4.425
  s; command wall fell from 36.54 to 6.67 s, and peak RSS fell from 3.86 GB to
  397 MB.
- **What changed the conclusion:**
  Nothing.
- **Final disposition:** Production; support-path cache.
- **Lesson:** Bounded
  reuse can improve both time and peak memory when initialization dominates.

<a id="out-04"></a>
### OUT-04: Bounded detokenizer tail

- **Hypothesis:** Avoiding growing trailing slices would remove quadratic
  allocation behavior.
- **Variants tested:** Original and bounded-tail
  detokenization over ASCII, byte-heavy, and mixed Unicode text.
- **Evidence:**
  ASCII and byte-heavy cases were neutral; mixed Unicode improved 17.51%, while
  the change removed explicit growing stable/trailing slices.
- **What changed the conclusion:** The cleanup bounded avoidable allocation without claiming linear
  scaling or an end-to-end decode win.
- **Final disposition:** Production; bounded-allocation hygiene.
- **Lesson:** State the resource goal separately
  from local microbenchmark movement.

[Previous: Fusions, head, and orchestration](07-fusions-head-and-orchestration.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Validation and measurement lessons](09-validation-and-measurement-lessons.md)
