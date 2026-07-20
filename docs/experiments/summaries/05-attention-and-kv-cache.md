# Attention and KV-cache experiments

[Previous: RDADVISE](04-rdadvise.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Prefill](06-prefill.md)

The current runtime uses exact split-K/V attention and an FP16 KV cache. Packed
K4/V4 saved only about 82 MiB at 4K, grew across all 30 attention layers, and
failed the quality gate. It was rejected and removed.

| Current result | Disposition |
| --- | --- |
| Split-KV plus GQA-aware SWA | Production |
| Exact split full attention | Production |
| FP16 KV ring | Production |
| Packed K4/V4 and alternate codecs | Rejected and removed |

## Decode attention geometry

<a id="kv-01"></a>
### KV-01: Split-KV attention

- **Hypothesis:** Partitioning the sequence and combining partial online-softmax
  states would expose decode parallelism.
- **Variants tested:** Single-pass and
  split-KV attention across SWA, full attention, and long context.
- **Evidence:**
  SWA improved from about 995 to 289-302 microseconds; full attention from 1473
  to 360-382; full 4K from 6103 to 1471, about 4.1x. Long-context cb1 GPU time
  fell about 28%, while short-context end-to-end speed stayed nearly flat.
- **What changed the conclusion:** Whole-step share explained the short-row
  neutrality.
- **Final disposition:** Production.
- **Lesson:** A large kernel win
  matters only where that kernel occupies enough of the step.

<a id="kv-02"></a>
### KV-02: GQA-aware sliding-window attention

- **Hypothesis:** Query heads that share KV heads should share decode work.
- **Variants tested:** Generic and GQA-aware SWA kernels.
- **Evidence:** The generic split-partial control took 295.04 microseconds; the
  GQA-aware candidate took 273.69 microseconds and passed the runtime gate.
  Full attention required a different geometry.
- **What changed the conclusion:**
  Nothing.
- **Final disposition:** Production for SWA.
- **Lesson:** Architectural
  reuse helps only when the kernel maps the sharing relationship directly.

<a id="kv-03"></a>
### KV-03: Full-attention GQA staging

- **Hypothesis:** The SWA GQA strategy would transfer to full attention.
- **Variants tested:** Default full attention and the A3 full-GQA candidate at
  1024 and 4096.
- **Evidence:** In isolated kernel rows, the candidate was 21.2% faster at 1024 and 54.6%
  slower at 4096.
- **What changed the conclusion:** Long-context scaling reversed
  the short-row win.
- **Final disposition:** Rejected.
- **Lesson:** Attention
  geometry needs at least one long-sequence gate.

<a id="kv-04"></a>
### KV-04: Local full-attention variants

- **Hypothesis:** Broadcast softmax, `exp2`, K16 combine, or another threadgroup
  width would trim the full-attention floor.
- **Variants tested:** Each change at
  1024 and 4096, including 64, 128, and 256 threads.
- **Evidence:** In isolated
  rows, broadcast softmax, `exp2`, and K16 improved 1024 by 0.9%, 1.5%, and 3.1%,
  respectively, but were neutral or slower at 4096. TG64, TG128, and TG256
  regressed at both lengths.
- **What changed the conclusion:** The sweep showed a
  geometry problem rather than one expensive instruction.
- **Final disposition:**
  Rejected.
- **Lesson:** Stop local tweaking when all nearby variants preserve the
  same bound.

<a id="kv-05"></a>
### KV-05: MLX geometry v1

- **Hypothesis:** MLX's vectorized full-attention layout would solve the geometry
  problem.
- **Variants tested:** The current split kernel and the first MLX-style
  mapping.
- **Evidence:** Isolated time improved 65-68%; early end-to-end rows
  improved 2.8-3.7%. A forced-prefix exact-output gate then failed and the
  candidate was removed.
- **What changed the conclusion:** A corrected v2 reopened the geometry family,
  but the instruction checkpoint's full quality gate later rejected it too.
- **Final disposition:** Rejected; exact split remains production.
- **Lesson:** Select the quality oracle from the candidate's mathematical
  contract without retroactively declaring an old candidate safe. See
  [METH-01](09-validation-and-measurement-lessons.md#meth-01).

<a id="kv-06"></a>
### KV-06: MLX geometry v2

- **Hypothesis:** A corrected MLX-style mapping could retain the speed signal and
  pass an appropriate quality gate.
- **Variants tested:** Exact reference repair,
  isolated geometry, a 9,216-row quality corpus, and balanced M2 end-to-end
  rows.
- **Evidence:** Isolated speed improved 71-74%, and an earlier decode path
  improved 1.30-1.59%. On the instruction checkpoint's 9,216-row evaluation,
  exact split-K/V matched the FP16 reference while MLX-v2 exceeded the accepted
  mean delta-NLL threshold.
- **What changed the conclusion:** Revalidation against the shipping instruction
  checkpoint replaced the earlier promotion result.
- **Final disposition:** Rejected for the instruction checkpoint; exact split is
  production.
- **Lesson:** Revalidate a strong speed signal when the rejection instrument
  was wrong. See [METH-01](09-validation-and-measurement-lessons.md#meth-01).

## Packed KV and quality

<a id="kv-07"></a>
### KV-07: Packed K4/V4 attention

- **Hypothesis:** Keeping K and V in packed 4-bit form would save memory and
  reduce bandwidth enough to pay for decode.
- **Variants tested:** Single-pass and
  split packed attention, Q pretransform, function constants, K staging, and
  full-GQA forms.
- **Evidence:** Split packed attention beat packed single-pass by
  about 2.4x but remained slower than FP16. Q pretransform and function constants
  helped; K staging and full-GQA regressed.
- **What changed the conclusion:**
  Optimization could not erase packed access and dequantization cost.
- **Final disposition:** Rejected and removed.
- **Lesson:** Improve a
  candidate against the production control, not only against its first version.

<a id="kv-08"></a>
### KV-08: Alternative K/V codecs

- **Hypothesis:** `vllm4nc` or `affine4g32` could improve the packed-cache
  balance.
- **Variants tested:** Current K4/V4 stores one scale per head role;
  `vllm4nc` uses norm metadata for K and affine scale/zero metadata for V;
  `affine4g32` stores scale/zero metadata per group of 32. Each ran in isolated
  writer and packed-attention rows, with FP16 as a separate production control.
- **Evidence:** The
  alternate codecs improved some writer rows but made packed attention much slower.
- **What changed the conclusion:** Writer speed was not the controlling end-to-end cost.
- **Final disposition:** Rejected.
- **Lesson:** Evaluate a storage format across
  write, read, attention, quality, and memory together.

<a id="kv-09"></a>
### KV-09: K4/V4 quality characterization

- **Hypothesis:** The packed cache's memory saving would fit the quality budget.
- **Variants tested:** Packed K4/V4 against FP16 over the full quality corpus.
- **Evidence:** Mean delta-NLL was +0.015197, p95 +0.287202, top-1 agreement
  dropped 5.0781 percentage points, and top-8 dropped 5.5990 points. Every split
  failed. Against the final FP16 ring, packed storage saves only about 82 MiB.
- **What changed the conclusion:** The full corpus and ring-relative memory
  comparison replaced earlier short and obsolete-linear comparisons.
- **Final disposition:** Rejected and removed.
- **Lesson:** Approximation must clear quality against the best exact memory
  layout.

<a id="kv-10"></a>
### KV-10: FP16 full-attention island

- **Hypothesis:** Keeping five full-attention layers exact while packing SWA
  layers could recover most quality.
- **Variants tested:** All-packed and an FP16
  full-attention island.
- **Evidence:** A 16-row sample looked better. At 256 rows,
  mean delta-NLL improved relative to all-packed, but top-1 still fell 1.5625
  points and top-8 fell 0.5371. Advancement required mean delta-NLL to improve
  by at least 0.002 nat/token and both agreement metrics to improve by at least
  1.0 percentage point.
- **What changed the conclusion:** The larger gate
  reversed the small sample.
- **Final disposition:** Rejected and removed.
- **Lesson:** Small quality samples are useful for screening, not promotion.

<a id="kv-11"></a>
### KV-11: Packed-attention chunk 32

- **Hypothesis:** More final SWA chunks would improve packed-attention occupancy.
- **Variants tested:** 16 and 32 final chunks with isolated, short-decode,
  and independent quality rows.
- **Evidence:** Isolated speed improved 5.42-6.30%;
  an earlier short decode rose from 6.511 to 7.110 tok/s. Holdout delta-NLL was
  +0.005853 and top-1 agreement fell 0.846 points, both outside the gate.
- **What changed the conclusion:** The independent quality holdout rejected the
  speed winner.
- **Final disposition:** Rejected and removed.
- **Lesson:** Keep the
  quality holdout independent from candidate selection.

## Exact FP16 ring

<a id="kv-12"></a>
### KV-12: FP16 KV ring

- **Hypothesis:** Sliding-window layers need only a ring, not the retired linear
  4K allocation.
- **Variants tested:** Linear and ring FP16 storage with split-KV
  attention.
- **Evidence:** Against the retired linear FP16 allocation, the ring saved about
  575-591 MiB. Near-4K decode was neutral to slightly faster, 4.357 to 4.417
  tok/s; 1K was slightly slower.
  Token parity held after a PSO-selection asymmetry was fixed.
- **What changed the conclusion:** Correctness repair established an exact storage optimization.
- **Final disposition:** Production.
- **Lesson:** Reclaim exact lifetime waste
  before spending quality on compression.

<a id="kv-13"></a>
### KV-13: Ring-specific kernel follow-up

- **Hypothesis:** The promoted ring kernel still had meaningful local headroom.
- **Variants tested:** The retired linear control and current ring-1152 kernel,
  used to decide whether a new modulo-hoist or segment-split candidate was
  justified.
- **Evidence:** The initial ring row improved 4.5% over the linear control,
  implying about 0.14% whole-step opportunity. A terminal alternating median
  favored ring by 10.9%, but weighted opportunity remained about 0.34%, below
  the 0.5% action gate.
- **What changed the conclusion:** Earlier work had reduced the target's share.
- **Final disposition:**
  Stopped without promotion.
- **Lesson:** Recompute whole-step value after every
  major stack change.

## Prefill attention correctness

<a id="kv-14"></a>
### KV-14: Prefill tiled-attention race

- **Hypothesis:** Production-shaped validation could expose synchronization
  defects hidden by toy tests.
- **Variants tested:** The original shared scratch
  plus a corrected single-bank path with a third barrier, then a two-bank layout.
- **Evidence:** The unsafe bank lacked a reader-to-next-writer edge. A third
  barrier made it correct but slowed production shapes 5.1-14.2%. Two alternating
  banks retained byte-identical output and recovered 2.23-6.43% versus corrected
  single-bank.
- **What changed the conclusion:** The layout removed the cost of an
  otherwise necessary barrier.
- **Final disposition:** Two-bank correctness repair
  in production.
- **Lesson:** Every reused threadgroup-memory cycle needs an explicit
  reader-to-next-writer edge. The surrounding compute path is covered in the
  [prefill summary](06-prefill.md).

[Previous: RDADVISE](04-rdadvise.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Prefill](06-prefill.md)
