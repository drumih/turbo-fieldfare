# Model installation and expert I/O

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Decode, MoE, INT4, and router](02-decode-moe-int4-and-router.md)

The model's routed-expert pool is much larger than physical memory, so disk
access is part of every decode step. These experiments established the bounded
remote installer and the demand-read path used by the frozen M2 reference
runtime.
The entries retain their stable ID order, so the installer that creates the
runtime artifact appears last.

| Current result | Disposition |
| --- | --- |
| Remote range repack with 512 KiB peak payload and scratch heap | Production |
| Default bounded parallel miss-read path | Production |
| Whole-pool `mmap`, `mlock`, compression, speculative reads, and MTLIO | Rejected |

## Runtime expert I/O

<a id="io-01"></a>
### IO-01: `mmap` versus `pread`

- **Hypothesis:** Faulting mapped expert pages might avoid copies and beat
  explicit reads.
- **Variants tested:** Cold and warm `mmap`, sequential and
  parallel `pread`, and a full-token simulator.
- **Evidence:** A cold expert read
  took 9.88 ms with `mmap` and 2.79 ms with `pread`, about a 3.5x difference.
  Warm `mmap` won its local comparison, but the expert working set exceeds usable
  cache on the target machine. The simulator reached 3.97 tok/s with parallel `pread` and
  0.50 tok/s with `mmap`.
- **What changed the conclusion:** Warm-file results
  ceased to matter once the gate modeled continual cold misses.
- **Final disposition:** Production uses bounded parallel `pread`.
- **Lesson:** Benchmark
  the cache state the application can sustain, not the cache state a microbench
  can manufacture.

<a id="io-02"></a>
### IO-02: Parallel reads and command-buffer coalescing

- **Hypothesis:** Concurrent expert reads and fewer GPU submissions would reduce
  the two largest early stalls.
- **Variants tested:** The serial stage-0 path,
  parallel `pread`, and parallel `pread` plus command-buffer coalescing.
- **Evidence:** Decode moved from about 1.13 to 2.08 tok/s with parallel reads and
  to about 2.13 tok/s after coalescing.
- **What changed the conclusion:** Nothing;
  later stages kept the structure.
- **Final disposition:** Production.
- **Lesson:**
  Fix the largest wall-time bucket before tuning arithmetic inside a smaller one.

<a id="io-03"></a>
### IO-03: Darwin file hints

- **Hypothesis:** Darwin-specific caching and read-ahead controls could improve
  miss service without changing the data path.
- **Variants tested:** `F_RDAHEAD=0`, `F_NOCACHE`, and Darwin
  [`F_RDADVISE`](04-rdadvise.md), which advises the kernel about upcoming file
  ranges.
- **Evidence:** The first two were neutral.
  Untimed ahead advice reached 3.61-3.78 GB/s and justified a real-decode
  candidate.
- **What changed the conclusion:** Because advice ran outside the
  timed body, the probe justified only a separate real-decode program.
- **Final disposition:** The simple hints are rejected; RDADVISE is evaluated separately.
- **Lesson:** A
  promising system-call probe earns an end-to-end test, not immediate promotion.

<a id="io-04"></a>
### IO-04: Dedicated expert-I/O executor

- **Hypothesis:** A controlled executor and split-read schedule would outperform
  the existing concurrent reads.
- **Variants tested:** Worker and submission
  counts around the production path.
- **Evidence:** The best executor result was
  8.59 ms; the existing path measured 8.42 ms. Real decode was unchanged at four
  and eight workers.
- **What changed the conclusion:** The candidate failed its
  first production comparison.
- **Final disposition:** Rejected.
- **Lesson:** A new
  scheduler must beat the existing concurrency, not merely expose more controls.

<a id="io-05"></a>
### IO-05: Custom I/O worker pool

- **Hypothesis:** A fixed custom worker pool could beat bounded parallel miss
  reads.
- **Variants tested:** The default path and a four-worker override in
  repeated 256-token decode.
- **Evidence:** The pairs were mixed: 5.964 to 6.132
  tok/s, then 6.093 to 6.022; profile I/O stayed at 71.3 ms/token.
- **What changed the conclusion:** The repeated pair reversed the first result.
- **Final disposition:** The custom worker override was rejected; production retains
  parallel reads over the current miss set.
- **Lesson:** A concurrency wrapper
  must beat the existing parallel work distribution across repeated rows.

<a id="io-06"></a>
### IO-06: `mlock` and resident-model pages

- **Hypothesis:** Expert streaming evicted the 1.58 GB resident model, so pinning
  it should restore GPU time.
- **Variants tested:** Locked resident buffers and a
  diagnostic run that skipped streaming.
- **Evidence:** `mlock` recovered
  essentially 0 ms. Skipping expert streaming recovered about 12-14 ms.
- **What changed the conclusion:** The mechanism was memory-system contention,
  not resident-page eviction.
- **Final disposition:** Rejected.
- **Lesson:** Prove
  the mechanism before choosing a remedy with a large memory-policy cost.

<a id="io-07"></a>
### IO-07: Expert compression

- **Hypothesis:** Compressing expert blobs could reduce bytes read enough to pay
  for decompression.
- **Variants tested:** Zstandard and LZ4 over representative
  packed experts.
- **Evidence:** Zstandard saved about 10%; LZ4 saved 0.06%.
- **What changed the conclusion:** Neither ratio supported the required CPU work
  and format complexity.
- **Final disposition:** Rejected.
- **Lesson:** Already
  quantized weights may contain too little redundancy for transparent runtime
  compression.

<a id="io-08"></a>
### IO-08: Speculative expert reads

- **Hypothesis:** Reading or advising likely future experts could convert misses
  into warm page-cache hits.
- **Variants tested:** Entry probes followed by a
  paired real decode with speculative reads enabled.
- **Evidence:** Probes showed post-advice residency and up to 10.63 GB/s. In
  the first end-to-end pair, decode fell from 4.937 to 4.742 tok/s, total
  prefill wall time rose from 82.50 to 123.64 s, and emitted IDs diverged. The
  pair therefore failed both its speed and token-parity gates.
- **What changed the conclusion:** The probe measured page residency,
  not lead time, contention, or pipeline behavior.
- **Final disposition:**
  Rejected.
- **Lesson:** A mechanism probe cannot substitute for the first clean
  end-to-end pair. Cache prediction and replacement are covered in the
  [cache summary](03-expert-cache-prediction-and-layout.md).

<a id="io-09"></a>
### IO-09: MTLIO

- **Hypothesis:** A GPU-oriented I/O queue could remove CPU staging for routed
  experts.
- **Variants tested:** Warm MTLIO reads and production miss-state
  classification.
- **Evidence:** Warm reads reached 13.1-13.3 GB/s, but only
  5.4-7.5% of observed misses were fully warm.
- **What changed the conclusion:**
  The fast state was too rare to control token time.
- **Final disposition:**
  Rejected for the runtime.
- **Lesson:** Optimize the state distribution the
  application occupies, not the fastest state an API exposes.

## Model installation

<a id="io-10"></a>
### IO-10: Remote streaming repack

- **Hypothesis:** The installer could convert the source model directly from
  remote byte ranges without storing or loading the complete source checkpoint.
- **Variants tested:** A pinned remote revision, range planning, bounded payload
  copying, interruption guards, and final manifest verification.
- **Evidence:** A
  validated run downloaded 14,952,958,284 bytes in 229 ranges. Its largest
  transfer was 64 MiB; peak payload and scratch heap were each 524,288 bytes.
  The freshly installed output loaded in 0.97 s. A greedy eight-token smoke
  for `The capital of France is` generated at 5.015 tok/s; this validated the
  install/load path, not answer quality.
- **What changed the conclusion:** Nothing; later installs kept the same
  core path.
- **Final disposition:** Production.
- **Lesson:** Model installation
  must obey the same bounded-memory architecture as inference.

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Decode, MoE, INT4, and router](02-decode-moe-int4-and-router.md)
