# Optimizing a 14.4 GB model for an 8 GB machine

TurboFieldfare runs Gemma 4 26B-A4B with about 14.4 GB of packed text-only
weight payload on an 8 GB Apple Silicon machine. The completed installation is
about 14.5 GB after its manifest, expert layout, and tokenizer files. The model
cannot remain resident in memory, so the runtime keeps roughly 1.58 GB of
common weights resident and streams the roughly 11.6 GB routed-expert payload
from NVMe. This constraint shaped the useful optimizations: reduce demand I/O,
expose enough GPU parallelism, and never buy speed by loading the model into
RAM.

This article covers the experiments that materially changed the runtime. The
[experiment inventory](experiments/EXPERIMENT_INVENTORY.md) retains the complete record,
including narrow wins and negative results. Measurements below come from their
named controls and should not be combined into one historical speed curve.

## What worked and what did not

The successful experiments attacked costs visible in the complete token path.
Parallel bounded `pread` made expert misses explicit and schedulable;
persistent multi-threadgroup MoE exposed enough GPU work; a 16-slot LFU cache
and coarse compute/I/O overlap reused resident experts; layout-aware INT4
vectorization shortened hot kernels; split-KV attention and staged affine MPP
changed the shape of expensive work; and one-pass Gumbel sampling removed
repeated vocabulary scans. Each change survived production-shaped and
end-to-end gates.

The unsuccessful experiments often improved a proxy while making the runtime
worse. Demand-paged `mmap` won only with warm pages. Cooperative MoE reduced
the number of row streams but starved the GPU. Progressive expert overlap
added synchronization and regressed decode. Simple Darwin read hints were
neutral; RDADVISE reversed at longer contexts; and `F_SPECULATIVE_READ`, read
reordering, grouped `preadv`, and MTLIO won narrow probes but not broader
runtime gates. Packed K4/V4 saved 82 MiB against the final FP16 ring but failed
the quality gate. A monolithic fusion removed dispatches yet lost throughput.

The sections below pair these wins with the nearby failures that clarified
them. Changes that claim identical math must preserve exact output; changes
that reorder floating-point operations must pass deterministic,
reference-relative quality tests. A promising microbenchmark starts an
experiment, but only end-to-end speed and quality decide whether it ships.

## Explicit reads transformed expert streaming

The first streaming design memory-mapped the expert pool and relied on demand
faults. It looked attractive because the source code contained no explicit
copy, but it gave the virtual-memory system control over read timing and
concurrency. Warm mapped pages won a narrow benchmark; the cold working set on
the target machine did not stay warm.

A cold expert read took 9.88 ms through demand paging and 2.79 ms through
`pread`. In the full streaming simulation, `mmap` delivered about 0.50 tok/s,
while parallel `pread` reached 3.97 tok/s. The production runtime maps the
common model file for Metal access, but services routed-expert misses from
per-layer files through explicit reads into bounded slots. Parallel reads later
moved staged decode from about 1.13 to 2.08 tok/s. See [`mmap` versus
`pread`](experiments/summaries/01-model-install-and-expert-io.md#io-01).

The installer follows the same design. Remote repacking downloaded
14,952,958,284 source bytes in 229 range requests, with a largest transfer of
64 MiB and only 524,288 bytes of payload and scratch heap. Installation is not
an offline exception to the memory architecture.

## MoE needed parallel work, reuse, and coarse overlap

After explicit I/O, routed MoE dominated GPU time. A SIMD-cooperative kernel
assigned several lanes to one expert, reducing hundreds of independent row
streams to eight expert streams. The second command-buffer phase—the routed
computation after expert I/O, called `cb2` in the profiler—grew from about 230
to 527 ms.

Persistent multi-threadgroup workers made the opposite choice. Threadgroups
claimed independent rows until the dispatch completed, cutting `cb2` from
about 239 to 60 ms, a 75% reduction. Paired decode improved from 2.188 to 3.313
tok/s, or 51%. This was the largest end-to-end gain from redesigning a core
forward-pass Metal kernel. It came from exposing more schedulable work, not
from making each worker's arithmetic more elaborate. [Persistent
MoE](experiments/summaries/02-decode-moe-int4-and-router.md#dec-03) became the
production structure.

A bounded 16-slot cache per layer then reduced repeated expert I/O from about
166 to 88 ms and raised decode from 3.313 to 4.261 tok/s. Replacing LRU with
LFU later reduced canonical I/O from 72.6 to 64.8 ms/token and raised decode
from 5.476 to 5.631 tok/s; long-run I/O fell from 110.5 to 97.9 ms/token. A
64-token window reduced simulated misses slightly but produced neutral or
mixed real-decode results, so monotonic LFU remained the default. [The cache
policy record](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-01)
keeps the full comparison.

Cacheability did not imply predictability. Expert choices repeated enough
within the same layer across tokens for LFU to help, but adjacent layers within
one token shared almost no experts. Across 448 decode tokens, the mean
adjacent-layer Jaccard was 0.039, the median was zero, and copying layer L's
experts predicted only 7% of layer L+1. That decisive [offline
no-go](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-05)
stopped speculative cross-layer prefetch before runtime wiring. Reuse paid
through caching, but the tested routing signal could not identify the required
weights early enough to prefetch them.

The shared dense MLP provided a safe overlap boundary: it uses resident
weights, so the GPU can execute it while the CPU fills routed-expert misses.
That change moved a later control from 4.404 to 4.736 tok/s.

Finer-grained overlap failed. Progressive execution launched each expert group
as soon as its reads completed, but the repaired implementation regressed the
1K/256 gate from 4.799 to 4.648 tok/s and diverged. The coarser hit-first split
was faster and easier to synchronize. Overlap helped only when ownership and
dependencies remained explicit.

## Vectorization helped when it respected the storage layout

The model stores affine INT4 weights in groups of 64, decoded as
`q * scale + bias` with BF16 metadata. The successful routed-MoE kernel handled
four groups together, used `half4` activation loads, and exposed independent
arithmetic. Routed GPU time fell from 36.5 to 31.3 ms. The gain came mainly
from shorter dependency chains and instruction-level parallelism, not from
the widest possible integer load.

Applying 32-bit packed loads to resident GEMV initially passed an offset-zero
fixture and then produced garbage in real decode. Resident sub-tensors were
only guaranteed 2-byte alignment because BF16 metadata preceded the weights.
Two `ushort` loads were safe and still reduced LM-head GPU time from about
21.5 to 16 ms. Vector width is therefore a property of the live byte offset,
not just the element format. [The corrected
path](experiments/summaries/02-decode-moe-int4-and-router.md#dec-07) made
production-shaped offsets part of later kernel tests.

## Attention and prefill rewarded structural changes

Split-KV attention divided the cached sequence across threadgroups and merged
online-softmax partials in a second pass. It improved isolated sliding-window
attention by about 3.3x and full 4K attention by about 4.1x. Short-context
decode stayed nearly flat, whereas long-context GPU time in that phase fell
about 28% as attention's share of the step grew.

A later MLX-style geometry reduced isolated full-attention time by roughly
71–74%, yet complete decode improved only 1.30–1.59%. The measurements did not
conflict. Only five of thirty layers use full attention, and attention occupied
a small fraction of these token steps. The microbenchmark proved that the
geometry worked; the full pass measured how often that faster work mattered.

Chunked prefill replaced scalar prompt replay. Increasing the bounded chunk
from 32 to 128 reduced a 1,017-token prefill from 92.89 to 52.35 seconds.
Staged affine MPP then dequantized a small FP16 tile, ran the hardware matrix
primitive, and discarded the tile. It improved the 512-token prefill gate by
about 11% without changing the source format or violating the memory budget.
[Direct shader-local UInt4](experiments/summaries/06-prefill.md#pf-13) was
20.40% slower than this staged path at M128.

Prefill showed the same effect. Batched routed MoE cut its isolated kernel time
by about 31%, while balanced prefill improved about 2%. Reusing activation
tiles in the remaining quantized matrix kernels improved individual families
by 3–10%, but their combined whole-prefill opportunity was only about 0.4%
after earlier promotions. The tile-reuse optimization worked locally; its
target was no longer large enough to justify promotion.

## Sampling removed an algorithmic factor

Sampling exposed an algorithmic mistake rather than a kernel bottleneck. The
old non-greedy path repeatedly extracted candidates across the entire
vocabulary. One Gumbel value per entry followed by a single maximum reduction
restored sampled decode to roughly greedy speed. Because the baseline was very
naive, its speedup ratio would exaggerate the value of the optimization. The
useful result was removing an unnecessary algorithmic factor, not accelerating
the old loop.

## Promising results that did not survive validation

`F_RDADVISE` produced credible short-run wins. One paired median reduced I/O
from 87.4 to 72.2 ms/token and raised throughput from 5.176 to 5.449 tok/s.
Longer and differently cached runs reversed the result: at a 1,536-token
context, advice produced 4.028 tok/s against 5.687 with advice disabled.
Coalescing adjacent advice ranges was a small success: 10,385 requested ranges
became 10,144 calls in one 128-token run. Routed misses were rarely adjacent,
however, and call limits, cooldowns, adaptive policies, and asynchronous advice
never found a stable rule across workloads. Production keeps RDADVISE off
because the runtime cannot reliably identify the favorable state. [The
RDADVISE record](experiments/summaries/04-rdadvise.md) preserves both the
initial win and the later rejection.

Other read-side candidates failed for different reasons. `F_RDAHEAD=0` and
`F_NOCACHE` were neutral. `F_SPECULATIVE_READ` showed page warming and up to
10.63 GB/s in a probe, then reduced decode from 4.937 to 4.742 tok/s and raised
prefill from 82.50 to 123.64 seconds in the first end-to-end pair. Sorting
misses by file offset improved one row from 6.012 to 6.117 tok/s, then reduced
the repeated median from 6.099 to 6.067 tok/s while I/O rose from 75.0 to
76.7 ms/token. MTLIO reached 13.1-13.3 GB/s on warm reads, but only 5.4-7.5% of
observed misses were fully warm. Dedicated executor and worker-pool variants
also failed to beat bounded parallel `pread`, which remained the production
path. [The expert I/O record](experiments/summaries/01-model-install-and-expert-io.md)
keeps the individual controls and outcomes.

Packed K4/V4 KV storage also looked stronger against an obsolete control. It
used about 223 MiB at 4K versus roughly 880 MiB for the old linear FP16 cache.
An exact FP16 ring then exploited the model's 25 sliding-window layers and
required only about 305 MiB, reducing K4/V4's incremental saving to 82 MiB.
The packed path subsequently lost 5.0781 top-1 percentage points and worsened
mean negative log-likelihood. It remains an explicit experiment; exact FP16 is
the default.

Fusion produced three different outcomes. Targeted QKV, layer-tail, and
row-based head fusions reached production after parity checks and whole-step
gates. A monolithic post-attention/pre-FFN wrapper was a clear
regression: it removed launches and intermediates but achieved only 1.811
tok/s versus 2.756 for the control because adjacent operations had
incompatible geometry and data lifetimes.

Other candidates proved a local gain without proving a system gain. Tiling the
language-model head cut its GPU time from 14.2 to 13.1 ms/token. That saved
only 1.1 ms inside a 167.7 ms token step, a theoretical reduction of about
0.66%, while pipeline wait increased by 1.4 ms. The resulting 5.962 versus
5.926 tok/s comparison was therefore inconclusive, not evidence of a
regression. Fused Gumbel sampling similarly averaged 5.124 versus 5.192 tok/s,
but variation between repeats and divergent sampled paths exceeded the mean
gap. These candidates remained off by default because they did not demonstrate
a repeatable whole-step gain, not because they were proven slower. [The fusion
record](experiments/summaries/07-fusions-head-and-orchestration.md) retains the
individual gates.

Other local proxies failed in similar ways. A trace-trained expert layout with
grouped adjacent `preadv` improved natural-text replay by 3.61% but regressed
near-4K decode by 16.1%. An argument-buffer ring cut 21,217 allocations to two
yet slowed long prefill by about 9%. Fewer reads, allocations, or dispatches
describe a mechanism; only end-to-end time and quality determine whether that
mechanism belongs in production.

## The durable method

The successful experiments converged on a repeatable process: profile the
whole step, isolate the largest measured share, reproduce production shapes
and resource contracts, then return to a clean end-to-end comparison. Exact
changes require identity; reordered floating-point kernels require a quality
oracle. Controls and candidates must be interleaved when page cache, warm-up,
or thermals can bias the result.

Promotion is asymmetric. A candidate needs a repeatable gain to become the
default; calling it slower requires a repeatable loss. A narrow difference
that cannot be separated from run-to-run variation is inconclusive, so the
default stays unchanged without turning the result into a rejection.

The main gains came from explicit `pread`, persistent MoE parallelism, bounded
expert reuse, safe compute/I/O overlap, layout-aware INT4 vectorization,
split-KV attention, chunked prefill, and one-pass sampling. Their common trait
was not novelty. Each removed a measured system cost while preserving the
8 GB architecture.

For the resulting runtime, see [System design](SYSTEM_DESIGN.md). The dated
result and reproduction command live in [Benchmarks](BENCHMARKS.md).
