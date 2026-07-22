# Optimizing a 14.3 GB model for an 8 GB machine

TurboFieldfare runs Gemma 4 26B-A4B on an 8 GB Apple Silicon machine. Its
text-only installation, without the vision tower, is about 14.3 GB. The model
was never going to fit politely in memory.

Instead, the runtime keeps 1.35 GB of common model weights available to Metal
and streams the 12.9 GB routed-expert pool from NVMe. Every useful optimization
had to work within that design. We could reduce I/O, schedule it better, or
shorten the work around it. Loading more of the model into RAM was not an
option.

This article covers the experiments that shaped the runtime. The
[experiment inventory](experiments/EXPERIMENT_INVENTORY.md) has the full
record, including narrow wins and negative results. The measurements below
compare each experiment with its own baseline. The experiments use different
checkpoints, hosts, and runtime states, so they should not be combined into one
historical speed curve.

## What worked and what did not

The wins came from four recurring moves: controlled reads, more GPU work,
reuse, and shorter hot paths. Every change we kept later passed its runtime and
correctness checks.

Many unsuccessful experiments were convincing right up until the full runtime
got involved. Warm pages made demand-paged `mmap` look fast. A cooperative
expert kernel looked tidy. Progressive overlap started work earlier. Read
hints, grouped reads, MTLIO, packed KV caches, and larger fusions all improved
something local.

The full runtime told a different story. Some candidates starved the GPU or
added synchronization. Others helped only when the cache was warm.
The TurboQuant packed K4/V4 KV cache failed the full quality evaluation. At
longer contexts, it also used more memory than Gemma 4's exact FP16 layout. The
FP16 runtime uses a fixed circular cache for the 25 sliding-window layers and
grows only for the five full-attention layers.

The sections below show the useful changes alongside the failures. Changes
that claim identical math must preserve exact output. Changes that reorder
floating-point operations must pass deterministic tests against reference
outputs.

A promising microbenchmark starts an experiment. End-to-end speed and quality
decide whether it ships.

## Explicit reads made expert streaming work

The first streaming design used `mmap` for the expert pool. Expert data was
read when the runtime first touched each page. The code performed no explicit
copy, which made the approach look simple and efficient.

That simplicity gave the virtual-memory system control over read timing and
concurrency. Warm mapped pages won a narrow benchmark, but the target machine
could not keep the cold expert working set warm.

A cold expert read took 9.88 ms through demand paging and 2.79 ms through
`pread`. The difference grew in the full streaming simulation: `mmap`
delivered about 0.50 tok/s, while parallel `pread` reached 3.97 tok/s.

That became the production design. The runtime still maps the
common model file for Metal access. Routed-expert misses use explicit reads
into preallocated per-layer cache slots. Parallel reads later moved staged
decode in a historical benchmark from about 1.13 to 2.08 tok/s.
See [`mmap` versus
`pread`](experiments/summaries/01-model-install-and-expert-io.md#io-01).

The installer follows the same memory rule. During validation of the
current instruction-tuned checkpoint, remote repacking transferred
14,620,479,420 source bytes in 223 range requests. Payload and scratch heap
were each capped at 524,288 bytes. The completed text-only installation
occupied 14,291,921,884 bytes.

Installation follows the same memory rule as inference.

## MoE needed parallel work, reuse, and coarse overlap

Once reads were explicit, routed MoE was the largest GPU cost.

The results below come from separate paired experiments at different stages of
the runtime. They explain why we kept each change, but they do not form one
continuous performance curve.

The first redesign used a SIMD-cooperative kernel in which several lanes worked
on one expert. It looked tidier. Unfortunately, it also took away parallel work
the GPU needed. The routed-computation phase after expert I/O more than
doubled, from about 230 to 527 ms.

Persistent GPU workgroups took the opposite approach. Independent workgroups
claimed rows until the dispatch completed, giving the GPU much more work to
schedule. The same phase fell from about 239 to 60 ms, a 75% reduction. Decode
rose from 2.188 to 3.313 tok/s, or 51%.

This was the largest end-to-end gain from redesigning a core forward-pass
Metal kernel. The winning kernel gave the GPU more independent work to
schedule. [Persistent
MoE](experiments/summaries/02-decode-moe-int4-and-router.md#dec-03) became the
production path.

Then reuse paid off. Expert selections repeat across tokens. A bounded cache
holding 16 routed experts per layer cut repeated expert I/O from about 166 to
88 ms/token.

A later experiment switched from LRU to LFU, reducing I/O from 72.6 to 64.8
ms/token in its paired benchmark.

A 64-token LFU window looked slightly better in simulation, but full decode
results were neutral or mixed. The simpler whole-run policy remained the
default. [The cache policy
record](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-01)
keeps the throughput rows, long-run results, and other controls.

Repeated selections made experts cacheable, but not predictable. Adjacent
layers within one token shared almost no experts.

Copying one layer's choices predicted only 7% of the next layer's experts.
That [offline
analysis](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-05)
stopped speculative cross-layer prefetch before it reached the runtime.

Past routing decisions identified weights worth keeping. The tested signal
could not identify the next layer's weights early enough to prefetch them.

The always-resident shared MLP offered a safer way to hide read time. The GPU
can execute it while the CPU fills routed-expert misses. In a later paired
benchmark, this coarse overlap raised throughput from 4.404 to 4.736 tok/s.

Finer-grained overlap did not help. Launching each routed-expert group as soon
as its read completed made synchronization harder. It also reduced throughput
from 4.799 to 4.648 tok/s and changed the generated output.

The coarser design ran cached expert hits first, then processed experts that
had required a read. It was faster and easier to synchronize. Overlap worked
when ownership and dependencies stayed explicit.

## Vectorization helped when it respected the storage layout

Wider loads looked like an easy win. Real tensor offsets disagreed.

The INT4 projections store affine groups of 64 with BF16 scale and bias.

In routed MoE, the successful kernel processed four groups together and loaded
activations with `half4`. This exposed more independent arithmetic and reduced
routed GPU time from 36.5 to 31.3 ms. Width helped because it shortened
dependency chains, not simply because the loads were wider.

A later resident-GEMV experiment exposed the trap. A 32-bit packed-load path
passed an offset-zero fixture, then produced garbage in real decode. Live
sub-tensors were guaranteed only 2-byte alignment.

Two `ushort` loads respected that alignment and still reduced LM-head GPU time
from about 21.5 to 16 ms.

Vector width depends on the live byte offset as well as the stored element
format. [The corrected
path](experiments/summaries/02-decode-moe-int4-and-router.md#dec-07) made
realistic tensor offsets part of later kernel tests.

## Splitting the work improved attention and prefill

Attention improved when we divided the work differently.

Split-KV attention divided the cached sequence across threadgroups. A second
pass merged their online-softmax partials. In isolation, it improved
sliding-window attention by about 3.3x and full 4K attention by about 4.1x.

The end-to-end effect depended on context length. Short-context decode stayed
nearly flat because attention occupied little of the token step. At longer
contexts, where attention mattered more, GPU time in that phase fell by about
28%.

Prefill needed a similar structural change. Replaying a prompt one token at a
time left the runtime on a scalar path. Chunked prefill replaced that replay
with bounded batches.

Increasing the chunk size from 32 to 128 reduced a 1,017-token prefill from
92.89 to 52.35 seconds.

Staged affine MPP then handled the quantized matrix work without changing the
source weights. The runtime dequantized a small FP16 tile, ran the hardware
matrix primitive, and discarded the tile.

That path improved the 512-token prefill benchmark by about 11% without
increasing memory use. [Direct shader-local
UInt4](experiments/summaries/06-prefill.md#pf-13) was 20.40% slower than the
staged path at M128.

Not every strong isolated result still mattered to the whole prefill. Batched
routed MoE reduced its kernel time by about 31%. End-to-end prefill improved by
only about 2%.

Activation-tile reuse improved individual remaining quantized matrix kernels
by 3-10%. After earlier improvements, however, those kernels accounted for only
about 0.4% of the remaining prefill time.

The optimization worked, but the affected kernels were now too small to
matter.

## Sampling removed repeated vocabulary scans

Sampling revealed an algorithmic problem rather than a slow kernel.

The old plain-temperature path repeatedly extracted candidates across the
entire vocabulary, taking about 2.651 seconds per token. Its replacement drew
one Gumbel value per vocabulary entry and selected the maximum in a single
reduction. Sampled decode returned to 5.86-5.89 tok/s, roughly matching greedy
decode.

The current default sampler combines Top-P 0.95 with Top-K 64 in a separate
staged reduction. It computes Top-P from the full distribution, caps the
surviving set at 64 tokens, and applies temperature to the final draw. The
Gumbel result describes the earlier untruncated sampling problem, not the
complete current sampling pipeline.

The baseline was unusually naive, so its speedup ratio would overstate the
overall value of the change. The useful result was simpler: the new path
removed a repeated vocabulary-wide operation.

## What local results missed

Some ideas failed outright. Others improved one measured component without
making the complete runtime faster. Both showed why a local result could
mislead.

### Read hints were unstable

`F_RDADVISE` through `fcntl` was a particularly convincing false start.

In one paired median, advice reduced I/O from 87.4 to 72.2 ms/token. Throughput
rose from 5.176 to 5.449 tok/s. Its behavior changed with runtime state. One
1,536-token probe fell from 5.687 tok/s without advice to 4.028 tok/s with it.
Later repeats did not reproduce that collapse, and some variants won again.

No tested policy could reliably identify when the hint would help. Production
therefore keeps it off. [The RDADVISE
record](experiments/summaries/04-rdadvise.md) shows both the wins and why it is
not the default.

Other read-side candidates followed the same pattern. Cache and read-ahead
flags were neutral. Speculative reads looked fast in isolation, then slowed
decode and stretched prefill from 82.50 to 123.64 seconds. Sorting reads by
file offset won once and lost on repetition. MTLIO was fast when data was
already warm, but few observed misses met that condition.

Bounded parallel `pread` remained the simplest design that worked across the
full workload. [The expert I/O
record](experiments/summaries/01-model-install-and-expert-io.md) keeps the
individual controls and measurements.

### The packed KV cache failed two gates

The TurboQuant packed K4/V4 KV cache looked like the obvious memory win. At 4K,
however, it saved only about 82 MiB compared with exact FP16.

The advantage disappeared as context grew. The packed cache expanded across
all 30 attention layers. The FP16 runtime used a fixed circular cache for 25 of
them. At longer contexts, the packed layout became larger.

It also performed worse against trusted reference outputs. The packed cache
lost its memory advantage and failed the quality evaluation, so the path was
removed.

### Fusion needed boundaries

Fusion rewarded restraint.

Some targeted fusions worked. QKV, layer-tail, and row-based head fusions
reached production after parity checks and full-token benchmarks. Each
combined operations with compatible shapes and data lifetimes.

A monolithic post-attention/pre-FFN fusion did not. It removed launches and
temporary buffers, but forced incompatible work into one wrapper. Throughput
fell from 2.756 to 1.811 tok/s.

### Local wins could disappear in the full token step

A faster kernel can still be irrelevant to the system around it.

Dividing the language-model head into smaller tiles reduced its GPU time from
14.2 to 13.1 ms/token. That 1.1 ms saving occupied less than 1% of a 167.7 ms
token step. The end-to-end result, 5.962 versus 5.926 tok/s, was inconclusive.
[The fusion
record](experiments/summaries/07-fusions-head-and-orchestration.md) has the
individual tests.

A [file layout derived from earlier routing
traces](experiments/summaries/03-expert-cache-prediction-and-layout.md#cache-08)
showed the same danger from another direction. It improved one recorded
workload by 3.61%, then slowed decode near a 4,096-token context by 16.1%.

[Fused Gumbel
sampling](experiments/summaries/08-sampling-tokenization-and-output.md#out-02)
was also inconclusive because repeat variation and the different generated
token sequences exceeded its small mean difference.

[Reusing Metal argument
buffers](experiments/summaries/06-prefill.md#pf-08) cut 21,217 allocations to
two, yet slowed long prefill by about 9%.

None of these candidates produced a repeatable whole-step gain.

Fewer reads, allocations, or dispatches can explain how a candidate might
help. End-to-end time and quality show whether it belongs in production.

## The method that worked

After enough ideas failed, the process became clear.

First, profile the whole token step. Then isolate the largest measured share
and reproduce its real matrix sizes and memory constraints. After changing it,
return to a clean end-to-end comparison.

The correctness check depends on the change. Exact transformations require
identical output. Kernels that reorder floating-point work are checked against
reference outputs. When page cache, warm-up, or thermals can bias a result,
control and candidate runs must alternate.

A candidate needs a repeatable gain to become the default and a repeatable loss
to be called slower. If run-to-run variation hides the difference, the result
is inconclusive and the default stays unchanged.

The model stayed 14.3 GB. The runtime got better at choosing what to read, what
to keep, what to run in parallel, and what to leave out. Clean local designs
often lost in the full runtime. That pattern mattered more than any single win.

For the current runtime, see [System design](SYSTEM_DESIGN.md).
[Benchmarks](BENCHMARKS.md) has the latest result and reproduction command.
