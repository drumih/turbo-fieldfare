# Qwen decode optimization plan

Status: DeltaNet, Phase 3, the fused Qwen head, 24-slot LFU, and cross-layer
command chaining integrated; schema-5 routed-tail profiling complete

Planning review: GPT-5.6 Luna, 2026-08-21

This plan targets the Qwen3.6 35B-A3B decode path. It separates profiling from
optimization so that a plausible storage or GPU explanation does not become a
production change without end-to-end evidence.

The plan does not set Gemma throughput as the initial target. Qwen and Gemma
have different architectures, kernels, and runtime maturity. The first target
is a repeatable same-host Qwen improvement that preserves output, state, memory,
and cancellation behavior.

## Current evidence

The following measurements came from the same host and the same short raw
completion shape, with 64 generated tokens:

| Model or setting | Decode rate |
| --- | ---: |
| Gemma 4 26B-A4B | 20.568 tok/s |
| Qwen, original comparison | 3.648 tok/s |
| Qwen, 16-slot LFU | 3.683 tok/s |
| Qwen, 24-slot LFU | 3.700 tok/s |
| Qwen, 32-slot LFU | 3.647 tok/s |

The slot experiment does not show a meaningful decode improvement from 24 or
32 slots. Keep 16-slot LFU as the control unless a later, valid experiment
reverses that result.

### Artifact location and weight I/O

The verified Qwen artifact was moved on 2026-08-22 from purgeable
`/private/tmp/qwen36-35b-a3b-4bit.gturbo` to the gitignored persistent path
`scratch/qwen36-35b-a3b-4bit.gturbo`. Both directories are on the same internal
APFS device, and the move preserved directory inode `26195355`, so it was an
atomic rename rather than an 18 GB copy. The manifest SHA-256 remains
`90f353b07d3bdfa7c226dfa461d02fc80bc07c26c9b0a73635d7e07cb1145940`; the
verified-install receipt SHA-256 remains
`72735217ec7f80e35c631359d3ea5116b280e174ee153d1a2d0754ad2c50ba47`.

Location does not change the runtime's read strategy or raw SSD performance.
`model_weights.bin` is mmap-backed and can incur page faults under memory
pressure. Routed experts are different: each layer file is opened lazily and
expert cache misses use explicit `pread` into a fixed per-layer Metal-visible
slot cache. The diagnostic expert-fetch timer therefore includes cache-plan
miss service and filesystem-cache effects; it is not proof that every logical
byte reached the physical SSD. Recent 256-token rows recorded 48,100 cache hits,
33,500 misses, and about 59.3 GB of logical estimated expert bytes, so streamed
expert I/O remains material even though moving the artifact cannot reduce it.

On 2026-08-22, diagnostic schema version 2 added opt-in routed-expert read
count, summed worker time, and maximum single-read time globally and per layer.
The production read path remains unchanged when diagnostics are disabled. One
controlled release-server row used the persistent artifact, 16-slot LFU, the
standard process warmup, and a 256-token completion. It passed token-count,
diagnostic-attribution, resource-sampling, and memory-pressure checks and
measured 17.00 tok/s with 0.996878 wall-time attribution. The row recorded:

| Metric | Result |
| --- | ---: |
| Expert-fetch wall time | 5,356.3 ms |
| Summed parallel read-worker time | 15,593.7 ms |
| Cache hits / misses | 48,100 / 33,500 |
| Average successful miss read | 0.465 ms |
| Maximum single read | 10.135 ms |
| Summed worker time / fetch wall | 2.91x |
| Top layer / top 5 / top 10 fetch share | 4.83% / 21.47% / 35.59% |

The summed read time is parallel worker time and must not be added to fetch
wall time. The broad per-layer distribution falsifies the current hypothesis
that a few pathological layers or APFS extents dominate fetch wall. One
10.135 ms read outlier exists, but it cannot explain 5.356 seconds of aggregate
fetch wall. The next I/O candidate should reduce cache-miss demand or improve
admission quality; do not duplicate or repack the 18 GB artifact for extent
testing from this evidence. Larger LFU caches and LRU have already failed to
provide a new mechanism worth repeating.

The external harness initially rejected newer schemas before measurement
because it accepted only version 1. The completed rows used an in-memory adapter
that mapped only `schema_version` to 1 for the existing validator; all
decode-step, attribution, completion, token-ID, resource, and memory-pressure
checks remained active. Update the external harness to accept schema version 3
before using this row in a formal interleaved comparison.

### Cache-direction follow-up

Diagnostic schema version 3 adds opt-in per-layer routed-expert IDs and routing
weights for every decode step. This permits exact offline policy replay without
changing the production cache or issuing speculative reads. A validated
256-token trace under 16-slot LFU reproduced 48,100 hits and 33,500 misses at
16.84 tok/s with 0.995591 attribution. Resource sampling and memory-pressure
checks passed.

One warmup replay followed by one scored replay of the same 40-layer route
stream found no better memory-neutral replacement policy:

| Policy | Miss change versus LFU |
| --- | ---: |
| Lifetime LFU | control |
| Decayed LFU | +0.07% to +4.80% |
| Route-weighted LFU | +1.28% |
| Decayed route-weighted LFU | +1.32% to +6.67% |
| Segmented LFU/LRU | 0.00% to +14.58% |
| LRU | +14.58% |
| Reuse-interval prediction, held out | +14.67% to +17.77% |

A fixed-total heterogeneous allocation trained on the first half of the trace
reduced held-out misses by only 0.86%. Even an allocation selected with
knowledge of the held-out half reduced them by only 2.50%. By contrast,
future-aware Belady replacement reduced misses by 19.07%, from 32,886 to
26,616 on the warmed replay. Capacity is therefore not the absolute limit, but
the tested online recency, frequency, route-weight, reuse-interval, and
per-layer-allocation signals do not recover the oracle gap.

Because the current runtime is much faster than the historical slot-capacity
experiment, 24-slot LFU was screened again and then measured with three warmup
and five interleaved 256-token cycles against 16-slot LFU:

| Metric | 16 slots | 24 slots | Change |
| --- | ---: | ---: | ---: |
| Median decode | 17.26 tok/s | 17.94 tok/s | +3.94% |
| Cache misses | 33,500 | 26,145 | -21.96% |
| Expert-fetch wall | 4,977.4 ms | 4,354.5 ms | -12.51% |
| Summed read-worker time | 14,127.7 ms | 11,211.8 ms | -20.64% |
| Median peak backend RSS | 1.275 GB | 1.676 GB | +0.401 GB |
| Minimum available memory | 3.941 GB | 3.586 GB | -0.355 GB |

All measured rows had exact token parity, valid attribution and resource
samples, and valid memory-pressure snapshots. The generated rates were
17.18/17.26/17.32/17.14/17.76 tok/s for 16 slots and
17.48/18.23/17.94/17.62/18.29 tok/s for 24 slots. The throughput gain is real
but misses the 15% acceptance gate and spends additional memory. A later
combined comparison below determines its production disposition.

The correctness-proven fused Qwen greedy head, Qwen-specific LM-head function
constants, and 24-slot LFU cache were then measured together against the active
16-slot runtime. Three warmups and five interleaved 256-token cycles produced
17.77 tok/s for the control and 18.96 tok/s for the combined candidate, a
6.70% median gain. Control rates were 17.77/17.69/18.18/17.92/17.77 tok/s;
candidate rates were 19.13/19.06/18.60/18.53/18.96 tok/s. Every row produced
the exact expected output SHA-256
`62578360fa5015aaf9505788a93e824e1b6463a3566fce8e2e7ec5fad9c341ff`.

This stack is now the production default: fused greedy output remains selected
unless logits are explicitly requested, and LFU uses 24 cache slots by default.
The combined gain is useful but still below the 15% project acceptance gate, so
it is an integrated incremental stack rather than a gate-clearing phase result.
The separately tested fused shared/routed combine remains excluded because its
3.92% subset result was noisy and did not receive equivalent formal validation.

After integration, a fresh release build was confirmed with diagnostics off,
24-slot LFU, one 64-token process warmup, and one 256-token measured request. It
produced 18.53 tok/s and the same expected output hash. Peak backend RSS was
1.760 GB, minimum available memory was 4.071 GB, and peak wired memory was
5.775 GB across 34 resource samples. This confirms the source-built result lies
within the formal candidate range; it is not an additional formal comparison.

A subsequent current-runtime 32-slot screen confirmed that more capacity does
not continue the throughput trend. Offline replay predicted 19,653 misses,
40.24% fewer than 16 slots, and the live row recorded 20,746 misses. Despite
that hit-rate improvement, the validated row fell to 16.19 tok/s, expert-fetch
wall rose to 4,776.0 ms, peak backend RSS reached 1.928 GB, and peak wired
memory reached 6.245 GB. Attribution was 0.995234 and memory-pressure and
resource checks passed. This agrees with the historical 32-slot result: the
additional resident buffers reduce read count but increase enough memory and
system cost to lose end-to-end throughput. Do not promote 32 slots or spend a
formal interleaved campaign on it without a new memory-cost mechanism.

The schema-3 runs used an in-memory external-harness adapter that mapped only
the schema number to version 1 for the unchanged validator. All token-count,
token-ID, attribution, resource, and pressure checks remained active. The
formal comparison output is at
`/private/tmp/turbo-fieldfare-qwen-cache-slots-interleaved/comparison.json`.

Absolute throughput across these experiments must not be read as a source
regression. A three-cycle alternating check of the current binary measured a
17.89 tok/s median with diagnostics disabled and 17.77 tok/s with schema-3
diagnostics enabled, only 0.67% instrumentation overhead, with exact output
hash parity. More importantly, the preserved Qwen-specialized fused binary
whose historical median was 18.88 tok/s produced 18.06 tok/s under the current
host state with the same output hash. Historical fused runs had 5.303-5.394 GB
minimum available memory; current comparable rows had 4.384-4.491 GB. Thus the
headline difference from 18.9 to the high-17 range combines a stashed roughly
1% fused-head optimization with same-binary host and filesystem-cache drift.
Use interleaved same-session deltas, not absolute rates from different runs, to
judge changes.

Use the persistent path in future CLI, server, and benchmark commands. Temporary
external harnesses that still name `/private/tmp/qwen36-35b-a3b-4bit.gturbo`
must be given the new artifact argument rather than recreating or duplicating
the model.

A separate 4,002-token Qwen request completed in 893.452 seconds with 106 output
tokens. Live stack samples observed work in `PreadExpertStreamer` and `pread`,
but this does not prove that physical SSD traffic dominates decode. Filesystem
cache behavior, task scheduling, command-buffer waits, route readback, and GPU
work remain confounded until the runtime exports aggregate diagnostics.

## Current decode dependency graph

[`QwenForwardRunner`](../Sources/TurboFieldfare/Runtime/Inference/QwenForwardRunner.swift)
currently submits and synchronously completes:

1. One embedding command buffer.
2. Three command buffers for each of 40 layers:
   mixer, shared-expert/router, and routed-expert/combine.
3. One final-head command buffer.

That is:

```text
1 + (3 * 40) + 1 = 122
```

The count covers the Qwen forward runner, not the complete token loop.
[`RawCompletion`](../Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift)
also performs sampling work. For a 64-token completion, the first generated
token is normally seeded by prompt logits, so the loop performs 63 Qwen
`produce` calls and 64 sampling steps.

The 122-wait structure is a measured code fact. The following are hypotheses
until aggregate diagnostics quantify them:

- Command-buffer completion and CPU/GPU synchronization are a major cost.
- Expert fetch is a major cost.
- Route readback and cache-plan construction are a material cost.
- Exact-demand expert I/O can overlap enough shared-expert GPU work to improve
  the full token step.

## Goals and acceptance gates

Profiling must account for at least 90% of the defined token-step wall time.
An optimization is accepted only when all of these conditions hold:

- Same-host median decode improves by at least 15%.
- Baseline and candidate use the same model receipt, build mode, prompt,
  generation settings, cache settings, and run ordering policy.
- Greedy token IDs and output hashes match the baseline.
- DeltaNet recurrent state and full-attention KV state remain aligned.
- Continuation, prompt replay, and cancellation remain correct.
- Cache hit/miss accounting is internally consistent.
- No swap, sustained memory pressure, invalid resource sample, or artifact
  attribution failure occurs.
- Package tests pass through `Scripts/test.sh`.

The recent 3.683 tok/s row implies a provisional 15% threshold of about
4.24 tok/s. The formal threshold must be calculated from a fresh interleaved
baseline rather than treating one row as a permanent control.

## Phase 0: isolate the prefill repair

The working tree currently contains Qwen multi-chunk prefill changes in:

- [`QwenForwardRunner.swift`](../Sources/TurboFieldfare/Runtime/Inference/QwenForwardRunner.swift)
- [`PrefillRuntimeConfig.swift`](../Sources/TurboFieldfare/Runtime/Prefill/PrefillRuntimeConfig.swift)

Review and land that correctness repair independently. Decode profiling and
optimization must preserve it without folding unrelated decode changes into the
same review.

Rollback point: restore only the previous decode implementation. Do not remove
or rewrite the multi-chunk prefill repair as part of a decode rollback.

## Phase 1: aggregate and export diagnostics

### Runtime aggregate

Extend
[`QwenDecodeDiagnostics.swift`](../Sources/TurboFieldfare/Runtime/Inference/QwenDecodeDiagnostics.swift)
with a versioned aggregate value containing:

- Decode-step count.
- Summed forward wall time.
- Embedding, layer, final-head, and expert-fetch time.
- Command-buffer submission count.
- Router evaluation count.
- Routed expert count.
- Cache hit and miss counts.
- Logical estimated expert bytes.
- Per-layer elapsed and expert-fetch totals.
- Attributed and residual wall time.

Logical estimated bytes must remain explicitly labeled as an estimate. It is
`missCount * expertStride`, not measured physical SSD traffic.

Add enough phase timing to distinguish:

- Mixer and router completion.
- CPU route readback and cache-plan construction.
- Shared-expert completion.
- Expert-fetch wait.
- Routed-expert/combine completion.
- Sampling and outer-loop residual time.

Do not serialize JSON, write files, or perform extra Metal operations inside the
timed token path. Capture one completed token diagnostic after `produce`
returns, then aggregate in memory.

### Raw completion integration

Add an optional Qwen aggregate to `RawDecodeResult`. Preserve existing
initializers with a default `nil` value.

Aggregate only post-prefill Qwen `produce` calls. Exclude:

- Chunked prompt prefill.
- Scalar prompt replay.
- The first generated token when it is seeded directly from prompt logits.

Tests belong in:

- [`QwenDecodeDiagnosticsTests.swift`](../Tests/TurboFieldfare/Core/Runtime/Inference/QwenDecodeDiagnosticsTests.swift)
- [`RawCompletionLoopTests.swift`](../Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests.swift)

Cover zero-step behavior, layer merging, overflow handling, attribution math,
63-versus-64 forward-step accounting, and exclusion of prompt work.

### CLI export

Add an opt-in `--diagnostics-json <path>` option in:

- [`Args.swift`](../Sources/TurboFieldfareCLI/Args.swift)
- [`Run.swift`](../Sources/TurboFieldfareCLI/Run.swift)
- [`CLIArgumentsTests.swift`](../Tests/TurboFieldfare/Core/CLI/CLIArgumentsTests.swift)

Write one JSON document after generation timing completes. Keep generated text
on stdout and preserve the existing timing footer on stderr.

### Server and harness export

After the CLI schema is stable, add opt-in server export. A normal OpenAI
response must not change unless diagnostics are explicitly enabled.

- Non-streaming: add one top-level `turbo_fieldfare_diagnostics` object.
- Streaming: add the object only to the final usage chunk.
- Serialize once after generation, never once per token.
- Preserve normal OpenAI `usage` fields.

Likely server files:

- [`ServerInference.swift`](../Sources/TurboFieldfareServer/Core/ServerInference.swift)
- [`HTTPServer.swift`](../Sources/TurboFieldfareServer/Core/HTTPServer.swift)
- [`ServerArguments.swift`](../Sources/TurboFieldfareServer/Core/ServerArguments.swift)
- [`HTTPServerTests.swift`](../Tests/TurboFieldfareServer/HTTPServerTests.swift)

The external benchmark harness should retain the versioned diagnostics object
in its machine-readable report and require it for profiling runs only.

## Phase 2: establish the formal baseline

Run baseline and later candidate rows with:

- Same host, artifact receipt, release build, and runtime settings.
- Temperature 0 and exact token-ID capture.
- 16-slot LFU and RDADVISE off.
- Three warmups and at least five valid measured runs per configuration.
- Interleaved baseline/candidate ordering.
- 64-, 256-, and 512-token decode workloads.
- Fresh-server comparisons when evaluating cache policy or capacity.

Record:

- Prompt and generated token counts.
- Time to first token, decode time, total time, and decode tok/s.
- Diagnostic attribution and residual time.
- Command-buffer count.
- Expert-fetch wait, hits, misses, and logical estimated bytes.
- CPU route-planning time.
- Peak RSS, available memory, pressure state, and run validity.
- Token IDs or an output hash suitable for exact parity checks.

Do not proceed to optimization until at least 90% of token-step wall time is
accounted for. If the residual is too large, improve diagnostics rather than
selecting a candidate by intuition.

The formal baseline passed on 2026-08-21 for Qwen3.6 35B-A3B using the settings
above. Three fresh-process parity probes and all five measured runs per target
produced identical greedy token IDs. Mean measured rates were 3.72 decode
tokens/s for 64-token completions, 3.75 decode tokens/s for 256-token
completions, and 3.73 decode tokens/s for 512-token completions. The release
build and package validation also passed.

## Phase 3: reduce command-buffer synchronization

The first candidate should reduce synchronization without changing model math.
Refactor the decode-only `encodeLayer` and `encodeMoE` boundary so one command
buffer contains:

1. Input RMS normalization.
2. Attention or DeltaNet mixer.
3. Residual update.
4. Post-attention RMS normalization.
5. Shared expert.
6. Router.

Wait once for router output, read the exact route IDs, construct the existing
cache plan, fetch required experts, then submit routed expert/combine work.

The expected forward-runner command count becomes:

```text
1 + (2 * 40) + 1 = 82
```

### Phase 3 measurement result

The candidate was measured against a clean `HEAD` baseline using the same
Qwen3.6 35B-A3B artifact, host, prompt, temperature `0`, 16-slot LFU cache,
prefill settings, `max_context=16384`, and RDADVISE off. The baseline release
binary SHA-256 was
`e2e68e456ffdc6e8d1a5f2185e242b36b72299e1d8038fdddb19b6ce19590e03`; the
candidate release binary SHA-256 was
`928bb31b737e35234bab8223085163a8c09f21f5745389379cf321f51a6b3abf`.

Five measured runs per target produced these medians:

| Completion target | Baseline | Candidate | Change | Forward-runner submissions per decode step |
| ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 3.76 tok/s | 3.84 tok/s | +2.13% | 122 -> 82 |
| 256 tokens | 3.78 tok/s | 3.87 tok/s | +2.38% | 122 -> 82 |
| 512 tokens | 3.77 tok/s | 3.86 tok/s | +2.39% | 122 -> 82 |

All three exact output hashes matched. The candidate workload validation and
package tests passed, but the candidate is rejected because every measured
gain is below the required 15% median improvement. The baseline and candidate
were run as separate fresh-process series rather than interleaved rows, which
is a protocol deviation and prevents this result from being treated as a
final accepted performance comparison. The Phase 2 report format also did not
contain resource samples with backend-memory attribution, so the resource
gate remains unverified. The command-buffer reduction is therefore retained
as an exploratory result, not an accepted Phase 3 optimization.

Safety requirements:

- Keep all dependent encoders on the same Metal command queue.
- Read route IDs only after the combined command buffer completes.
- Keep expert buffers and argument buffers alive through routed command
  completion.
- Preserve cancellation cleanup before reset or runner reuse.
- Do not overlap separate Qwen layers; recurrent and KV state remain strictly
  ordered.
- Verify the diagnostic command count instead of assuming coalescing occurred.

Reject the candidate if the same-host median gain is below 15%, even if the
command-buffer count improves.

Rollback point: restore the current three-stage decode layer path while keeping
diagnostics and the prefill repair.

## DeltaNet recurrent-kernel follow-up

Profiling after the Phase 3 rejection attributed about 83% of Qwen forward time
to mixer work and only 6-7% to expert-fetch wait. The decode recurrent kernel
was launching one thread per value head and serializing every independent value
column inside that thread. The follow-up candidate changed
[`QwenGatedDeltaNet.swift`](../Sources/TurboFieldfare/Kernels/LinearAttention/QwenGatedDeltaNet.swift)
and
[`linear_attention.metal`](../Sources/TurboFieldfare/Metal/LinearAttention/linear_attention.metal)
to dispatch one thread per `(value head, value column)`. The production grid is
128 columns by 32 value heads. Each thread retains the original ordered
key-dimension accumulation for its column, so columns gain parallelism without
changing their arithmetic order or sharing writable state.

The rejected Phase 3 synchronization change was removed before this candidate
was built. The candidate therefore retains the original 122 forward-runner
submissions per decode step and isolates the recurrent-kernel change. A strict
alternating baseline/candidate comparison used three warmup cycles and five
measured cycles per arm and completion target. The clean baseline server
SHA-256 was
`e2e68e456ffdc6e8d1a5f2185e242b36b72299e1d8038fdddb19b6ce19590e03`, and
the candidate server SHA-256 was
`4f42c2594420b3f12229e65e4ce52cd201216bcf6e709739836b6bae65f21c00`.

| Completion target | Baseline median | Candidate median | Change |
| ---: | ---: | ---: | ---: |
| 64 tokens | 3.72 tok/s | 16.60 tok/s | +346.24% |
| 256 tokens | 3.73 tok/s | 16.48 tok/s | +341.82% |
| 512 tokens | 3.72 tok/s | 16.24 tok/s | +336.56% |

Exact greedy token IDs matched on every baseline/candidate row. The output
SHA-256 values were
`8e65e1b5adf49bd49523cffc8ee00c4d896927019d76abbc541b2ca861f61ddf`,
`62578360fa5015aaf9505788a93e824e1b6463a3566fce8e2e7ec5fad9c341ff`, and
`50286bf4360a7ed31e4011fd3e0ff360b2fad7c1c8c55bac700882e4b91b90d1` for
the 64-, 256-, and 512-token targets. The standalone candidate protocol also
passed with medians of 16.13, 16.48, and 16.34 tok/s, no validation errors, and
a minimum diagnostic attribution ratio of 0.995057.

All four focused DeltaNet tests passed, followed by 765 package tests across
141 suites and a release build. The 16 interleaved arm runs had no invalid
resource samples, a maximum peak RSS of 1.577 GB, and at least 5.245 GB
available memory.

The final G4 service gate passed on 2026-08-22. It covered a 4,002-token prompt,
a 15,362-token near-context-limit prompt, 50 successful sequential requests,
client-disconnect cleanup, ten successful server restarts, and orphan-listener
cleanup. Resource sampling was valid; peak process-group RSS was 1.483 GB
against the 3.0 GB limit, with at least 5.175 GB available memory. The temporary
Python harness required `jinja2==3.1.6` for chat-template rendering. Its
tokenizer-only Transformers installation did not include PyTorch, which is
expected because inference remained in the Swift/Metal server. No repository
dependency changed.

A corrected gate rerun on 2026-08-23 measured queue admission rather than
replacement completion and passed against the schema-4 release server. It
completed the 4,002-token and 15,362-token workloads, passed 50 of 50 sequential
requests and 10 of 10 restarts, released the queue 0.539 seconds after graceful
disconnect, left no listener, and peaked at 1.814 GB process-group RSS. This
supersedes the earlier disconnect timing while preserving the other G4 results.

This candidate clears the 15% performance gate with exact parity and passes the
correctness, lifecycle, and resource gates, so the recurrent-kernel change is
accepted.

### Stacked Phase 3 full verification

After Phase 3 merged, a full alternating A/B sweep compared commit `62b29f7`
(DeltaNet only) against merged commit `42b898d` (DeltaNet plus Phase 3). The
protocol used three warmup cycles and five measured cycles per arm at 64, 256,
and 512 completion tokens. The baseline release binary SHA-256 was
`28f63249cdbacff86ec84e9a366e679788905f5bae02d660431f0aaa11bab671`; the
candidate release binary SHA-256 was
`a0b7f2996ef7019f2ed5bdd262e7452e801b63cb39540ab6649defe151ece683`.

| Completion target | DeltaNet-only median | Phase 3 median | Change | Submissions per decode step |
| ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 15.35 tok/s | 17.21 tok/s | +12.12% | 122 -> 82 |
| 256 tokens | 15.49 tok/s | 17.58 tok/s | +13.49% | 122 -> 82 |
| 512 tokens | 15.32 tok/s | 17.42 tok/s | +13.71% | 122 -> 82 |

All 16 arm processes completed their workloads. Resource sampling produced
1,121 valid samples with no unavailable fields; maximum peak RSS was 1.549 GB,
and minimum available memory was 4.327 GB. A supplemental fresh-process row per
arm confirmed exact token parity at all three lengths with output SHA-256 values
`8e65e1b5adf49bd49523cffc8ee00c4d896927019d76abbc541b2ca861f61ddf`,
`62578360fa5015aaf9505788a93e824e1b6463a3566fce8e2e7ec5fad9c341ff`, and
`50286bf4360a7ed31e4011fd3e0ff360b2fad7c1c8c55bac700882e4b91b90d1`.

The harness stopped before writing its final comparison report because the
64-token row failed the configured gate. The remaining medians were calculated
from the five completed measured samples per arm. The full sweep supersedes the
earlier three-sample 64-token result: Phase 3 consistently improves decode and
preserves exact output, but all three medians fall below the plan's 15%
acceptance threshold.

## Phase 4: overlap exact-demand expert I/O

Phase 4 began after the recurrent-kernel and Phase 3 changes merged. The earlier
profile attributed only 6-7% of baseline forward time to expert-fetch wait, so
this remained an evidence-gathering candidate rather than a presumed win.

The decode-only implementation used this schedule:

1. Submit mixer/router work.
2. Queue shared-expert work.
3. Wait for mixer/router completion.
4. Read exact route IDs and reserve the exact cache plan.
5. Start exact expert fetch while shared-expert GPU work executes.
6. Await shared-expert and expert-fetch completion.
7. Submit routed expert/combine work.

Mixer/router, shared expert, and routed combine remained on the same Metal
queue. Queue order preserved the normalized-input dependency, while the
asynchronous pread overlapped the already-committed shared-expert command.
Every downstream error path drained the submitted shared command before
returning, and routed combine was not encoded until both fetch and shared
completion succeeded. Prefill and cross-layer scheduling were unchanged.

### Phase 4 measurement result

A short alternating A/B run compared merged Phase 3 against Phase 4 using one
64-token warmup cycle and three measured cycles per arm. The baseline release
binary SHA-256 was
`6886e35655d5902de07fbee7b6d9ce56262efa11afe640e5150e68a30752b1fc`; the
candidate release binary SHA-256 was
`e5166ad864e062365bab750fc4361dbd1f66ca7df0ec64beecf7ae32c9b31768`.

| Completion target | Phase 3 baseline | Phase 4 candidate | Change | Submissions per decode step |
| ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 18.53 tok/s | 18.91 tok/s | +2.05% | 82 -> 122 |

Baseline samples were 18.54, 18.53, and 17.92 tok/s. Candidate samples were
18.91, 18.89, and 18.97 tok/s. Exact token IDs and output SHA-256 matched across
arms, diagnostic attribution remained above 0.998, and no validation errors
occurred. All eight resource samples were valid, maximum peak RSS was 1.355 GB,
and minimum available memory was 5.291 GB. The focused recurrent suite passed 4
tests, the full package suite passed 765 tests across 141 suites, and the release
build completed.

The candidate was rejected because the +2.05% median improvement was well below
the 15% acceptance gate. Separating shared-expert work also restored the third
per-layer command buffer, undoing Phase 3's 82-submission decode shape. The
runtime candidate was rolled back completely; only this measured result remains.

## Fused greedy LM-head candidate

This candidate reused `LMHeadChainInt4` for Qwen's untied 2,048-wide,
248,320-row output projection. Pure greedy generation performed the final BF16
RMSNorm, INT4 projection, and GPU argmax without materializing or scanning full
FP16 logits on the CPU. Chunked prefill applied the same fused chain to its final
hidden row and returned a greedy-token seed. The runner factory passed
`RuntimeConfiguration.headPath` into Qwen construction, so sampled generation
and the server retained the existing full-logits path.

### Correctness diagnosis and repair

The first standalone run reported 21.973 tok/s, but its repeated-token output
was invalid. `QwenForwardRunner` exposed `usesFusedGreedyHead` and
`lastGreedyToken` without declaring `FusedGreedyLogitProducer` conformance.
`RawCompletion` therefore ignored the correctly computed fused token and sampled
an unwritten logits buffer. A same-hidden diagnostic confirmed that the fused
head selected token `264`, matching materialized logits, while the generation
loop emitted stale token `623`.

The repair added the missing protocol conformance. Prompt-state snapshots also
saved and restored `lastGreedyToken`, and exact fused replay seeded directly
from that token rather than stale logits. A focused poisoned-logits regression
test covered this replay contract. All temporary Metal, score-rounding, and
logging probes were removed.

The corrected candidate binary SHA-256 was
`a843075e4ea92162f237d54cc66f8512d4dfc051431ce0ef7ab2260d14e39038`.
A fresh cold two-token comparison and fresh cached 64-, 256-, and 512-token
comparisons matched exact token IDs across the logits and fused binaries.

### Corrected formal performance result

A corrected formal alternating comparison used three warmup cycles and five
measured cycles per arm at all three completion targets. All 16 arm processes
completed every workload and cleaned up their listeners. The harness verified
64-token parity, then stopped at the configured 15% performance gate; the
supplemental fresh comparisons established parity at the remaining targets.
The retained server logs produced these cached-request duration medians:

| Completion target | Logits median | Fused median | Duration reduction |
| ---: | ---: | ---: | ---: |
| 64 tokens | 3.376 s | 3.406 s | -0.88% |
| 256 tokens | 13.924 s | 13.503 s | +3.12% |
| 512 tokens | 27.771 s | 27.619 s | +0.55% |

Resource and lifecycle checks passed. Sampling captured 1,018 valid rows with a
1.617 GB maximum process-group RSS and at least 4.863 GB available memory. No
model process or listener remained after the run.

The corrected formal sweep does not reproduce the invalid 21.973 tok/s result
and does not clear the 15% performance gate. Never cite that invalid-output run
as valid throughput. Its apparent speed also coincided with unusually low
expert-fetch wait, so most of the difference did not come from the head.

### Qwen head specialization follow-up

Corrected diagnostics showed why head work cannot recover 21.973 tok/s by
itself. At 512 tokens, removing CPU sampling saved about 362 ms and the fused
projection saved about 133 ms, while forward execution still took more than 27
seconds.

The generic fused kernel did leave a smaller safe opportunity: only Gemma's
2,816 by 262,144 head used compile-time Metal function constants. A Qwen-specific
2,048 by 248,320 pipeline was added without changing kernel arithmetic. A
one-warmup, three-measurement alternating 256-token subset compared the corrected
generic fused binary against the specialized fused binary. Exact token IDs and
output SHA-256 matched.

| Path | Measured decode rates | Median | Median fused-head time |
| --- | --- | ---: | ---: |
| Generic fused | 18.70, 18.60, 18.83 tok/s | 18.70 tok/s | 698.69 ms |
| Qwen-specialized fused | 18.91, 18.80, 18.88 tok/s | 18.88 tok/s | 677.44 ms |

The specialization improved median head time by 3.04% and end-to-end decode by
0.96%. This is discovery evidence rather than a formal acceptance run. It is
the only post-correctness optimization retained in the stash because it
preserved exact output, added no runtime control surface, and directly improved
the measured phase.

### Post-head decode follow-ups

Four arithmetic-preserving Qwen decode candidates were measured after the head
specialization. All preserved exact 256-token output SHA-256
`62578360fa5015aaf9505788a93e824e1b6463a3566fce8e2e7ec5fad9c341ff`, but none
cleared the 15% gate, so all runtime changes were removed.

| Candidate | Discovery result | Decision |
| --- | ---: | --- |
| Routed MoE function constants (`D=2048`, `F=512`, top-k 8) | 18.84 -> 18.87 tok/s (+0.16%); routed-combine median 3,225.46 -> 3,191.59 ms | Reject as neutral |
| Fuse shared/routed combine with residual add | 64-token median 15.29 -> 15.89 tok/s (+3.92%); combine median 926.95 -> 909.27 ms | Reject below gate and above run noise |
| Reuse one routed-expert Metal argument buffer | 17.31 -> 17.22 tok/s (-0.52%) | Reject as neutral-negative |
| Add Qwen shared-expert INT4 function constants | 17.85 -> 17.37 tok/s (-2.69%); shared-phase median 5,097.01 -> 5,206.19 ms | Reject as regression |

The combine/residual candidate was bit-exact against the original two-kernel
sequence by explicitly retaining the intermediate FP16 rounding. Its first
256-token pair showed a small local reduction, but later rows suffered broad
host slowdown in mixer and expert-fetch phases while available memory declined.
A shorter three-warmup, five-measurement 64-token confirmation retained exact
parity and valid resource sampling, but the intended phase improved only 1.91%
and individual rows varied much more than the change. It is the only rejected
follow-up worth reconsidering experimentally, after finer GPU timing exists.

The argument-buffer run was stable, retained identical 48,100 cache hits and
33,500 misses per row, and confirmed that eliminating those allocations does
not improve throughput. The shared-expert specialization was slower in every
measured comparison.

These results narrow the remaining opportunity: launch and setup reductions in
the routed tail are too small on this host. Future work should begin with finer
GPU timing inside the combined mixer/shared/router command buffer before
changing another kernel, because its current wall-time counters overlap and do
not identify which enclosed encoder dominates.

### Non-overlapping GPU stage result

Diagnostic schema version 4 brackets mixer, shared-expert, and router execution
with four Metal stage-boundary marker passes in their existing command buffer.
The timer is opt-in with diagnostics, defaults off, and reports a sample only
when all eight boundary timestamps are valid and ordered. The release server
SHA-256 was
`5494bc0d2fccd79356a289a9c12e574a5c115eee50321a577839be5a693ebdd2`;
the model manifest and verified-install SHA-256 values remained
`90f353b07d3bdfa7c226dfa461d02fc80bc07c26c9b0a73635d7e07cb1145940` and
`72735217ec7f80e35c631359d3ea5116b280e174ee153d1a2d0754ad2c50ba47`.

Three alternating fresh-process 256-token cycles compared diagnostics off and
on with 24-slot LFU, a 16,384-token context, single-prefix caching, and chunked
prefill. Every diagnostics row resolved all 10,200 expected layer samples and
produced the same output SHA-256 as its control:
`62578360fa5015aaf9505788a93e824e1b6463a3566fce8e2e7ec5fad9c341ff`.

| Metric | Result |
| --- | ---: |
| Diagnostics-off median | 18.02 tok/s |
| Diagnostics-on median | 17.73 tok/s |
| Instrumentation cost | 1.61% |
| Maximum process-group RSS | 1.967 GB |
| Minimum available memory | 4.891 GB |

The three exclusive GPU-stage rows were stable:

| Stage | Median total over 255 decode steps | Median share of measured GPU window |
| --- | ---: | ---: |
| Mixer | 168.163 ms | 81.62% |
| Shared expert | 20.576 ms | 10.07% |
| Router | 16.874 ms | 8.17% |

Mixer dominates this three-stage GPU window, but the full window is only about
0.81 ms per decode step and roughly 1.4% of end-to-end decode time. Eliminating
it entirely cannot clear the 15% gate. Do not select another mixer-kernel
micro-optimization from the 82% share alone. The next investigation should
measure host command submission and synchronous waits against the routed-expert
fetch/execute tail, then evaluate a redesign only if that larger interval has a
credible overlap or batching mechanism.

### Routed-tail timing result

Diagnostic schema version 5 adds opt-in host clocks for routed setup, command
encoding, commit, and completion wait, plus four Metal markers around routed
phase 1, phase 2, and combine. The release server SHA-256 was
`9160ae1c597a6d3390126fbb88fbaba2094a630bd6963bd103e54670d7f695ed`.

Three alternating fresh-process 256-token cycles used the same 24-slot LFU,
16,384-token context, single-prefix cache, and chunked-prefill settings as the
schema-4 profile. Every diagnostics row resolved all 10,200 expected samples in
both GPU groups. All six rows produced output SHA-256
`62578360fa5015aaf9505788a93e824e1b6463a3566fce8e2e7ec5fad9c341ff`.

| Metric | Result |
| --- | ---: |
| Diagnostics-off rates | 15.92, 16.29, 15.54 tok/s |
| Diagnostics-on rates | 14.69, 14.92, 14.25 tok/s |
| Diagnostics-off median | 15.92 tok/s |
| Diagnostics-on median | 14.69 tok/s |
| Observed median runtime overhead | 8.37% |
| Maximum process-group RSS | 1.745 GB |
| Minimum available memory | 4.099 GB |
| Maximum wired memory | 5.967 GB |

The three pairs varied too widely to treat 8.37% as a stable instrumentation
cost. Diagnostics remain unsuitable for production throughput measurements;
use diagnostics-off interleaved arms for candidate comparisons.

The median diagnostics-on row spent 16,847.780 ms in forward execution. Host
subfields were:

| Routed host field | Median total | Share of forward |
| --- | ---: | ---: |
| Route planning | 72.071 ms | 0.43% |
| Expert fetch | 4,442.438 ms | 26.37% |
| Routed setup | 108.639 ms | 0.64% |
| Command encoding | 278.042 ms | 1.65% |
| Command commit | 29.436 ms | 0.17% |
| Command completion wait | 4,211.043 ms | 25.00% |
| Routed expert/combine wall | 4,518.642 ms | 26.82% |

These fields are not additive. Command wait is inside routed expert/combine,
and summed expert-read worker time overlaps within expert-fetch wall time. The
median row recorded 26,145 reads, 55,455 cache hits, 26,145 misses, and
11,190.565 ms of summed parallel worker time. Expert fetch plus routed
expert/combine are sequential and account for 8,961.080 ms, or 53.19% of the
median forward wall.

The exclusive routed GPU intervals were much smaller:

| Routed GPU stage | Median total | Share of routed GPU | Share of forward |
| --- | ---: | ---: | ---: |
| Phase 1 | 339.052 ms | 92.32% | 2.012% |
| Phase 2 | 10.949 ms | 2.98% | 0.065% |
| Combine | 17.254 ms | 4.70% | 0.102% |
| Total | 367.255 ms | 100.00% | 2.180% |

The routed command waited 412.847 microseconds per decoded layer execution,
while the three bracketed GPU phases occupied 36.005 microseconds. The
3,843.788 ms aggregate gap includes marker encoders and any queueing,
residency, scheduling, and completion-notification cost outside the bracketed
kernels. It must not be labeled removable launch overhead without a candidate
comparison.

Reject another phase-1, phase-2, combine, argument-buffer, or setup
micro-optimization: even deleting the largest measured GPU stage cannot clear
the 15% gate. Expert fetch remains material, but the cache-policy, capacity,
allocation, and exact-demand overlap experiments above provide no new
memory-safe mechanism to test.

### Cross-layer command chaining result

The selected prototype chains routed layer N with layer N+1's mixer, shared
expert, and router in one command buffer. The final routed buffer also carries
the language-model head. Queue order preserves exact layer arithmetic and
recurrent/KV dependencies; fetched expert views remain alive until the combined
command completes. The optimized schedule is:

```text
embedding + initial front + 40 routed/next-front-or-head = 42 submissions
```

A diagnostics-export probe with Metal timing disabled measured exactly 42
forward-runner submissions per decoded step. Schema-5 diagnostics retain the
unchained 82-submission schedule because their two independent marker groups
require separate front and routed command buffers. Diagnostics output is
therefore not a throughput measurement of the optimized schedule.

Three fresh-process alternating diagnostics-off cycles at each completion
length compared schema-5 baseline binary
`9160ae1c597a6d3390126fbb88fbaba2094a630bd6963bd103e54670d7f695ed`
with candidate binary
`f7d8fa47fa5a2cc09405ed1a217ef1d1c406a410cfcd174890c67fb88e342d64`.
Every row preserved exact output:

| Completion target | Baseline median | Candidate median | Change | Output SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 64 tokens | 17.25 tok/s | 20.45 tok/s | +18.55% | `8e65e1b5...f61ddf` |
| 256 tokens | 18.20 tok/s | 21.02 tok/s | +15.50% | `62578360...9c341ff` |
| 512 tokens | 17.60 tok/s | 20.45 tok/s | +16.19% | `50286bf4...1b90d1` |

The 18 measured arms produced 381 valid resource samples. Maximum process-group
RSS was 1.794 GB, minimum available memory was 3.947 GB, and maximum wired
memory was 6.165 GB. No resource row had unavailable fields.

Relinking after a reversible command-count probe produced source-equivalent
candidate binary
`567ddc56ce1ceca35dac43103a8490b76f9e7804c36c01601fd6d548bb8c942a`.
A fresh 256-token check retained exact output and measured 17.81 versus 20.92
tok/s (+17.46%). The full G4 gate then passed against that exact binary:

- 4,002-token prompt: 186 output tokens at 16.67 decode tok/s.
- 15,362-token prompt: 64 output tokens at 12.03 decode tok/s.
- 50 of 50 sequential requests passed.
- Disconnect released admission in 0.536 seconds.
- 10 of 10 restart cycles passed with no orphan listener.
- All 548 resource samples were valid; peak RSS was 1.630 GB and minimum
  available memory was 3.954 GB.

The candidate clears the 15% gate at all three measured decode lengths, keeps
exact output and the existing cache behavior, and passes package and G4
validation. Cross-layer command chaining is accepted as the production decode
schedule. Rollback is the isolated `finishCurrentTokenChained` dispatch and its
encoder extractions; the schema-5 diagnostics path remains a working reference
schedule.

### Prompt-cache and process-cold assessment

Three fresh server processes ran a 45-token A -> A -> B -> A sequence with 64
generated tokens per request, 24-slot LFU, a 16,384-token context,
single-prefix caching, chunked prefill, and schema-4 diagnostics. A and B differ
inside the prompt, so B cannot continue A. All three cycles produced one stable
hash for A across cold, replay, and post-B recomputation, and one stable hash
for B.

| Request | Median TTFT | Cached tokens | Median decode |
| --- | ---: | ---: | ---: |
| First A in a fresh process | 8.3608 s | 0 | 15.93 tok/s |
| Immediate A replay | 0.0198 s | 45 | 18.55 tok/s |
| Different prompt B | 1.2465 s | 0 | 18.29 tok/s |
| A after B | 1.2431 s | 0 | 18.09 tok/s |

The A-after-B result is within 0.3% of B and confirms that B replaces the only
server entry and the runner's only private snapshot. Replay avoids about 1.23
seconds of warmed prefill for this short prompt, but a multi-prefix cache is not
a server-map-only change: snapshot identity and ownership must move into an
explicit runner API. Each retained Qwen snapshot plus logits consumes about
62.8 MiB at 45 tokens, 140.0 MiB at 4,002 tokens, 361.9 MiB at 15,362 tokens,
and 381.9 MiB at the 16,384-token limit. The three measured processes peaked at
2.127 GB RSS with at least 5.007 GB system memory available.

Do not implement multi-prefix retention without a representative trace showing
that interleaved active conversations lose enough reusable prefix work to clear
the end-to-end gate. Such a change would require bounded eviction, explicit
snapshot handles, restore/cancellation tests, and memory-pressure validation;
the current single-prefix cache remains the production choice.

Median process start-to-health was 1.6411 seconds. The first A prefill reported
7.309 seconds of expert-fetch time, versus 0.695 seconds for warmed B, accounting
for 6.614 seconds of the 7.114-second cold-to-warm TTFT gap. The server explicitly
uses full SHA-256 integrity, and first routed-layer access verifies each layer
before opening its streamer. This agrees with PF-11, where receipt-trusted load
avoided 6.741 seconds of hashing. Keep full SHA-256 as the production server
default: trusted-receipt mode is a distinct integrity-policy trade-off and its
warmer page-cache state must not be reported as a compute-speed improvement.

### Current disposition and recovery

The corrected fused-head implementation, Qwen head specialization, replay test,
and an older copy of this evidence were stashed on 2026-08-22 from branch
`perf/qwen-fused-greedy-head` with message
`qwen fused greedy head correctness and optimization evidence`. The five source
and test changes have since been integrated manually into the diagnostics-enabled
worktree, together with the 24-slot LFU production default. The stash remains an
independent recovery point, and no commit was made. Because stash indices can
move, locate the entry by message before restoring its files:

```bash
git stash list
git restore --source=stash@{N} -- \
  Sources/TurboFieldfare/Kernels/Fusions/LMHeadChainInt4.swift \
  Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift \
  Sources/TurboFieldfare/Runtime/Inference/ForwardRunnerFactory.swift \
  Sources/TurboFieldfare/Runtime/Inference/QwenForwardRunner.swift \
  Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests+Continuation.swift
```

The stash changes those five files plus an older copy of this plan. Applying the
whole stash is also possible, but it may conflict with the newer evidence in
this document; keep the current plan when resolving that conflict. Before the
stash was created, 766 tests in 141 suites passed, the release build passed, all
23 Markdown files passed link validation, source diagnostics and
`git diff --check` were clean, the server default remained logits-first, and no
model process or listener remained.

Useful patterns from
[`RealForwardRunner.swift`](../Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift)
include:

- Explicit pending-command records.
- Bounded pending depth.
- Expert and argument-buffer lifetime management.
- Cache-hit/miss phase splitting.
- Slot reservation until the consuming command completes.

Do not copy its prefill tile scheduler or assume Gemma's recurrent/KV
dependencies match Qwen. Do not pipeline multiple Qwen layers without
independent scratch buffers and explicit cache-slot lifetime protection.

Cancellation must not abandon a task that is still writing an expert-cache
slot. No later fetch may evict a reserved slot while a Metal command buffer
still references it.

Rollback point: disable overlap and return to exact sequential fetch after
router completion.

## Validation matrix

### Focused tests

- Diagnostic aggregation and schema stability.
- Decode-only accounting and first-token seeding.
- Command-buffer count before and after coalescing.
- Exact greedy token parity.
- Prompt continuation and prompt-state replay.
- Cancellation during GPU work and expert fetch.
- Cache-plan hit/miss consistency.
- Buffer lifetime under delayed command completion.

### Package validation

Run:

```bash
Scripts/test.sh
swift build -c release
```

### Live validation

Follow the repository model-process and memory-pressure checks before every
model run. Run only one model workload at a time. Report the commit, artifact
receipt, hardware, RAM, macOS, Swift version, exact command, exit code, timing
footer, resource validity, and every protocol deviation.

For each candidate, compare an interleaved baseline and candidate series. Do
not average invalid runs or compare different artifacts, hosts, prompts, output
lengths, or cache settings.

## Risks

| Risk | Required control |
| --- | --- |
| Metal uses a released expert or argument buffer | Explicit pending-command lifetime record |
| Cache slot is evicted while the GPU reads it | Reservation through command completion |
| Cancellation leaves a cache write active | Structured fetch task cleanup before reset |
| DeltaNet state advances out of order | One strictly ordered layer/token dependency chain |
| Full-attention KV position diverges | Position and continuation parity tests |
| Diagnostics change measured timing | Aggregate in memory; serialize after timing |
| Estimated bytes are reported as disk traffic | Label them logical estimated bytes |
| Lower command count does not improve wall time | Enforce the 15% end-to-end gate |

Any buffer-lifetime, cache-eviction, cancellation, recurrent-state, KV-state,
or token-parity failure is an immediate rollback, not a performance tradeoff.

## Deferred work

Do not implement these as part of the first decode optimization:

- Larger expert-cache defaults.
- Speculative expert prediction or prefetch.
- Expert-file layout reordering.
- RDADVISE default changes.
- Speculative decoding.
- Multiple Qwen layers in flight.
- Recurrent/KV state redesign.
- Gateway routing or default-model promotion.
- Gemma-level throughput promises.

Previous experiments found that several of these ideas improved local metrics
but failed broader runtime gates. Reopen one only with a new measured mechanism
and an interleaved same-host control.

## Execution checklist

- [x] Preserve and review the existing multi-chunk prefill diff.
- [x] Add versioned aggregate Qwen diagnostics and unit tests.
- [x] Attach decode-only aggregate diagnostics to `RawDecodeResult`.
- [x] Add opt-in CLI JSON output.
- [x] Add opt-in server and harness export after the CLI schema stabilizes.
- [x] Establish a valid interleaved baseline.
- [x] Verify the current 122 forward-runner command buffers per Qwen step.
- [x] Account for sampling and residual token-loop time.
- [x] Require at least 90% wall-time attribution.
- [x] Implement the decode command-buffer coalescing candidate.
- [x] Run parity, continuation, cancellation, package, and resource checks.
- [x] Enforce the 15% same-host median improvement gate; reject the candidate
  below threshold.
- [x] Profile the rejected candidate again.
- [x] Implement and measure exact-demand I/O overlap; reject it below the 15%
  gate and restore the merged Phase 3 runtime.
- [x] Accept the DeltaNet recurrent-kernel candidate after all correctness and
  resource gates pass.
- [x] Fix fused greedy-head token handoff and cached replay; verify exact parity.
- [x] Measure the corrected fused head and Qwen-specific head specialization.
- [x] Measure and reject four below-gate post-head decode follow-ups.
- [x] Add opt-in expert-read count, summed-time, and maximum-latency diagnostics;
  measure one controlled 256-token row and reject the concentrated-tail I/O
  hypothesis.
- [x] Add opt-in route-and-weight tracing; replay memory-neutral replacement and
  allocation candidates and quantify the Belady upper bound.
- [x] Integrate the correctness-proven fused Qwen head and promote 24-slot LFU
  after their valid combined comparison gained 6.70% with exact output parity.
- [x] Add non-overlapping GPU timing inside the combined mixer/shared/router
  command buffer before selecting another optimization candidate.
- [x] Measure A -> A -> B -> A prompt-cache behavior and defer multi-prefix
  retention pending a representative interleaved workload.
- [x] Separate process readiness from first-request expert verification and
  retain full SHA-256 as the production integrity policy.