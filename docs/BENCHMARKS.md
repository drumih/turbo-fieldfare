# Benchmarks

This page records TurboFieldfare measurements on an 8 GB M2 MacBook Air and a
24 GB M5 Pro. Each number belongs to the workload shown. Prompt length,
generated length, cache state, and hardware all change throughput, so ranges
across workloads are not run-to-run variation.

Each table states its workload and decoding settings. TurboFieldfare uses the
model installed by the [command-line instructions](../README.md#command-line-interface).
Decode rate excludes model installation, model loading, and prompt prefill.

## Results at a glance

| Host and runtime | Decode rate | Reported memory |
| --- | ---: | ---: |
| 8 GB M2 MacBook Air, TurboFieldfare, prefetch on | 5.5–6.8 tok/s | 2,078–2,092 MiB footprint |
| 24 GB M5 Pro, TurboFieldfare, prefetch off | 35.3–39.2 tok/s | 2,163–2,168 MiB footprint |
| 24 GB M5 Pro, [mlx-lm](#same-host-mlx-comparison) | 76.3–82.1 tok/s | 8.3–9.8 GB RSS; 14.7–15.3 GB GPU allocation |

Prefetch defaults off. TurboFieldfare uses the full-generation workloads below;
MLX uses the separate [comparison workloads](#same-host-mlx-comparison).

## Full-generation measurements

Measured on 2026-11-13 using the [community prompts](COMMUNITY_BENCHMARKS.md).
M2 has one measured run per case after warmup. M5 shows both off-mode runs
from the paired comparison, in run order.

| Host / case | Prefetch | Prompt / generated tokens | Decode tok/s |
| --- | --- | ---: | ---: |
| M2 / short | On | 61 / 527 | 6.8 |
| M2 / medium | On | 430 / 665 | 6.2 |
| M2 / long | On | 3,015 / 597 | 5.5 |
| M5 Pro / short | Off | 61 / 497 | 39.1 / 39.2 |
| M5 Pro / medium | Off | 430 / 721 | 37.5 / 37.4 |
| M5 Pro / long | Off | 3,015 / 610 | 35.4 / 35.3 |

All ended at `endOfTurn`; outputs matched within each case. Different output
lengths across hosts prevent a controlled cross-host ratio.

<details>
<summary>Measurement settings and source revisions</summary>

16 LFU slots, 128-token chunked prefill, RDADVISE off, context 4096, max-new
1024, temperature 0.2, Top-K 64, Top-P 0.95; short/medium/long seeds
20260721/20260722/20260723. Enabled prefetch uses one stage. No profiling.
M2: macOS 26.5.2, Swift 6.3.3, revision `d85c1ee2`.
M5 Pro: macOS 27.0, Swift 6.4.2, revision `18b68ce`.
These measurements predate 0.10.0. Peak footprint comes from `/usr/bin/time -l`;
all listed runs recorded zero host swapout delta. Cache state, clocks and
background work were not controlled.

</details>

## Speculative expert prefetch

Same-binary off/router comparisons used two measured pairs per workload:

| Host | Short explanation | Medium review | Long synthesis |
| --- | ---: | ---: | ---: |
| M2 MacBook Air, 8 GB (2026-11-12) | +12.6–14.3% | +11.2–12.7% | +12.3–13.1% |
| M5 Pro, 24 GB (2026-11-13) | −4.1% to −3.4% | −5.1% to −4.4% | −12.5% to −9.9% |

Positive means faster decode with prefetch. Output bytes, counts and stop
reasons matched within each case. The M2 pairs predate its enabled-only rows
above. Results apply to these hosts and workloads; use the
[comparison protocol](COMMUNITY_BENCHMARKS.md#optional-speculative-prefetch-comparison)
to test another system.

## M2 measured decode

These rows ran on a `Mac14,15` M2 MacBook Air with 8 GB of memory. No
experiment, profiler, or trace mode was active.

| Prompt / generated tokens | Prefill | TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: | ---: |
| 6 / 32 | 7,025 ms | 7,979 ms | 6.30 tok/s | 1,304 / 1,791 MiB |
| 121 / 64 | 7,934 ms | 8,862 ms | 5.10 tok/s | 1,528 / 1,776 MiB |
| 527 / 64 | 21,736 ms | 22,649 ms | 5.90 tok/s | 1,535 / 1,886 MiB |
| 1,017 / 128 | 36,729 ms | 37,656 ms | 5.38 tok/s | 1,455 / 1,971 MiB |

Each workload ran once in a fresh process. The file cache was warm but
uncontrolled, and every row produced the same token IDs as its validation
control. These four points show the production path running under the 8 GB
rule; they do not form a confidence interval or describe sustained long
generation.

### Where the short M2 row spent its time

A separate diagnostic pass on the six-token prompt divided a 162.8 ms decode
step into four broad parts:

| Work | ms/token |
| --- | ---: |
| Expert reads | 83.1 |
| Waiting in the command-buffer pipeline | 55.6 |
| Tied output head | 14.2 |
| Other runtime work | 9.9 |

The diagnostic instrumentation disabled the normal command-buffer pipeline
and reduced throughput to 4.23 tok/s. The breakdown explains where that run
spent time; it does not describe independent speedups or a performance bound.

## M5 measured decode

These rows ran on 2026-07-20 on a 24 GB M5 Pro (`Mac17,8`) with macOS 26.5.1,
Xcode 26.6, and Swift 6.3.3. No profiler or trace mode was active.

The benchmark uses chat-framed prompts and fixed, non-repeating natural
continuations. This keeps the generated text and expert-routing workload stable
without rewarding a model repetition loop. The complete production sampling
and decode path still runs for every token.

One warmup preceded three fresh-process measurements per workload. The table
reports medians; the file cache was warm but uncontrolled. A separate
free-generation smoke reached the end of each model turn without a repetition
loop.

| Prompt / generated tokens | Prefill / TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: |
| 61 / 256 | 5,096 / 5,668 ms | 35.17 tok/s | 1,834 / 2,126 MiB |
| 430 / 256 | 6,762 / 7,325 ms | 34.72 tok/s | 1,851 / 2,142 MiB |
| 3,015 / 256 | 23,038 / 23,610 ms | 31.01 tok/s | 1,835 / 2,126 MiB |

## Same-host MLX comparison

The same M5 Pro ran MLX 0.32.0 and mlx-lm 0.31.3 against the same checkpoint,
prompt-token IDs, and generated-token counts. MLX measured 82.07, 80.25, and
76.33 tok/s for the 121-, 527-, and 1,017-token prompts.

Treat this as throughput context, not a complete engine comparison:

- The engines ran in separate blocks rather than a balanced, interleaved order.
- Their first-token clocks started at different points, so TTFT is not comparable.
- Generated IDs matched for the shortest prompt but diverged for the two longer prompts.
- TurboFieldfare recorded a 1.89-2.09 GiB physical footprint. MLX reported
  14.66-15.31 GB of peak GPU allocation and 8.27-9.79 GB of peak process RSS.
  Those counters measure different things and should not be compared as a
  direct memory ratio.

The MLX process required the larger host and is not an 8 GB TurboFieldfare
deployment path.

## Reproduce and contribute a result

The [community benchmark guide](COMMUNITY_BENCHMARKS.md) uses short, medium,
and long chat-framed prompts with fixed seeds. It requires coherent output and
a normal end of turn, so a repetition loop cannot become a published speed
result. The public CLI's timing footer reports decode-only throughput without a
separate research harness.

The full-generation and community runs generate their own output. Only the
July M5 table uses fixed continuations. Matching prompt and generated token
counts are necessary, but not sufficient, for a controlled comparison.

A current checkout may not reproduce a historical number after the runtime,
compiler, or operating system changes. Report the commit and all three rows
rather than presenting one run as a general hardware result.

Read [System design](SYSTEM_DESIGN.md) for the runtime and resource split,
[Experiments](OPTIMIZATION_JOURNEY.md) for the main wins and failures, and the
[measurement lessons](experiments/summaries/09-validation-and-measurement-lessons.md)
for the rules used to evaluate performance changes.
