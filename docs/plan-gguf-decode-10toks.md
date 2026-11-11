# Plan: the as-is Unsloth GGUF at 10 tok/s decode under `--ram-budget 10G`

Owner request (2026-11-11): 10 tok/s decode with the Unsloth
Qwen3.8-Flash-Next UD-IQ3_XXS GGUF, run as is. Plan first, measure after,
code only once confident, then verify. Quality bar unchanged: greedy text
identical and `GGUFLogitProbeTests` (top-1/top-5 against llama.cpp) passing
for every step. A step that cannot keep that bar is rejected and recorded.

## Where a token goes today (24 GB M3, 10 GB budget, warm, 200 tokens)

| part | ms/token | notes |
| --- | ---: | --- |
| GPU, 48 layers (mixer, router, shared, routed) | 75-90 | mixer kernels at ~55-61 GB/s; routed ~50 GB/s |
| expert fetch exposed to the CPU | 90-105 | 112 misses (220 MB); `F_RDADVISE` of a top-10 guess cut it ~20% |
| output head + embedding + logits | ~12 | head 6 ms of GPU |
| waits / encode overhead | ~20 | 48 waits, ~0.4 ms each |
| **total** | **220-260** | **3.9-4.5 tok/s** |

10 tok/s is 100 ms a token. **What has to be true:**
- GPU at or below 65 ms.
- I/O almost fully hidden: at most 15 ms exposed.
- Overhead at or below 15 ms.

## The floor, so the target is honest

A token's GPU work reads every non-expert weight once (attention, GDN and
hyper-connection matrices about 2.9 GB, head 0.5 GB) plus the ten experts of
every layer (0.95 GB): about 4.35 GB. The M3's unified memory peaks near
100 GB/s and well-tuned GEMVs reach about 80 GB/s, so the GPU floor is
**about 45-55 ms**. 10 tok/s therefore leaves 45-55 ms for everything else
and needs the reads entirely off the critical path. That is the engine's
own state at 12 GB (`io_hidden` 99.7%, `docs/qwen38-decode-12gb-measurements.md`).
It is ambitious but not physically excluded. If the GPU floor measures
higher than 55 ms, the target becomes about 1000 / (GPU + 15) and this
document records that.

## Measured ceilings (2026-11-11)

- **Drive.** Random 1 MiB `pread`s from the 50 GB shard reach 3.62 / 3.77 /
  3.91 / 3.96 / 3.93 GB/s at 4 / 8 / 16 / 32 / 64 threads. At 10 tok/s that
  is about 390 MB a token in total. The true misses alone are 220 MB (76.7%
  hits at 57 slots), so the misses have to be read in full overlap with the
  GPU, and wrong guesses fit only in the remaining ~170 MB.
- **Advice width.** The drive sets this budget, and the measurement shows
  it. Two interleaved rounds, 200 tokens:

  | guesses advised per layer | fetch per token | decode (tok/s) |
  | --- | --- | --- |
  | 6 | 109 / 105 ms | 3.66 / 3.79 |
  | 8 | 100 / 101 ms | 3.84 / 3.84 |
  | 10 (default) | 91 / 91 ms, then 102 / 96 | 4.45 / 4.32, then 3.68 / 3.87 |
  | 16 | 115 / 120 ms | 3.34 / 3.50 |
  | 24 | 212 / 265 ms | 2.23 / 2.13 |

  Wider guesses saturate the drive and lose. The levers that remain on the
  I/O side are guesses that are *more precise* (recall at a fixed width),
  not more of them, plus a higher hit rate (more of the 10 GB for the
  cache).
- **Consequence.** Even with perfect overlap the drive needs about 56 ms a
  token for the true misses at today's hit rate. 10 tok/s requires the
  GPU's 48 layers and the miss reads to overlap almost completely. That
  pipelines the routed half of a layer against the next layer's reads,
  which is the engine's 99.7% `io_hidden` state.

## Steps (each: measure first, then code, then verify; one commit each)

### 1. I/O off the critical path (target: exposed fetch at or below 15 ms)

1. **Deeper, earlier advice.** Advise the next layer's guesses as soon as
   its router logits exist, which is already done. Also advise the layer
   after it, using router(L+2) on layer L's input; that costs one more
   512x2560 F32 GEMV. Raise the guess count: a wrong advice costs only idle
   drive time, never a cache slot.
   - **Measure first:** replay the routing logs with
     `.build/routing/policy.py`-style code for guess recall at top-10/16/20,
     one and two layers ahead.
2. **Read misses into the pool without blocking the GPU.** Hit experts are
   already submitted before the miss reads finish. Next, submit the
   following layer's mixer before waiting on this layer's misses? This
   stops at the dependency: layer L+1 needs layer L's routed output. So the
   remaining lever is making misses rare (step 1) and cheap (page-cache
   copies, about 0.2 ms per 2 MB on several threads).
3. **Verify:** add `io_exposed_ms` and `advise_recall` to the trace.

### 2. GPU at or below 65 ms

1. **Profile per kernel family on real weights.**
   `GGUFMixerStageBenchmarkTests` and `GGUFRoutedKernelBenchmarkTests`
   already exist. Extend them with GB/s per kernel.
2. **Q6_K GEMV** (attention and GDN projections, about 2/3 of mixer bytes):
   - two rows per SIMD group sharing the `x` loads;
   - `x` staged once per threadgroup in threadgroup memory;
   - 16-byte weight loads.
   - Target: 75-80 GB/s, from about 55-61.
3. **Fewer dispatches per layer.** Today a layer encodes about 25 compute
   encoders.
   - Fuse the GDN small ops: conv, qk-norm and gate.
   - Run the hyper-connection read's RMS norm and down-projection as one
     kernel.
   - Use one compute encoder per command buffer with
     `memoryBarrier(scope: .buffers)` between dependent dispatches, instead
     of a new encoder each.
4. **Routed IQ kernels.** IQ2_S and IQ3_XXS read their sign and grid
   tables from device memory per lane; put the tables in `constant` memory
   and take two rows per SIMD group.

### 3. Overhead at or below 15 ms

1. Encode the next layer's mixer into its command buffer while the current
   one runs (double buffering on the CPU), so a wait never waits on encode.
2. Read the logits only for the sampler's needs: the argmax or top-k, on the
   GPU at temperature 0 and with small k. This avoids the 248k-element
   conversion.

## Verification protocol (every step)

- Two interleaved rounds against the previous commit, 200 tokens, warm
  window reported (`.build/m1/window.py`-style; to be promoted into
  `benchmark/`).
- Greedy text prefix-identical. `GGUFLogitProbeTests`,
  `GGUFBatchedPrefillTests`, `GGUFDecoderDeterminismTests` and the full
  suite passing. Lint clean.
- The machine's own noise is recorded: other projects' builds have moved a
  run by up to 25% on this host. A step is kept only on a paired win in
  both rounds.

## Status

| step | state |
| --- | --- |
| 1.1 advise next layer, top-10, concurrent issue | **landed**: fetch 115-124 -> 90-92 ms, tok/s 4.13 -> 4.22 and 3.35 -> 3.94 |
| 1.1 wider advice (16, 24) | **rejected**: saturates the drive (table above) |
| 1.1 two-layers-ahead advice | planned; only with a precision measurement first |
| 2.x GPU kernels | planned |
| 3.x overhead | planned |
