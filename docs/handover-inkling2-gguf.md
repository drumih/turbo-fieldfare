# Handover — native Inkling2 GGUF (`UD-Q2_K_XL`, `UD-Q4_K_XL`) on Swift/Metal

**Prompt for the next agent:**

> Continue the native Swift/Metal execution of Unsloth's Inkling-2-Mini
> `UD-Q2_K_XL` and `UD-Q4_K_XL` GGUFs in TinyTitan (wiki tracker TT-113). The
> goal is greedy parity with llama.cpp first, then **at least 25% faster than
> llama.cpp** on this 24 GB M3: `UD-Q2_K_XL` **≥17 tok/s**, `UD-Q4_K_XL`
> **≥14 tok/s**. **The names are misleading: neither file is mostly Q2_K or
> Q4_K** (table below) — kernels are chosen per tensor. Use every engine on the
> chip where it measurably pays: GPU for the token graph, CPU/AMX for small
> f32 ops and expert prefetch, ANE for prefill. Work only from measured,
> paired results and parity checks. There is still **no end-to-end GGUF run**;
> do not claim the target.

## What the two files actually contain (measured 2026-11-21)

Read with llama.cpp's `gguf-py` and pinned by
`Inkling2GGUFManifestTests` (`TINYTITAN_INKLING2_GGUF=<file>`):

| Tensor | `UD-Q2_K_XL` (9.14 GB) | `UD-Q4_K_XL` (17.63 GB) |
| --- | --- | --- |
| `ffn_gate_up_exps` | IQ1_S (0–4, 6), IQ2_XXS (5, 7, 12–14), IQ2_XS (8, 10), IQ2_S (rest) | NVFP4 (0–19), Q8_0 (20–23) |
| `ffn_down_exps` | Q2_0 (0–4, 6, 12–14, 21–23), NVFP4 (5, 7–11, 15–19), MXFP4 (20) | NVFP4 (0–19), Q8_0 (20–23) |
| `attn_q/k/v/output/gate/rel` | IQ3_XXS, Q4_K, Q5_K, Q6_K, Q8_0 (mixed per layer) | Q8_0 |
| `token_embd` / `output` | IQ4_XS / Q4_K | Q8_0 / Q8_0 |
| small tensors | F32 (norms, convs, router, loop controllers), BF16 (sink, conv proj) | same |
| `ffn_*_exps.scale` | F32 per-expert multiplier, layers 0–19 | same |

By bytes, Q2_K_XL is IQ2_S 2.39 GB, NVFP4 2.10, Q2_0 1.15, IQ2_XXS 0.88,
IQ1_S 0.80, IQ2_XS 0.39, Q8_0 0.39, **Q4_K 0.30**, IQ4_XS 0.25, MXFP4 0.18.
Q4_K_XL is NVFP4 11.47 GB and Q8_0 6.0 GB, **no Q4_K at all**. GGUF NVFP4 is
ggml's `block_nvfp4` (64 values: 4 UE4M3 sub-block scales + 32 bytes), not the
safetensors layout the native NVFP4 kernel reads.

Geometry: 24 stored layers, `block_count` 48 unrolled (core 6..<18 × 3),
full attention at 5/11/17/23 (sliding window 1024 elsewhere), 24 q / 8 kv
heads of 128, 128 experts top-8 with expert FF 1152, hidden 2304, vocabulary
200,058 (padded 201,024), logit softcap 30.

## Goal and reference

- Repository: `/Users/andreborchert/Downloads/TinyTitan`, branch `main`.
- Hardware: Apple Silicon M3, 24 GB unified memory, macOS with Metal 4.
- Reference engine: llama.cpp checkout at `~/.unsloth/llama.cpp`,
  binaries `~/.unsloth/llama.cpp/build/bin/llama-bench` and `llama-server`.
- Models (already downloaded, do not re-download):
  - `~/.cache/huggingface/hub/models--unsloth--Inkling-2-Mini-GGUF/snapshots/*/Inkling-2-Mini-UD-Q2_K_XL.gguf`
  - `.../Inkling-2-Mini-UD-Q4_K_XL.gguf`
- llama.cpp measurements on this machine:
  - Q2_K: stable **~12.8–13.8 decode tok/s**; `pp64` ~71–80 tok/s.
  - Q4_K: **~10.1–14.2 decode tok/s**, memory-sensitive; high variance.
  - Best usual Q2 settings: `-ngl 99 -t 8 -fa on -b 512 -ub 256`, f16 KV.
  - Q4: all layers on GPU, `-fa on`; `-ncmoe 4/8` is slower.
  - Sampling recommendation: temperature 1.0, top-p 1.0, top-k 0, min-p 0.0.
    Sampling does not change benchmark throughput.
- Native target: **`UD-Q2_K_XL` ≥17 tok/s, `UD-Q4_K_XL` ≥14 tok/s**.

## What is complete

1. **Q2_K/Q4_K host block decoding** in
   `sources/TinyTitan/Infrastructure/ModelIO/GGUFDequant.swift`.
   - Bit-exact against ggml's `dequantize_row_q2_K` / `dequantize_row_q4_K`.
   - Fixture: `tests/TinyTitan/Infrastructure/ModelIO/GGUFDequantVectors.txt`.
   - Generator: `tools/gguf-ref-vectors.c`.
   - Block layouts:
     - Q2_K, 84 bytes: `scales[0..15]`, `qs[16..79]`, `d` at 80, `dmin` at 82.
     - Q4_K, 144 bytes: `d` at 0, `dmin` at 2, `scales[4..15]`, `qs[16..143]`.
   - Host throughput: Q2_K 1.42 ms/4096 blocks (0.242 GB/s),
     Q4_K 1.23 ms/4096 blocks (0.478 GB/s).

2. **Fused Q2_K/Q4_K Metal GEMV**:
   - Shader: `sources/TinyTitanCore/Metal/Quant/gguf_q2k_q4k_gemv.metal`.
   - Swift wrapper: `sources/TinyTitan/Kernels/Quant/GGUFKQuantGEMV.swift`.
   - Registered in `MetalContext.shaderModules` as `gguf_q2k_q4k_gemv`.
   - Correctness test + microbenchmark:
     `tests/TinyTitan/Kernels/GGUFKQuantGEMVTests.swift`.
   - One SIMD group per row, 8 rows per threadgroup.
   - Prepared-buffer path (`prepare` + `gemv(prepared:)`) exists because the
     convenience path's per-call allocation/upload dominated the measurement.

3. **Kernel optimizations that survived measurement**:
   - Packed 64-bit quant loads (`ulong`) instead of 8 separate byte loads.
   - Q2_K two-accumulator block loop.
   - **Row-tiled geometry (llama.cpp's `kernel_mul_mv_q{2,4}_K_f32` shape)**
     in `gguf_q2k_q4k_tiled.metal` is the kernel of record
     (`GGUFKQuantGEMV.Kernel.rowTiled`, the default). Paired
     `pairedGeometryBenchmark`, 2304 × 2304, GPU time per projection
     (median of 7 rotated rounds × 200 dispatches in one command buffer),
     three runs on the 24 GB M3, 2026-11-21:
     | Kernel | Q2_K | Q4_K |
     | --- | --- | --- |
     | `rowPerSIMDGroup` (baseline) | 0.0447–0.0494 ms | 0.0510–0.0557 ms |
     | `rowTiled` | **0.0300–0.0332 ms** | **0.0333–0.0370 ms** (up to 89.6 GB/s) |
     These are **GPU-only** times; the older ~0.34 ms figures were wall clock
     including commit/wait per projection and are not comparable. The one-row
     kernel is kept only as the A/B baseline.
   - **Rejected by measurement:** two rows per SIMD group, threadgroup-shared
     `x`, vectorized `x` loads, `#pragma unroll 2`, the original one-row
     kernel without packed loads, and eight lanes per row / four rows per SIMD
     group (`gguf_q2k_q4k_multi.metal`, 0.0430 / 0.0494 ms — deleted).

4. **Commits on `main`**:
   - `173838e7` — Q2_K/Q4_K decode
   - `e996339d` — decode throughput
   - `1eb3ce71` — fused GEMV
   - `45ac7937` — kernel-only microbenchmark
   - `b32ffab3` — packed 64-bit q loads
   - `bcf31e0d` — two-accumulator Q2_K loop
   - `b0866b08` — tiled Q2_K/Q4_K GEMV geometry
   - the commit after it — eight-lane kernel deleted, tiled made the default

5. **GGUF reader fix (2026-11-21).** `GGUFType.nvfp4.blockSize` was 1
   instead of ggml's 64, so every NVFP4 tensor was sized 64× too large and
   both Inkling GGUFs were refused as running past the end of the file.
   `GGUFTypeSizeTests` now pins every type's (elements, bytes) to ggml's
   `GGML_QUANT_SIZES`; NVFP4 was the only wrong row.

## What is missing — the plan

`Inkling2GGUFManifest` (`sources/TinyTitan/Infrastructure/ModelIO/`) now reads
and validates both files. Everything after that is open:

1. **Per-type GEMV coverage.** Have: Q8_0, Q6_K, IQ2_S, IQ3_XXS, IQ4_XS, Q2_0
   (`gguf_gemv_vec.metal`, Qwen3.8 route), Q2_K/Q4_K (tiled). Missing:
   GGUF-layout NVFP4, MXFP4, IQ1_S, IQ2_XXS, IQ2_XS, Q5_K. Each lands with a
   host decode bit-exact against ggml and a paired microbenchmark.
   **Start with `UD-Q4_K_XL`**: it needs only NVFP4 + Q8_0.
2. **GPU-resident token graph.** The native safetensors engine
   (`sources/Engines/Inkling2`) has the right math and passes 6/6 factual
   prompts but decodes 1.7–1.9 tok/s because it is eager and CPU-bound. The
   GGUF engine encodes the whole token — 48 layer visits, depth mixes, loop
   controllers, short convolutions, relative attention with sinks, router —
   into one command buffer, with the router's top-8 written to a GPU buffer
   that indirect expert GEMVs read (no CPU round trip). Port the math from
   the native engine and `~/.unsloth/llama.cpp/src/models/inkling2.cpp`.
3. **Residency and prefetch.** Q2_K_XL (9.1 GB) fits resident. Q4_K_XL
   (17.6 GB) leaves little headroom on 24 GB: map experts (`GGUFMappedShards`)
   and prefetch the next layer's routed experts on CPU threads while the GPU
   runs the current one; measure hit rates before tuning.
4. **CPU/AMX and ANE.** Measure, don't assume: small f32 ops (loop
   controllers, router) may be cheaper on AMX while the GPU streams experts;
   prefill on the ANE only if it beats batched GPU GEMM.
5. **Parity, then the paired benchmark.** Greedy tokens and top-k logits
   against `llama-server` on the same prompts, then a paired tok/s run against
   `llama-bench` under identical machine state.

## Microbenchmark procedure

- Build release once; do not benchmark debug.
- Use prepared Metal buffers. Measuring `gemv(weights:)` includes a 1.7 MB
  upload and allocation and is not a kernel benchmark.
- Run at least three interleaved A/B rounds. The last recorded full test:

  ```bash
  swift test -c release --filter GGUFKQuantGEMVTests
  ```

- Paired A/B: `TINYTITAN_KERNEL_BENCH=1 swift test -c release --no-parallel
  --filter GGUFKQuantGEMVTests/pairedGeometryBenchmark`; output lines are
  `Q2_K rowTiled 2304x2304: median ... ms (min ..., max ...) ... GB/s`.
- Validate every kernel change against `matchesDequantizedDotProduct` before
  recording speed.
- Record commit, machine state, and the run-to-run spread. Do not publish a
  single best run as the result.

## Metal traps already hit

- `half`, `simd`, `group`, `block`, `blocks` are problematic identifiers in
  MSL here. Rename to `halfBlock`, `sg`, `groupIndex`, `blockIndex`,
  `blockCount`.
- Thread attributes `simdgroup_index_in_threadgroup` and
  `thread_index_in_simdgroup` are **scalar `uint`**, not `uint3`; keep
  `threadgroup_position_in_grid` scalar too.
- `as_type<half>` requires a `ushort`; wrap bitwise expressions in `ushort(...)`.
- Misaligned `uchar4`/`ulong` loads or wrong quant mapping silently produce
  plausible but wrong output; keep the correctness test.
- Q2_K layout is not `d` first. Getting it wrong produced the initial fixture
  mismatch; the reference C and test vector are authoritative.

## Useful commands

```bash
cd /Users/andreborchert/Downloads/TinyTitan
swift build -c release --product TinyTitanCLI
swift test -c release --filter GGUFKQuantGEMVTests
swift test -c release --filter GGUFDequantTests

# llama.cpp references
llama-bench -m "$Q2" -ngl 99 -t 8 -fa on -b 512 -ub 256 -p 64 -n 64 -r 5
```

Do not claim the 25% target until there is a native end-to-end Inkling2 GGUF
run at ≥17 tok/s Q2_K and ≥14 tok/s Q4_K, with quality checks. The current
number is a kernel microbenchmark, not a model tok/s.
