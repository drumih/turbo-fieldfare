# Handover — Inkling2 Q2_K/Q4_K GGUF port to Swift/Metal

**Prompt for the next agent:**

> Continue the native Swift/Metal port of llama.cpp's Inkling2 `UD-Q2_K_XL` and
> `UD-Q4_K_XL` GGUF execution into TinyTitan. The goal is parity first and then
> **at least 25% faster than the llama.cpp reference** on this 24 GB M3:
> Q2_K target **≥17 tok/s**, Q4_K target **≥14 tok/s**. Work only from measured
> microbenchmarks and bit-exact/quality checks. The current tree has working
> Q2_K/Q4_K block decode and fused GEMV kernels, but **no Inkling2 GGUF
> architecture path and no end-to-end model run**. Do not claim the speed target;
> it is not reached.

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
- Native target: **Q2_K ≥17 tok/s, Q4_K ≥14 tok/s**.

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

## What is missing

There is **no Inkling2 GGUF architecture path**. TinyTitan's current GGUF path
is hardcoded to the Qwen3.8 architecture in
`sources/TinyTitanCLI/RunGGUF.swift` and `GGUFDecoder`; it cannot load these
Inkling2 GGUFs. The Q2_K/Q4_K code above is a kernel/format foundation, not a
model implementation.

Still required:

1. **Kernel geometry is ported** (above). Whether the GEMVs fit the 17 tok/s
   budget can only be judged per token once the model runs; the expert
   projections are not 2304 × 2304. Further kernel work should be driven by
   an end-to-end per-phase profile, with paired A/B as before. Reference:
   `~/.unsloth/llama.cpp/ggml/src/ggml-metal/kernels/mul_mv.metal`.
2. **Inkling2 GGUF loader/architecture** from
   `~/.unsloth/llama.cpp/src/models/inkling2.cpp`, `conversion/inkling2.py`,
   and `models/templates/Inkling.jinja`. Needs tensor mapping, loop core
   (6–17 repeated three times), controllers/message bank, relative attention,
   short-convolution state, sliding/full attention pattern, MoE router and
   experts.
3. **Graph/command-buffer executor.** The existing native Inkling engine is
   eager and slow; llama.cpp wins through composed `ggml` graphs and fused
   Metal kernels. Encode a layer/ubatch before waiting; do not dispatch one
   projection at a time.
4. **Expert cache/prefetch and KV/conv state** using the lessons from
   TinyTitan's Qwen engine.
5. **End-to-end quality gate and paired benchmark** against llama.cpp.

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
