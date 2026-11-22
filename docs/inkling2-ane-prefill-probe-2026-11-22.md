# Inkling2 GGUF — ANE prefill probe (2026-11-22)

**Question.** Does the Apple Neural Engine speed up Inkling-2-Mini's prefill,
given that each model reacts differently (the Ornith ANE sidecar lost token
identity to fp16 accumulation, TT-099)?

**What can run there.** The ANE executes dense fp16 Core ML graphs. Inkling's
routed experts (NVFP4, top-8 of 128) are sparse and stay on the GPU. Its dense
per-visit projections can move: Q, K, V, gate and rel share one input
(8,576 × 2,304) and the attention output is 2,304 × 3,072. Per prefill token
these are 2.58 GFLOP over the 48 visits, about 30% of the token's FLOPs
(the routed experts are about 6.1 GFLOP).

**Method.** `benchmark/inkling2_ane_prefill_probe.py` reads one layer's real
`UD-Q4_K_XL` weights (Q8_0), dequantised by llama.cpp's own `gguf-py`, builds
them as fp16 Core ML programs at fixed chunk sizes, and runs the *same*
compiled model with `CPU_AND_NE`, `CPU_AND_GPU` and `CPU_ONLY`: median of 20–30
predictions after warm-up, relative error against an fp32 NumPy reference on
RMS-scaled inputs, and the compute plan's ANE placement. No TinyTitan model
process runs; M3 24 GB, coremltools 9.0, quiet machine.

## Results

Records: `benchmark/ane-prefill/inkling2-ane-probe-q4kxl-layer1.json` (sliding
layer) and `…-layer11.json` (full-attention layer). The compute plan put
**100%** of the operations on the ANE in every case.

| Stage | T | ANE | Core ML GPU | CPU | ANE / GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q/K/V/gate/rel (layer 1) | 256 | 1.50 ms (6.8 TF/s) | 5.04 ms (2.0 TF/s) | 4.48 ms | 3.4× |
| Q/K/V/gate/rel (layer 1) | 512 | 2.83 ms (7.2 TF/s) | 8.09 ms (2.5 TF/s) | 7.43 ms | 2.9× |
| Q/K/V/gate/rel (layer 11) | 256 | 1.21 ms (8.4 TF/s) | 4.32 ms (2.3 TF/s) | 4.04 ms | 3.6× |
| attn_output (layer 1) | 256 | 0.59 ms (6.1 TF/s) | 2.76 ms (1.3 TF/s) | 1.53 ms | 4.7× |
| attn_output (layer 11) | 512 | 1.49 ms (4.9 TF/s) | 3.24 ms (2.2 TF/s) | 2.70 ms | 2.2× |

Relative error against fp32: **ANE 3.6e-4 / 3.8e-4**, GPU fp16 2.1e-4, CPU
5e-3. Unlike Ornith's fused attention tail, these are plain GEMMs with no
cancellation-prone epilogue, and the ANE stays at fp16 rounding level.

## Verdict

**Go for the dense projection stage — but only inside a batched prefill.**
The ANE runs these GEMMs at 6–8 TF/s, about 3× the GPU through Core ML, at
fp16-level error. Today the native decoder prefills token by token (≈57 ms a
token, the decode path), so the prerequisite is a batched GPU prefill —
attention over a chunk and batched routed-expert GEMMs. The ANE then takes the
dense ~30% of the FLOPs (≈0.4 ms/token at 6.5 TF/s) concurrently with the GPU
experts.

Open before wiring it:

1. **Weight format.** fp16 copies of the 24 stored layers' dense projections
   are 1.29 GB on a 24 GB machine already holding the 17.6 GB file. Core ML's
   block-wise int8 quantisation can represent Q8_0 (block 32, one scale) without
   further rounding at about half that; probe it before choosing.
2. **End-to-end parity.** The probe measures one GEMM; the bar stays greedy
   parity with llama.cpp on the reference prompts and logits within the noise
   floor, as for every other change.
3. **Paired prefill benchmark** against the batched GPU-only prefill and
   llama.cpp's `pp`.
