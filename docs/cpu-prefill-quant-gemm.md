# CPU prefill, phase 0: does the quantized path survive batching?

`cpu-prefill-gemm-benchmark.md` measured FP32 GEMM shapes and answered "is the
matrix engine worth using?". It left the question the engine actually faces
open: the CPU engine reads 4/8-bit affine weights one token at a time through
`tinytitan_int{4,8}_affine_gemv`, so a batched path has to get those weights
into a GEMM without paying more for the conversion than the batching saves.
This is that measurement.

Measured 2026-10-24 on the M3 / 24 GB host, with the shipping GEMV kernels
(compiled into the harness, not re-implemented) and the public Accelerate
BLAS. Harness: `benchmark/cpu_prefill_quant_gemm.c`.

## Method

Two arms on the same weights and activations, for both shipped widths:

- **A, today:** the shipping `tinytitan_int8_affine_gemv` /
  `tinytitan_int4_affine_gemv` called once per prompt token, rows split across
  the performance cores exactly as `Int8AffineGEMV.threaded` splits them.
- **B, batched:** dequantize the weight matrix to FP32 once (same `q*scale +
  bias` factoring), then one `cblas_sgemm` over all M tokens. The dequantization
  is charged to arm B's time; it runs once per prompt, so the M=512 figures
  below overstate its cost for a 2K/8K prompt rather than hiding it.

Shapes are the four projections of the earlier document (QKV, output, gate+up,
down) for one layer, M=512 tokens, four performance-core threads, and the
per-token figure multiplies by the layer count. Attention, norms, the final head
(the prompt tokens skip it) and sampling are excluded from both arms.

The correctness check compares every output element of the two arms; the
maximum absolute difference is reported against the peak activation.

## Results, linear layers only

| Model | Width | 2K prefill, today | 2K batched | 8K today | 8K batched | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 2B | 8-bit | 44.6 s | **4.0 s** | 178.3 s | **16.1 s** | 11.1x |
| Qwen 3.5 2B | 4-bit | 55.4 s | **3.9 s** | 221.8 s | **15.7 s** | 14.2x |
| Qwen 3.5 4B | 8-bit | 117.9 s | **11.4 s** | 471.5 s | **45.7 s** | 10.3x |
| Qwen 3.5 4B | 4-bit | 145.0 s | **11.3 s** | 580.2 s | **45.0 s** | 12.9x |

Per projection the batched arm is 9.5x to 15.9x faster. Dequantization is
10.0-12.1% of its time, so the conversion is not what limits it; for a longer
prompt that share shrinks. The two arms agree to a maximum absolute difference
of 1.4e-4 at 8-bit and 1.5e-5 at 4-bit against peaks of 37.5 and 2.8, i.e.
about 4e-6 relative — float32 reassociation, not different arithmetic.

## The real engine, as a control

The harness models four projections per layer; the installed dense models have
24/32 mixed layers, and the gated-DeltaNet layers carry their own
`in_proj_qkv`/`in_proj_z`/`out_proj` instead of the softmax QKV/O. So the
harness is a shape model, not the engine. To anchor it, a 652-token prompt was
sent to the actual CPU backend (`TinyTitanServer --cpu`) on the installed 2B:

| Install | Short prompt (22 tokens) | 652-token prompt | Per token beyond fixed cost |
| --- | ---: | ---: | ---: |
| Qwen 3.5 2B 4-bit | 2.30 s | 32.16 s | **47.4 ms** |
| Qwen 3.5 2B 8-bit | — | 33.32 s (warm) | ~50 ms |

That is 97 s for a 2K prompt and 388 s for an 8K prompt on the 4-bit install —
the earlier document's 117.6 s / 470.3 s were decode-rate projections, and the
real prefill is faster because prompt tokens skip the output head.

Two things fall out. **The 4-bit install is not faster than the 8-bit one**
(47.4 against ~50 ms), which is the side-engine document's finding again:
nibble unpacking costs what the saved reads buy back. And the per-token time is
the weight stream: the 4-bit install carries 0.78 GB of non-embedding weights,
which at 47.4 ms is ~16.5 GB/s — the same effective rate the GEMV arm of the
harness measures. Almost none of the prefill is left for non-linear work.

## What this changes in the recommendation

The earlier 25-29x figure compared *batched linear* against *whole-token
decode*. Those are different quantities: batching the projections removes the
linear share, not the token. By the harness's four-projection model that share
is 44-57% of the measured prefill (2B, 8-bit against 4-bit), so swapping only
those projections would leave 1.7-2.1x.

What makes the feature worth doing is that the share is not four projections:
it is every dense projection, including the gated-DeltaNet ones the harness
does not name. Batching all of them costs ~2.6-3 ms/token (the batched column
scaled to the installed parameter count) instead of 47-50 ms, and the measured
non-linear residue is small. A batched prefill should therefore land in the
**6-16 s range for 2K** and **25-65 s for 8K** — a 5-10x TTFT improvement,
not 25x and not 2x.

That range is an estimate, not a measurement: it assumes the gated-DeltaNet
recurrence and attention stay per token and cost a few milliseconds, which no
profile has confirmed. A per-operation profile of `CPUQwen35.step` is the next
measurement to take before sizing the work.

## Recommendation

1. **Do it**, with the per-token path kept as an explicit fallback and decode
   untouched. The quantized conversion is cheap (10-12%), the numerics agree to
   4e-6 relative, and the weight stream is what prefill is made of.
2. **Batch every dense projection**, including the gated-DeltaNet
   `in_proj_qkv`/`in_proj_z`/`out_proj`; batching only the softmax shapes leaves
   more than half the stream in the per-token loop.
3. Batch full-attention layers; keep the gated-DeltaNet recurrence token-ordered,
   because it reads the previous token's state. A chunked scan of that recurrence
   is the next lever if a profile says it matters again.
4. Quality gate before shipping: greedy output identical or within the measured
   logit noise floor with zero top-1 divergence. The harness shows the batched
   arithmetic is a reassociation, but only the engine's own oracle can certify
   the whole pipeline.
5. No raw `.word` AMX encodings. Accelerate is the public route; on M4+ gate the
   standard SME path by capability and mark it unverified on this host, as 6.1
   did for tensor-op prefill.

## Implemented, and what it measured

The batched prefill is in the engine. `CPUQwen35.prefill(tokens:)` chunks a
prompt at 2,048 tokens and carries the KV cache, convolution tail and recurrent
state across chunks; every dense projection goes through `CPUBatchedMatmul`
(dequantize once, one `cblas_sgemm`); full-attention layers batch their score and
value products; the gated-DeltaNet convolution and recurrence stay token-ordered,
with the recurrence in a chunked kernel (`tinytitan_delta_rule_chunk`, see
`plan-gdn-chunked-scan.md`) because the token-ordered one was three quarters of
the prefill once the projections were cheap.
`TINYTITAN_CPU_PREFILL=token` restores the per-token path and `=batch` selects
the chunked runner. The projection primitive shares the automatic 16-token
boundary even in that mode: below 16 tokens it calls the shipping GEMV once per
token instead of materializing FP32 weights. Decode is unchanged.

Measured on the installed 2B 4-bit with greedy decoding; the prompt is the
repeated pangram, and the times include model load and the generated tokens:

| Prompt tokens | Per-token path | Batched | Speedup | Output |
| --- | ---: | ---: | ---: | --- |
| 641 | 31.9 s | **3.9 s** | 8.2x | identical |
| 2,201 | 139.7 s | **8.9 s** | 15.7x | identical |
| ~4,000 | — | **18.2 s** | — | timing only |
| ~7,000 | — | **30.6 s** | — | timing only |

Once model load and the generated tokens are subtracted, that is roughly 3-4 ms
per prompt token against 47-63 ms, and the 2.2K result lands inside the 6-16 s
range this document predicted. The per-token figures at 4K and 7K grow because
both the sequential recurrence and the quadratic attention scale with prompt
length, and the recurrence is the larger of the two.

That ordering came from profiling a 7K prompt rather than from the earlier
guess: the gated-DeltaNet path was about two thirds of prefill and attention
about a tenth. So the second pass was spent there, not on blocking the
attention: the convolution's SiLU now runs through vForce over the whole
prompt, and the delta rule's independent heads are split across the performance
cores. The recurrence is still token-ordered; a chunked scan is the remaining
lever and it is an algorithmic change, not a scheduling one.

The quality gate is the output column: greedy text is byte-identical at 641 and
2,201 tokens, and 2,201 crosses the 2,048-token chunk boundary, so a state
hand-off between chunks would show there. The C delta-rule kernel and the
dequantized product are additionally checked against their scalar counterparts in
`tests/TinyTitan/CPUEngine/CPUBatchedPrefillTests.swift`.

### The 9B, both widths

The 9B is the one dense model with an untied head (`lm_head.weight`) and a
different value-head ratio, so it exercises paths the 2B/4B do not. Both of its
installs are present on this host and were run after the first pass had left
them out:

| Install | Prompt | Per-token | Batched | Output |
| --- | ---: | ---: | ---: | --- |
| Qwen 3.5 9B 4-bit | 151 | 33.3 s | **11.9 s** | identical |
| Qwen 3.5 9B 8-bit | 151 | 33.8 s | **16.6 s** | identical |
| Qwen 3.5 9B 4-bit | 401 | 85.3 s | **15.7 s** | identical |

The batched win is smaller here than on the 2B because the model is larger and
the dequantization cost is paid once per prompt, but it is still 2.0-5.4x and
the greedy text is unchanged at both widths.

### The side-engine's own prompts

The 12 authored T3/T5 cases from `benchmark/side_engine_wired_cases.py` are the
prompts the memory path actually sends: 110-124 rendered tokens each, chat
template and system message included. Run through `cpu35batch` on the 2B, the
set takes **14.9 s batched against 64.2 s per-token** (4.3x), and all twelve
completions are identical. That is the realistic short-memory-request shape, and
the batched path wins on it even though each request is its own prefill.
