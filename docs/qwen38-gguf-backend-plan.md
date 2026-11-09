# Qwen3.8 on GGUF: backend scope and the first landed component

> **Decision 2026-11-09: the model is used as it is. Conversion is parked.**
> The operator's instruction is to run the GGUF in place; the affine-conversion
> material below (probes, blob writer, planner, manifest builder) stays on the
> record as a rejected route, not as the plan. The as-is route needs IQ
> block-format kernels and a GGUF model path; its first milestone is a dequant
> kernel measured against the CPU decoder that is already bit-exact.

**Status 2026-11-09.** The Unsloth `UD-IQ3_XXS` checkpoint is installed at
`models/qwen3.8-flash-next-unsloth-iq3xxs` (83.63 GB base + 4.14 GB MTP +
0.90 GB mmproj). This note records its measured inventory, decides that the
engine must read GGUF **directly** rather than convert it, and names the work
between that decision and a served model. The directory reader it describes is
committed with eleven tests, one of them against this install.

## Why the affine install is not the path

| artefact | size | source |
| --- | ---: | --- |
| affine 4-bit MLX install (`Vontra/…-MLX-4bit`) | 111.6 GB | `docs/qwen38-flash-next-port.md:18` |
| bf16 source it would be built from | ~360 GB (132 shards) | same, `:12` |
| GGUF now on disk | 88.67 GB | measured, `stat` |
| free on `/System/Volumes/Data` | 79 GiB (84.8 GB) | `df -h` |

A conversion needs the source **and** the output at the same time, and even the
output alone (111.6 GB) is larger than the free space. The GGUF was already
downloaded and moved out of the Hugging Face cache (the cache's `blobs/` is now
empty and the file's link count is 1), so there is no duplicate copy to free.
Reading the file in place is the only variant that fits this machine.

That is also the harder variant: it puts block decoding and per-expert addressing
in the engine instead of in the converter.

## Measured inventory

Parsed with the reader in this commit; byte counts are the tensors' declared
geometry, not the file spans.

| shard | file bytes | tensors | metadata keys |
| --- | ---: | ---: | ---: |
| `…-00001-of-00004.gguf` | 10,946,618 | 0 | 67 |
| `…-00002-of-00004.gguf` | 49,946,027,488 | 1,210 | 3 |
| `…-00003-of-00004.gguf` | 4,869,325,888 | 13 | 3 |
| `…-00004-of-00004.gguf` | 28,800,138,432 | 1 | 3 |
| `mmproj-F16.gguf` | 904,004,000 | 334 | 31 |
| `mtp-…-Q8_0.gguf` | 4,137,429,120 | 34 | 51 |

1,224 base tensors totalling **83,615,413,760 bytes**. Shard 1 is metadata-only:
67 keys, no tensors, and it ends unaligned (10,946,618 % 32 = 2) with no data
section at all — a shape the reader had to learn.

**Architecture.** `general.architecture` is `qwen4exp`; the runtime already
declares this exact production geometry as `.qwen4flash`
(`sources/TinyTitan/Infrastructure/ModelIO/ModelTypes+Qwen4.swift:20-45`): 48
layers, 512 experts top-10, hidden 2,560, expert FFN 640, 24 heads / 2 KV, head
dim 256, vocab 248,320. The GGUF keys agree field for field:

| key | value |
| --- | --- |
| `block_count` / `context_length` | 48 / 262,144 |
| `expert_count` / `expert_used_count` | 512 / 10 |
| `expert_feed_forward_length` / `…_shared_…` | 640 / 640 |
| `ssm.inner_size`, `conv_kernel`, `state_size`, `group_count`, `time_step_rank` | 6,144, 4, 128, 16, 48 |
| `full_attention_interval` | 4 |
| `hyper_connection.count` / `low_rank` | 4 / 320 |
| `attention.indexer.top_k` / `head_count` / `key_length` | 2,048 / 4 / 128 |
| `ple.layers`, `ngram_size`, `heads_per_ngram`, `embedding_length_per_layer_input` | [1], 3, 8, 160 |
| `rope.dimension_count` / `dimension_sections` / `freq_base` | 64 / [11,11,10,0] / 1e7 |

**Tensor types.** Nine of them, all in the reader's table:

| type | tensors | bytes | role |
| --- | ---: | ---: | --- |
| IQ4_NL | 46 | 50,033,802,240 | down-experts (44 layers) + the PLE table |
| IQ2_S | 52 | 13,972,275,200 | gate/up experts (48 layers) + 4 |
| IQ3_XXS | 42 | 13,487,308,800 | some gate/up + attention |
| Q6_K | 250 | 3,328,281,600 | embeddings, output head, projections |
| IQ4_XS | 2 | 891,289,600 | layer 47's gate/up experts |
| Q8_0 | 248 | 841,850,880 | shared-expert and small tensors |
| Q2_0 | 3 | 707,788,800 | layers 8/10/13 down-experts |
| F32 | 557 | 313,495,040 | norms and scalars |
| BF16 | 24 | 39,321,600 | — |

`Q2_0` is **GGML type 42**, 64 elements per block, 18 bytes (`ggml-common.h`),
new enough that the file is what identified it: its derived 0.28125 bytes per
element is 18/64, and that was known before the type header was read.

**Where the bytes are.** Experts are 144 tensors / 50.29 GB (three per layer,
`[in, out, 512]`); embeddings + head 0.52 GB each at Q6_K, plus the PLE table at
28.80 GB; attention 384 tensors / 2.12 GB; SSM 252 / 0.51 GB; hyper-connections
195 / 0.35 GB.

**What one token reads from experts.** At top-10, one expert slice is 1.93 MB in
layer 0, 1.48 MB in layers 8/10/13 (Q2_0 down), and 2.60 MB in layer 47
(IQ4_XS gate/up). Ten per layer over 48 layers is **982.3 MB per token before
any cache hit** — the number the expert cache exists to cut.

**Tokenizer.** `tokenizer.ggml.model = gpt2`, `pre = qwen35`, 248,320 tokens,
247,587 merges, a 9,993-character `chat_template`, EOS 248,046, BOS/PAD 248,044,
`add_bos = false`, image token 248,056. All of it is in the GGUF; none of it is a
sidecar file.

## The reader (landed)

`sources/TinyTitan/Infrastructure/ModelIO/`: `GGUFType.swift` (the block table),
`GGUFReader.swift` (errors, values, shard, bounded read), `GGUFParser.swift`
(the cursor), `GGUFSplitModel.swift` (filename discovery and merge).

It validates magic and version (2/3), rejects unknown value or tensor types
rather than guessing, bounds the directory read at 64 MiB with a 1 MiB window
that doubles, checks every tensor's block geometry, alignment and file extent,
and merges a split set by `-NNNNN-of-MMMMM.gguf` while ignoring `mmproj-*` and
`mtp-*`.

`tests/TinyTitan/Infrastructure/ModelIO/GGUFReaderTests.swift` is 11 tests: ten
synthetic (all thirteen metadata wire types, three dissimilar tensor types, and
refusals for bad magic, version 1, unknown type, block misalignment, truncation,
out-of-range tensors, missing shards, ambiguous directories, metadata-only
shards) plus the gated real-install test:

```bash
TINYTITAN_GGUF_MODEL=models/qwen3.8-flash-next-unsloth-iq3xxs \
  swift test --no-parallel --filter GGUFReaderTests
```

Two facts came only from the real file and are now pinned: shard 1 has no data
section, and the truncation error must keep the parse offset rather than replace
it with the read-window size.

## What is still missing

1. **Block decoding.** All **nine types this checkpoint uses** are landed — F32,
   BF16, Q8_0, Q2_0, Q6_K, IQ4_NL, IQ4_XS, IQ2_S and IQ3_XXS — plus F16
   (`GGUFDequant.swift`). The IQ grids live in `GGUFIQTables.swift`, generated
   from ggml's header by `tools/gguf-embed-tables.py` and cited as MIT data
   rather than taken as a dependency.
2. **Per-expert addressing.** **Landed** (`GGUFExpertLayout`). Experts are stored
   `[in, out, 512]` in one tensor; every slice is block-aligned, so the expert
   offset is `expert * sliceBytes`. The component proves tiling rather than
   assuming it (`byteCount == 512 × slice`), which is the check that catches a
   writer padding between experts; all 73,728 slices over 48 layers tile their
   tensors with no gaps, and a block decoded at expert 511's computed offset is
   finite.
3. **Tokenizer and template from metadata.** **Landed.** `GGUFTokenizerSidecar`
   writes `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`
   from the GGUF's own tokens, merges, token types, special-token ids and
   template, and the result loads through `GFTokenizer.load(from:)`. What was
   *not* in the metadata and had to be cited: the pre-tokenizer regex (`qwen35`,
   from llama.cpp's `llama-vocab.cpp`, which names it as the original
   tokenizer.json regex) and the token-type mapping (CONTROL → special stop
   tokens, USER_DEFINED → literal `<tool_call>`/`<think>` barriers, which the
   runtime's streaming decoder requires). Ten prompts match `llama-tokenize`
   id-for-id. The writer is not yet wired into a load or install path — that is
   step 4's adapter.
4. **A weight source the engine can load.** `Model.load` requires a manifest and
   a receipt; GGUF has neither. Two measured constraints now shape this, and both
   were checked rather than assumed:
   - **There is no CPU forward for this architecture.** `CPUModelFamily` serves
     one family, `qwen3_5_dense`; Qwen3.8's GDN + full-attention + indexer +
     hyper-connection + PLE layers have no CPU implementation. A "load the
     non-expert path on the CPU" milestone therefore does not exist — the seam
     has to target the Metal runner.
   - **Experts are affine-only, but a bf16 GEMV exists.** `ManifestReader`
     accepts only weight bits 4 and 8, expert entries carry `bits ∈ {4, 8}`, and
     the expert kernels are `DequantInt4GEMV`/`DequantInt8GEMV`. Non-expert
     tensors, however, run through `BF16GEMV`. So decode-on-fetch would not need
     a *new* kernel format — it would need an expert dispatch path for bf16 —
     and the question becomes arithmetic rather than architecture:

     | quantity | value |
     | --- | ---: |
     | expert bytes read per token (IQ, no cache hits) | 982.3 MB |
     | elements those cover | 2.359 G |
     | same tile as bf16 | 4.72 GB |
     | I/O budget at 3.95 GB/s | 249 ms |
     | host decode rate needed to hide inside it | > 9.5 G elements/s |

     Nine and a half billion elements a second on one core is not plausible, so
     the estimate says host decode-on-fetch loses to the I/O it would hide
     inside. That is a projection, not a measurement, and milestone 5's job is
     to measure the real decode rate before choosing. If the measurement agrees,
     the route is kernels for the IQ block types on the GPU, where the decode
     runs beside the fetch instead of in front of it.

   This is the largest piece and is now the project's decision point; the
   measurement below closes one side of it.
5. **Verification.** There is no receipt for a GGUF. The honest checks are a
   content digest over the shard bytes, tokenizer parity, and generated-text
   comparison against a llama.cpp run on the same file — which this host does not
   have installed, so it is named as *not checked* rather than assumed.

## Order and acceptance

| # | milestone | acceptance | model needed |
| --- | --- | --- | --- |
| 1 | directory reader | 1,224 tensors, exact byte total, all metadata types | directories only |
| 2a | block decode: F32/F16/BF16/Q8_0/Q2_0/Q6_K/IQ4_NL/IQ4_XS | **landed 2026-11-09**: bit-exact against ggml `7d7f028` (`GGUFDequantVectors.txt`, regenerated by `tools/gguf-ref-vectors.c`) | no |
| 2b | block decode: IQ2_S / IQ3_XXS grids | **landed 2026-11-09**: same bit-exact reference, tables generated from ggml | no |
| 3 | tokenizer + template | **landed 2026-11-09**: the sidecar exports and loads, and ten prompts tokenize exactly as llama-tokenize at `7d7f028` does | no |
| 4 | per-expert addressing | **landed 2026-11-09**: 73,728 slices tile 50.29 GB with no gaps; block decode at expert 511's computed offset | directories only |
| 5 | engine seam: weight source + kernel format | **measured 2026-11-09**: host decode 0.21–0.36 G elements/s, ≈9 s/token, so decode-on-fetch is rejected; the fork between IQ kernels and an affine conversion is open | yes (GGUF) |
| 6 | affine conversion probe | **landed 2026-11-09**: sizes exact (50.29 → 67.95 GB for experts, ≈100 GB install) and the added error measured at 8.7–10.1% relative | yes (GGUF) |
| 7 | converter producing a real install | **landed 2026-11-09 (one layer)**: `GGUFExpertBlobWriter` writes a `.ssdai` expert blob and layout.json that the engine's reader accepts; weights read back equal the quantise→dequantise reference | yes (GGUF) |
| 8a | full layout plan | **landed 2026-11-09**: 48 layers x 512 experts x 9 sub-tensors, `layout.json` 19.2 MB (under the 128 MiB cap), expert payload 68.05 GB | yes |
| 8b | manifest | **landed 2026-11-09**: built from the GGUF + `qwen38FlashNext` preset, accepted by `ManifestReader.decode`, and refusing a file whose geometry disagrees | yes |
| 8c | resident role map | **landed 2026-11-09**: 44 roles over 1,080 resident tensors, every shape checked against the preset; a wrong dim is refused | yes |
| 8d | resident writer | **landed 2026-11-09**: `model_weights.bin` written with the engine's index format; `ResidentIndexReader` accepts it and the 4-bit payload decodes to the reference | yes |
| 8e | width policy + conversion plan | **landed 2026-11-09**: per-role widths, exact totals, and the PLE table exposed as a format blocker (175.6 GB with today's kernels) | yes |
| 8f | PLE storage | group-32 affine in the int4 kernel (~105 GB), or padded group-64 rows (~107 GB) | yes |
| 8g | full run | every tensor written; hashes; KLD/top-1 against llama.cpp | yes |
| 9 | expert streaming + cache | measured tok/s and output identity against a llama.cpp reference | yes |

Milestone 8 is the converter proper: every layer through this writer, a
manifest built from the GGUF's architecture metadata, and the full ~100 GB run —
which needs the source moved to external storage first so the output fits. The
writer proves the payload; what remains is the orchestration around it, plus the
KLD/top-1 check against a llama.cpp run that decides whether the conversion is
shippable.

The decoder is checked against ggml's own `dequantize_row_*`, not against
itself: `tools/gguf-ref-vectors.c` prints deterministic blocks and their decoded
bits, the fixture stores both, and the test compares bit patterns. The fixture
names the ggml commit so a later format change cannot move the reference
silently. Five tests, and a gated one that decodes real blocks through the
directory's own shard/offset arithmetic for five formats. The IQ2_S/IQ3_XXS
arithmetic passed against the reference on its first run; the two bugs that did
appear were caught before it — an operator-precedence mistake in IQ4_XS and the
element order of IQ3_XXS's two grids.

## Milestone 5: the measurement, and the fork it forces

Host decode, release build, one whole expert slice per call, two runs:

| format | run 1 | run 2 |
| --- | ---: | ---: |
| IQ2_S | 0.247 G elements/s | 0.214 G elements/s |
| IQ3_XXS | 0.362 G elements/s | 0.253 G elements/s |

A token reads 2.359 G elements of experts, so host decoding alone would cost
**6.5–11 s per token** against a 216 ms baseline, and the 249 ms of I/O it was
supposed to hide inside. The projection needed >9.5 G elements/s; the hardware
does ~0.25. Decode-on-fetch on the host is rejected by measurement, not by
argument.

```bash
TINYTITAN_GGUF_MODEL=models/qwen3.8-flash-next-unsloth-iq3xxs \
  swift test -c release --no-parallel --filter GGUFDecodeMeasurement
```

That leaves two real routes, and they differ by an order of magnitude in work:

- **A — IQ compute kernels on the GPU.** The engine would decode the block
  formats inside the GEMV, as llama.cpp does. This is not one kernel: every
  weight path uses one of these formats — routed experts (IQ2_S, IQ3_XXS,
  IQ4_NL, IQ4_XS, Q2_0), attention (Q6_K, IQ3_XXS), embeddings (Q6_K), the PLE
  table (IQ4_NL, 28.8 GB read sparsely) and shared experts (Q8_0). That is a
  kernel project comparable to the quantised GEMV work llama.cpp has accumulated,
  not an adapter.
- **B — Convert the GGUF to the engine's affine `.ssdai` format.** The kernels,
  the streaming expert cache, the receipts and the quality harness already
  exist; what is new is a converter that reads GGUF blocks, dequantises and
  re-quantises to 4-bit affine. The output is the same order as the 111.6 GB MLX
  install (`docs/qwen38-flash-next-port.md:18`; the converter's own `--plan`
  would state it), and it needs that output **beside** the 83.6 GB source:
  ~195 GB against 79 GiB free. It does not fit this disk today; it fits if the
  GGUF moves to external storage and the output is written here. Because the
  weights are re-quantised, output is not byte-identical to the GGUF and the
  quality bar has to be measured (KLD/top-1 against the GGUF run, not against
  the absent affine install).
- **C — Keep the GGUF as a llama.cpp artifact.** No engine work; this tree
  cannot serve it, and that is stated rather than implied.

Recommendation: **B**, when storage allows. It reuses the entire engine and is
the only route whose new code is a converter rather than a kernel library, and
the GGUF stays as the quality reference for the conversion.

## Route B, first probe: one layer re-quantised

The operator chose conversion to the engine's affine format, starting with one
layer. `GGUFAffineProbeTests` decodes a real expert slice, re-quantises it with
the engine's own `quantizeInt4Affine` (4-bit, group 64, BF16 scale + bias) and
reports both the size and the reconstruction error:

| tensor | source format | weights | rmse | max abs | relative |
| --- | --- | ---: | ---: | ---: | ---: |
| blk.0 gate | IQ2_S | 1,638,400 | 0.00129 | 0.00677 | 9.2% |
| blk.0 down | IQ4_NL | 1,638,400 | 0.00116 | 0.00780 | 9.3% |
| blk.14 gate | IQ3_XXS | 1,638,400 | 0.00110 | 0.00757 | 8.7% |
| blk.47 gate | IQ4_XS | 1,638,400 | 0.00158 | 0.01903 | 10.1% |

**Size is exact, not sampled.** Affine bytes are a function of the element count
(`elements/2` of nibbles plus 4 bytes per 64), so all 144 expert tensors are
counted without decoding them: **120,795,955,200 weights, 50.29 GB as IQ →
67.95 GB as affine 4-bit, 1.35×**. Adding the PLE table (51.2 G weights, the same
0.5625 B/weight as its IQ4_NL), the two 636 M-weight embeddings at Q6_K→affine,
attention/SSM/hyper-connection tensors and the BF16 norms puts the whole install
at **≈100 GB** (the expert figure is exact; the non-expert part is an estimate
from the inventory).

Two consequences:

- **Disk.** ~100 GB of output does not fit in the 79 GiB free here. It fits if
  the 83.6 GB source moves to external storage first, which frees ~78 GiB and
  leaves ~157 GiB for the output. The source still has to be readable during the
  conversion, so the minimum is one external volume holding the GGUF (which also
  stays as the quality reference afterwards).
- **Quality.** A 9–10% relative weight error is the *added* error of the second
  quantisation, on top of the IQ error the file already contains. It is not a
  quality verdict: that needs KLD/top-1 against a llama.cpp run of the same GGUF,
  and the conversion is not shippable until it passes. The probe is the cheap
  filter that says the conversion is worth attempting.

## As-is execution: what runs today, and what the engine needs

The checkpoint runs as-is under llama.cpp, and the numbers are the honest
baseline for this machine (24 GiB, 78 GiB file):

| build | offload | test | t/s |
| --- | --- | --- | ---: |
| CPU-only, 8 threads | — | pp64 | 2.00 ± 0.38 |
| CPU-only, 8 threads | — | tg16 | **0.09 ± 0.00** |
| Metal + BLAS, 8 threads | `-ngl 8` | pp32 | 0.60 ± 0.00 |
| Metal + BLAS, 8 threads | `-ngl 8` | tg8 | **0.06 ± 0.00** |

Partial offload is *worse*: the working set is split across two pools and both
page. llama.cpp has no expert cache, so a 78 GiB model on a 24 GiB machine
faults on nearly every expert read. The recorded affine install on this machine
is 4.62 tok/s, and the difference is the cache, not the format.

**The engine's as-is path** therefore needs, in order:

1. **Dequant kernels for the expert formats** — IQ2_S and IQ3_XXS (gate/up),
   IQ4_NL (down and the PLE table), IQ4_XS and Q2_0 (the layer exceptions) —
   writing bf16 tiles the existing `BF16GEMV` and gather kernels already read.
   The CPU decoders in `GGUFDequant` are the bit-exact reference for each.
2. **A GGUF model path**: build the architecture from metadata and the family
   preset (the manifest builder's cross-check is reusable), skip the
   manifest/receipt requirement, and open tensors in place.
3. **Expert streaming**: feed the IQ blocks from `GGUFExpertLayout` ranges into
   the dequant kernel, inside the existing prefetch/eviction machinery.
4. **PLE gather over IQ4_NL** — the table stays in its block format; only the
   gathered rows are decoded.

**Expert GEMV fused with decode (2026-11-09).** `iq2s_gemv_simd` and
`iq4nl_gemv_simd` read the gate/up (IQ2_S) and down (IQ4_NL) expert matrices in
their GGUF blocks and write half results in one pass - one SIMD per row, no
decoded matrix ever materialised. Verified against CPU decode-then-dot on the
real blk.0 expert 0 matrices: worst relative error 4.6e-4 (IQ2_S) and 4.8e-4
(IQ4_NL), both half output rounding.

| format | role | per matrix |
| --- | --- | ---: |
| IQ2_S | gate/up, most layers | 0.33-0.49 ms (640x2560) |
| IQ3_XXS | gate/up, 14 layers | 0.334 ms (640x2560) |
| IQ4_XS | gate/up, layer 47 | 0.406 ms (640x2560) |
| IQ4_NL | down, most layers | 0.41-0.49 ms (2560x640) |
| Q2_0 | down, layers 8/10/13 | 0.709 ms (2560x640) |

All five decode and dot in one pass and match a CPU decode-then-dot to within
4.6-4.9e-4 relative (half output rounding). With the real mix - ~660 IQ2_S
gate/up, 280 IQ3_XXS, 20 IQ4_XS, 450 IQ4_NL down, 30 Q2_0 - a token's expert
compute is **~589 ms, about 1.7 tok/s**, before resident GEMVs and I/O, against
the 4.62 tok/s affine baseline.

That is the as-is expert path's measured compute ceiling before I/O, against the
4.62 tok/s affine baseline. It is correct and the shape is right; vectorised
weight and x loads are the next optimisation, and the same harness measures any
gain.

**All seven block formats now decode on the GPU (2026-11-09).** Each kernel is
element-for-element identical to `GGUFDequant`'s CPU decoder, which is bit-exact
against ggml, and each writes bf16 tiles the existing `BF16GEMV` and gather
kernels already read. Measured on this M3, one block per thread, 1.64 M elements
x 20 warmed:

| format | where it is used | GPU rate (G elements/s) |
| --- | --- | ---: |
| IQ4_NL | down experts, PLE table | 2.45-5.14 |
| IQ2_S | gate/up experts (most layers) | 2.08-3.27 |
| IQ3_XXS | gate/up, layers 14+ | 2.63-3.71 |
| IQ4_XS | layer 47 gate/up | 2.86-3.13 |
| Q2_0 | layers 8/10/13 down | 2.96-3.26 |
| Q6_K | embeddings, head, attention | 1.93-2.30 |
| Q8_0 | shared experts | 2.34-3.37 |

Rates are ranges across runs (the machine moves by up to 2x); the CPU decoders
of the same formats run at 0.21-0.36 G, so the GPU is 6-15x faster. At ~3 G/s a
token's 2.36 G expert elements still cost ~790 ms, so vectorised stores and
fewer, larger launches remain before the as-is path can approach the 216 ms
affine baseline. Decoders are not a runtime: the loader that opens a GGUF
directory in place is still the missing piece.

**ANE is a prefill-only path here, and it was measured for this model.** The
engine never uses the ANE for decode. For prefill it was tried on the affine
4-bit install and does not pay: 197.5 s on the GPU against 273.5 s on the ANE for
a ~4,333-token prompt (**0.72x**; 8-bit 0.87x, `benchmark/ane-prefill/README.md`).
The mechanism is measured, not inferred: the GPU path gathers only the indexer's
~2,051 selected keys while the ANE graph is dense over the context (~4x the score
arithmetic for that prompt), and each Core ML variant load costs 6.9-13.6 s with
24 variants per prompt.

The as-is GGUF path has no ANE graph at all, and an ANE graph needs fp16 weights
of the whole block - which the checkpoint does not carry in that form and which
this route may not create. If the operator wants ANE prefill re-measured, the
exporter still supports the family and takes an affine install (`tools/
export_ane_prefill.py --model ... --chunk 4096 --max-history 8192`); it cannot be
produced from the GGUF without converting the very weights this route keeps in
place. Until that measurement exists, prefill stays on the GPU with the QSA
selection, and decode stays on the GPU.

## Reference measurement: llama.cpp on this host (2026-11-09)

The checkpoint runs today under llama.cpp, and the numbers say why this project
exists. Build `7d7f028`, CPU-only (`GGML_METAL=OFF`), 8 threads, BLAS, 24 GiB
machine:

```bash
llama-bench -m models/qwen3.8-flash-next-unsloth-iq3xxs/\
Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf -p 64 -n 16 -r 2 -t 8
```

| model | size | params | test | t/s |
| --- | ---: | ---: | --- | ---: |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw | 77.87 GiB | 176.94 B | pp64 | 2.00 ± 0.38 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw | 77.87 GiB | 176.94 B | tg16 | **0.09 ± 0.00** |

Eleven seconds per token. llama.cpp has no expert cache: it mmaps 78 GiB and
lets the OS page cache thrash, so nearly every expert read is a fault. The
measured affine install on the same machine is **4.62 tok/s** (216 ms/token) —
51x this — and the difference is the streaming expert cache and prefetch, not
the file format. That is the case for the conversion, and also why a GGUF
backend that merely dequantises on fetch (rejected in milestone 5) would not
have helped.

## Route B, seventh probe: the PLE table does not fit today's format

`GGUFConversionPlanner` sizes the whole install from the file. The numbers, with
the current width policy:

| component | bytes |
| --- | ---: |
| residents, 4-bit (embedding/head/shared) | 847,872,000 |
| residents, 8-bit (attention/router) | 2,928,721,920 |
| residents, bf16 | 103,768,344,320 |
| experts | 68,048,388,096 |
| **total** | **175,593,326,336 (175.6 GB)** |

The bf16 line is one tensor. `per_layer_token_embd.weight` is
`[160, 320001536]` — 51.2 G weights whose rows are **160 wide**, and the int4
shader hardcodes `kGroupSize = 64`. 160 is not a multiple of 64, so the affine
format cannot describe a row of it, and the policy falls back to bf16:
**102,400,491,520 bytes** where group-32 affine would take **32,000,153,600**.
The plan reports that fallback by name rather than hiding it in a total.

Three ways through, with their cost:

- **Group-32 affine for the PLE table.** Make `kGroupSize` a function constant
  in `dequant_int4.metal` and `dequant_int8.metal`, declare `groupSize: 32` in
  the manifest's embedding slot, and quantise the table at 32. PLE drops to
  32.0 GB and the install lands at **~105 GB**. This matches the checkpoint's
  own quant family (the community MLX build uses group 32) and pads nothing.
- **Pad the rows to 192 and keep group 64.** PLE costs 34.6 GB and the install
  ~107 GB, but the gather has to know the padded stride and slice the 160 used
  columns; the padded columns are zeros that must never reach the key
  projection. Similar work to the first, with more room to get it wrong.
- **Leave it bf16.** 175.6 GB, which does not fit the storage this machine can
  offer; this is the status quo the plan reports, not a route.

**The CPU half of the group-32 route is landed.** `quantizeInt4Affine`,
`dequantizeInt4Affine` and the Int8 pair now take an explicit group; the
one-argument forms remain and call them with 64, so every existing caller and
function reference is unchanged. A 160-wide row round-trips at group 32 in
`QuantizationAffineTests`. What remains is the shader's `kGroupSize` (a function
constant in `dequant_int4.metal`, used by `embed_lookup_int4` and the GEMVs) and
the manifest slot that declares 32 for the embedding family.

Either real option is a kernel change of the same order, so the group-32 route
is the recommendation. The ~100 GB figure in the earlier probe was right only if
the PLE table could be affine at a group that divides 160; this is where that
assumption is checked, and it was wrong by 70 GB.

## Route B, sixth probe: `model_weights.bin`

`GGUFResidentWriter` writes the resident file the way the engine reads it: a
binary index (24-byte header, 72-byte entries, string table, padded to 16 KiB)
followed by the payload, every tensor 16 KiB aligned. dtype 0 carries affine
weights with BF16 scale/bias companions; dtype 1 is the checkpoint's bf16 with
no companions. The width per tensor is the **caller's**, because which role gets
4, 8 or bf16 is conversion policy that has to match the manifest's slots — the
writer's job is only to make bytes and index agree.

Verified on four tensors of layer 0 (`GGUFResidentWriterTests`): the engine's
`ResidentIndexReader` parses the file, the entries carry the sizes the roles
imply (4-bit shared-expert gate 819,200 B + 51,200 B scales; 8-bit router
1,310,720 B; bf16 norm 20,480 B with no companions), and the 4-bit payload read
back at its recorded offsets dequantises to exactly the row the GGUF quantises
to. One guard had to move: the group-64 alignment is an affine constraint, not a
bf16 one, and holding `ssm_a` (48 wide) to it refused a legal tensor.

What remains is policy and orchestration: the per-role width table (so the
writer and the manifest agree), the hashes over written files, the full 68 GB +
resident run on external-backed source, and KLD/top-1 against llama.cpp.

## Route B, fifth probe: resident tensors map by role, with two traps

The first reading of this was wrong, and the shapes are what corrected it. What
the GGUF actually holds, against the runtime's roles:

| GGUF tensor | dims | runtime role |
| --- | --- | --- |
| `blk.0.attn_qkv.weight` (GDN layer) | [2560, 10240] | `gdnQKV` (2·16·128 + 48·128 = 10240) |
| `blk.0.attn_gate.weight` (GDN layer) | [2560, 6144] | `gdnZ` — the z gate, 48·128 = 6144 |
| `blk.0.ssm_out.weight` | [6144, 2560] | `gdnOut` |
| `blk.3.attn_q.weight` (full-attention layer) | [2560, 12288] | `qProj`, already `[query ; gate]` per head (2·24·256) |
| `blk.3.attn_k/v.weight` | [2560, 6144] | `kProj`, `vProj` |

So the per-tensor conversion is a **role map**, not a composition: the gate that
the runtime expects inside `q_proj` is already fused in the file for the 12
full-attention layers, and on the 36 GDN layers `attn_gate` is not an attention
gate at all but the linear attention's z projection. Concatenating `attn_q` and
`attn_gate` — the first draft of this section — would have built a tensor of the
wrong shape from the wrong pair.

**That table is now code and verified.** `GGUFResidentRoleMap` maps all 44 roles
over the 1,080 resident tensors and checks each against the shape the preset
implies — `attn_qkv` against `2·numKHeads·keyHeadDim + numVHeads·valueHeadDim`,
`attn_q` against `2·numHeads·headDim`, the hyper-connection pairs against
`hcCount·hidden` and the rank-320 bottleneck, the PLE norms against
`hcCount·hidden`, and so on. The counts land where the architecture says: 36 GDN
qkv/z, 12 full-attention q/k/v sets, 48 routers and shared-expert triples, 1
embedding/head/PLE table. A tensor whose dims disagree is refused by name, and
the negative test mutates `attn_qkv`'s row count to prove it.

What remains before the resident writer is the **width per role** (4-bit
embedding/shared, 8-bit attention/router, bf16 norms), then the bytes — for
which `ResidentWriter`-equivalent index encoding and the external storage are
the open items.

## Route B, fourth probe: the manifest

`GGUFManifestBuilder` builds the manifest from two sources on purpose: the file
is authoritative for geometry, the runtime's `qwen38FlashNext` preset for the
conventions a GGUF does not carry in a form the engine reads. Every field with a
GGUF key is compared with the preset, and a disagreement is refused by name.

The result decodes through `ManifestReader.decode(data:expecting:)` with all the
extension geometry present: hyper-connections 4, indexer budget 2,048, PLE n-gram
3, and the affine 4-bit/group-64 quant slots. The negative test mutates
`block_count` to 47 and gets `numLayers` named in the refusal.

The cross-check earned its keep immediately: it caught this repo's own mapping
being wrong. `embedding_length_per_layer_input` is 160, the PLE table's row
width (`embedDim / ngramHeads`), not the preset's 2,560-wide injection — reading
it as `embedDim` would have described a tensor that does not exist.

`files` stays a parameter: sizes and SHA-256 values describe bytes the conversion
has not written yet, and inventing a hash would be worse than having no manifest.
Likewise the quant slots are a claim about bytes, so the resident writer has to
emit exactly what they say before the manifest is shippable.

## Route B, third probe: the full layout plan

`GGUFInstallPlanner` builds every layer's records from the GGUF's geometry
without reading a weight, encodes `layout.json` and decodes it through the
format's own validator. Measured on the install:

- **48 layers × 512 experts × 9 sub-tensors = 221,184 records**;
- `layout.json` is **19,190,558 bytes**, comfortably inside the reader's 128 MiB
  cap — that cap was the one number that could have made the whole layout
  unwritable, and it is not close;
- expert stride 2,768,896; the last expert starts at `511 × stride` and its
  highest sub-tensor ends inside the stride;
- expert payload **68,048,388,096 bytes (68.05 GB)** — the probe's 67.95 GB plus
  ~0.10 GB of 16 KiB per-expert padding.

What remains for a loadable install is the manifest and the resident weights:
the arch block can be taken from the runtime's `qwen4Flash` preset and
cross-checked field by field against the GGUF metadata, the quant slots describe
the affine scheme this writer emits, and `files` needs the sizes and hashes only
the full run can produce. The KLD/top-1 check against a llama.cpp run of the
same GGUF is still the bar the conversion has to pass.

## Route B, second probe: one layer in `.ssdai` layout

`GGUFExpertBlobWriter` writes a layer's routed experts exactly as the runtime
reads them: per role, 4-bit affine weights in U32 words with
`shape: [rows, columns]`, then BF16 scales and biases shaped
`[rows, columns/64]`, with sub-tensor offsets relative to the expert. It takes
the GGUF's IQ bytes, decodes them, and re-quantises with the engine's own
`quantizeInt4Affine`, so the payload is what the kernels already consume.

Verified on four experts of layer 0 (`GGUFExpertBlobWriterTests`): the layer
file is `experts × stride`, `layout.json` decodes through the engine's
`PackedExpertsLayoutReader`, and expert 3's first and last gate rows read back
from their recorded offsets equal the quantise→dequantise reference exactly.
The expert records carry `bits: 4` and the shipped shapes, so
`validateRoutedExpertLayout` has what it checks.

One format fact the first attempt got wrong: `expertStride` must be a multiple
of the 16 KiB `alignmentBytes`, and the raw blob is 2,764,800 bytes
(168.75 pages). Each expert is padded to 2,768,896; a converter that wrote the
unpadded size would be refused by the layout validator, which is how this was
found.

## Open questions

- **Dequant placement.** IQ2_S/IQ3_XXS decoding is table-heavy; doing it on the
  host per 2.6 MB expert slice may cost more than the GPU saves. Measure host
  decode throughput for each type before choosing.
- **PLE.** The 28.8 GB table is read sparsely (160-wide rows at 3-gram head
  offsets); it must stay on the pread/gather path, never resident.
- **Q2_0 provenance.** It is recent enough that the reference implementation
  must be pinned by llama.cpp commit, not "current main".
- **Headroom.** 79 GiB free, 28.8 GB of it a single table. There is no room for
  a second install of anything.
- **Quality bar.** The affine install this would replace is not on disk, so
  output identity cannot be compared to it. The reference has to be llama.cpp on
  the same GGUF, and the comparison has to be stated as such.
