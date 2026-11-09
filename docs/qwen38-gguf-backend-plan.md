# Qwen3.8 on GGUF: backend scope and the first landed component

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

1. **Block decoding for the nine types.** **Seven are landed** (F32, F16, BF16,
   Q8_0, Q2_0, Q6_K, IQ4_NL, IQ4_XS — `GGUFDequant.swift`); IQ2_S and IQ3_XXS,
   the two lookup-table formats in the expert tensors, are milestone 2b and are
   refused by name. The ggml sources are MIT-licensed, but the tables are code to
   port and cite, not a dependency to add.
2. **Per-expert addressing.** Experts are stored `[in, out, 512]` in one tensor.
   The runtime's `ExpertEntry` model wants a per-expert offset into a layer file;
   a GGUF source can synthesize those offsets without copying, provided each
   slice is block-aligned (the reader already proves the element count is).
3. **Tokenizer and template from metadata.** The engine currently reads
   `tokenizer.json` and `chat_template.jinja` from an install. A GGUF path must
   either build the BPE from `tokens`/`merges`/`pre` or write those sidecars at
   install time — the second is less code and keeps one runtime path.
4. **A weight source the engine can load.** `Model.load` requires a manifest and
   a receipt; GGUF has neither. The adapter must implement the tensor fetch and
   the expert-cache interface, and decide where dequantisation happens:
   host-side into fp16 tiles (simple, doubles slot bytes) or in-kernel (no extra
   bytes, new Metal work). This is the largest piece and should be measured
   before it is designed around.
5. **Verification.** There is no receipt for a GGUF. The honest checks are a
   content digest over the shard bytes, tokenizer parity, and generated-text
   comparison against a llama.cpp run on the same file — which this host does not
   have installed, so it is named as *not checked* rather than assumed.

## Order and acceptance

| # | milestone | acceptance | model needed |
| --- | --- | --- | --- |
| 1 | directory reader | 1,224 tensors, exact byte total, all metadata types | directories only |
| 2a | block decode: F32/F16/BF16/Q8_0/Q2_0/Q6_K/IQ4_NL/IQ4_XS | **landed 2026-11-09**: bit-exact against ggml `7d7f028` (`GGUFDequantVectors.txt`, regenerated by `tools/gguf-ref-vectors.c`) | no |
| 2b | block decode: IQ2_S / IQ3_XXS grids | same reference method, both expert formats | no |
| 3 | tokenizer + template | identical ids on a fixed prompt set and an identical rendered prompt | no |
| 4 | non-expert load, CPU | `The capital of France is` → ` Paris` continuation | yes (GGUF) |
| 5 | expert streaming + cache | measured tok/s and output identity against a llama.cpp reference | yes |

Milestone 2b is the right next step: it is model-free, and IQ2_S/IQ3_XXS are 97
of the 1,224 tensors and 27.5 GB of the expert weights. A wrong grid would
otherwise surface as plausible-looking garbage several days later.

The 2a decoder is checked against ggml's own `dequantize_row_*`, not against
itself: `tools/gguf-ref-vectors.c` prints deterministic blocks and their decoded
bits, the fixture stores both, and the test compares bit patterns. The fixture
names the ggml commit so a later format change cannot move the reference
silently. Five tests, and a gated one that decodes real blocks through the
directory's own shard/offset arithmetic.

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
