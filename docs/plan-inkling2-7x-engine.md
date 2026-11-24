# Inkling 2 Mini on the 7.0 engine — two variants, one comparison

Status: **plan**, written 2026-11-23 (TT-121), `.ssdai` installs decided. Supersedes the two side routes
as the way Inkling is served; they are retired in phase 6.

## Why

Inkling 2 Mini runs today on two routes that bypass the 7.0 engine:

| Route | Source | What it lacks |
| --- | --- | --- |
| `sources/TinyTitan/Inkling2GGUF/` | the GGUF file, mapped as-is | `.ssdai` install and receipt, SSD expert streaming, `--ram-budget`, expert cache, `--kv-bits`, prompt cache, more than one sequence, calibration, golden baseline, launcher/installer rows |
| `sources/Engines/Inkling2/` + `Inkling2ServerBackend` | the NVFP4/MXFP8 checkpoint | all of the above, plus the whole API (text only, 64-token answers — TT-118) |

The project rule is that every served model is a 7.0 variant
(`docs/plan-tinytitan-7-engine.md`) and that every route serves the whole API
but vision (AGENTS.md). Inkling meets neither. The measured consequence on a
24 GB M3: the Q4 GGUF maps 16.8 GB, `--ram-budget 4G` is accepted and ignored
(15.95 GB resident), and the machine pages the weights — 3.4 tok/s decode
against 19 tok/s measured when memory was free.

The owner wants **both** sources as variants so they can be compared:

- `inkling2-gguf-q4` — built from `Inkling-2-Mini-UD-Q4_K_XL.gguf` (unsloth,
  sha256 `e62d463c…4de15`), its K-quant/IQ blocks kept byte for byte.
- `inkling2-nvfp4` — built from the original checkpoint
  (`models/inkling-2-mini`): NVFP4 block-16 experts in layers 0–19, MXFP8
  block-32 in 20–23, BF16 elsewhere, kept byte for byte.

Neither requantizes (owner's rule, 2026-11-23: only the routed experts are
4-bit; everything else stays at its original 8 or 16 bits). Both sources
already have that shape, so both installs copy their bytes:

| | Q4_K_XL GGUF | original checkpoint |
| --- | --- | --- |
| experts, layers 0–19 | NVFP4 (11.47 GB) | NVFP4 (10.32 GB values + E4M3 scales) |
| experts, layers 20–23 | Q8_0 (4.33 GB) | MXFP8 |
| everything else | Q8_0 (1.67 GB), F32/BF16 norms | BF16 (3.85 GB), F32 |

The last four layers' experts are 8-bit in both, as the model's authors and
Unsloth chose; making them 4-bit would be a requantization, which the rule
excludes. The comparison then measures the formats and their kernels, not a
conversion.

## What a variant gets that the side routes do not

Everything the shared `Session` and `EngineFactory` give every other model:
the verified `.ssdai` install, routed experts streamed from `packed_experts/`
with the slot cache sized by `--ram-budget` / `--expert-cache-slots`, KV
precision (`--kv-bits`), the prompt cache (`captureInferenceState`), context
and YaRN handling from the load plan, the one generation path for CLI and
server (templates, tool calls, stop strings, JSON grammar, thinking), the
catalog and `/v1/models`, the launcher and installer rows, `calibrate`, golden
baselines and `release.sh`'s gates.

## Phases

Each phase ends green on `tools/lint.sh` and `swift test --no-parallel`, and is
committed on its own.

### 0. Format contract (Core, additive)

The `.ssdai` contract is affine-only today: a quantized tensor is `dtype 0`
packed words plus BF16 scales and biases, and `quant.<role>.scheme` is
`affine`.

- Add encodings as **format** facts, not model facts (`core-purity`): an index
  `dtype` per raw block format — GGUF `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_XS`
  and whatever else the Q4 file carries (enumerated from the file, not assumed),
  and `NVFP4` (block 16, FP8-E4M3 scales, tensor scale) and `MXFP8` (block 32,
  UE8M0 scales). `scheme` names them; `shape[1]` stays the logical width;
  `sizeBytes` is the raw byte count.
- `arch.variantGeometry`: named scalars and integer lists the variant alone
  interprets (keys below), instead of new per-family `arch` fields.
  `ArchConfig` gets only the generic fields the session reads (vocabulary,
  layers, hidden, heads, context).

**Done 2026-11-23:** encodings 4–7 with byte-exact size checks in the index
and the expert layout, `variantGeometry` through to the Core manifest, the
format document, and `SSDAIBlockEncodingTests`. The repacker's writer gains the
encodings in phase 1, where the first source that needs them is read.
- Reader, writer and receipt validation for the new encodings; existing
  installs re-verify unchanged. Unit tests on synthetic tensors.

#### Inkling's `variantGeometry` keys

What the decoder reads beyond the shared `arch` fields (hidden size, heads,
KV heads, head dimension, vocabulary, layers, experts, top-k, sliding window,
softcap), taken from what the GGUF decoder reads today
(`Inkling2GGUFManifest`). The repacker writes all of them; the variant refuses
an install missing one.

| Key | Kind | Meaning |
| --- | --- | --- |
| `loopStart`, `loopEnd`, `loopPasses`, `loopGateRank` | scalar | the looped core (layers 6–17, three passes, gate rank 64) |
| `unrolledLayers` | scalar | layer visits per token (48) |
| `attentionResidualBlock` | scalar | layers per attention-residual block |
| `relativeLength`, `relativeExtent`, `relativeClip` | scalar | relative-position bias rank, its table length, its clip |
| `convolutionTaps` | scalar | dynamic-convolution taps |
| `paddedVocabulary` | scalar | the head's padded row count |
| `rmsEpsilon`, `qkClip`, `logScaleFloor`, `logScaleAlpha` | scalar | norm epsilon, Q/K clip, attention log-scale |
| `expertWeightsScale`, `situBeta`, `situLinearBeta` | scalar | routed-weight scale, the SiTU activation |
| `localLayers` | list | the sliding-window layers; the rest are global |

### 1. Installs (TinyTitanRepack), streamed

The project's hard rule (AGENTS.md, "Models") is that every Hugging Face
download is converted to `.ssdai` while it downloads: a chunk is fetched,
converted into the install and deleted before the next is staged, and the full
source never sits on disk. Both Inkling sources are single files (17.6 GB GGUF,
19 GB safetensors), so a chunk is a byte range.

- `TinyTitanRepack --model inkling2-gguf-q4` and `--model inkling2-nvfp4` fetch
  the pinned file by ranged requests (the installer's 64 MiB ranged, length-
  checked, resumable download), parse its header first (GGUF metadata and tensor
  table; safetensors JSON header), and convert tensor by tensor as their bytes
  arrive, writing `models/inkling-2-mini_GGUF-Q4/` and
  `models/inkling-2-mini_NVFP4/`. Peak extra disk is the chunks in flight, not a
  second copy.
- Routed experts (`gate_up`, `down` per layer, 128 experts) to
  `packed_experts/layer_NN.bin`, one 16 KiB-aligned blob per expert; everything
  else resident in `model_weights.bin`; the tokenizer sidecar (tiktoken files and
  `chat_template.jinja`); `verified-install.json`.
- Verification: every tensor's payload hashed against its source bytes, so
  "byte for byte" is checked, not claimed. The GGUF install's tokenizer and
  template must equal the GGUF's own (the existing tokenizer parity tests
  apply).
- Transition: the raw `models/inkling-2-mini` and `models/inkling-2-mini-gguf`
  folders predate the rule. The converters also accept a local file as the
  source, so these two are converted once from disk, verified, and then
  deleted; every later install streams from Hugging Face.

Why `.ssdai` and not in-place reads, measured 2026-11-23 on the 24 GB M3
(cold reads, layers of identical expert size, F_NOCACHE): one aligned read per
expert ran 2.95 GB/s with a 13.3 ms p99 layer step against 2.63 GB/s and
17.6 ms for the GGUF's two unaligned reads, and the GGUF's reads were cached
despite F_NOCACHE — 53.5% of the expert pages read were resident afterwards,
against 0% for the aligned layout. In-place streaming would fill RAM with page
cache beside the slot cache.

**Progress 2026-11-23.** The GGUF half is done for a file on disk:
`GGUFHeaderV3` (format), `Inkling2GGUFPlanner` and `TinyTitanRepack
--input-gguf` (`1364ce88`, `75bab405`, `13d2f70a`). The Q4_K_XL file converted
in 57 s into `models/inkling-2-mini_GGUF-Q4` (16 GB); `--verify-install`
passes, and every tensor byte and the header compare equal to the source
through llama.cpp's gguf-py. The raw GGUF stays until phase 2's variant
serves the install.

The checkpoint half is done for a snapshot on disk: `TinyTitanRepack
--input-snapshot` routes a `config.json` of model type `inkling2_loop` to
`Inkling2CheckpointPlanner`, which reads both safetensors headers (the
text model and `mtp.safetensors`) and `config.json` through the same
`Inkling2Arch` builder the GGUF uses, so both installs carry one manifest
shape. Layers 0-19 keep NVFP4 experts (packed E2M1 values, E4M3 scales per 16,
per-expert F32 `amax`; the per-layer `outer_scale_denominator` is resident),
layers 20-23 keep MXFP8 (E4M3 values, UE8M0 scales per 32), and every other
text tensor stays BF16/F32. The MTP head is resident under `mtp.`; the vision
tower (422 tensors, 0.64 GB) is left out like every install's image input.
The tokenizer, chat template, tiktoken model and the quantization and
generation configs are copied under `tokenizer/`. The snapshot converted in
about 2 minutes into `models/inkling-2-mini_NVFP4` (18 GB, 35 files);
`--verify-install` passes, and an independent check against the safetensors
headers found all 19.525 GB of non-vision tensor bytes equal, none missing.
The raw checkpoint stays until phase 3's variant serves the install.

Streaming from Hugging Face is done for both: `TinyTitanRepack --model
inkling2-gguf-q4` (`unsloth/Inkling-2-Mini-GGUF` at `45fd0876`) and `--model
inkling2-nvfp4` (`thinkingmachines/Inkling-2-Mini` at `ed40e7da`). Before
planning, only the GGUF header (one 16 MiB range) or the two safetensors
headers, `config.json` and the index are fetched; every tensor byte then
arrives through the ordinary ranged copy with its checkpoint and `--resume`,
so no source file is staged. Each source is pinned by commit and by a
fingerprint over exactly what decides the layout: the GGUF header's SHA-256,
or `config.json` plus both safetensors headers. A dry run against Hugging Face
planned both installs identically to the local copies, and a full install of
each, streamed from the local files through a loopback mirror (GGUF 97 s,
checkpoint 92 s), hashed file for file like the local install and passed
`--verify-install` (`tests/TinyTitanRepack/Core/Remote/Inkling2RemoteInstallTests.swift`,
opt-in). Both sources are listed by the repack CLI only; the installer menus
and the README matrix gain them when a variant serves them (phase 4).

### 2. Variant `Engines/Inkling2GGUFQ4`

- Graph: the GGUF decoder's looped graph (48 unrolled visits, sliding and
  global attention with relative bias and sinks, dynamic convolution, message
  controllers, sigmoid top-k MoE, softcapped head) moved into the variant
  target with its kernels (copied with a provenance header, per the isolation
  rule), behind `EngineModel` / `EngineRunner`.
- Experts: read through the 7.0 streaming path — slot cache, prefetch,
  `--ram-budget` — instead of mapping the file. The resident set is the
  non-expert weights plus the slot cache.
- KV: per-visit caches allocated by the load plan; `--kv-bits 8` as a
  variant-owned format if it holds parity (that is what lets a 24 GB Mac give
  OpenCode a real window; the Unsloth app runs this model with 8-bit KV).
- Prompt cache: state capture/restore of the KV rings and convolution state.
- Gates: the llama.cpp parity suite (short Q4 fixture, chunked prefill),
  the long-context fixture (Q2 file when installed), golden capture,
  `api-feature-check` 10/10.

**Progress 2026-11-23.** The target exists (`sources/Engines/Inkling2GGUFQ4`):
the GGUF route's graph and its NVFP4/Q8_0/F32/BF16 kernels, copied, reading a
verified install — the receipt is checked against the manifest and the
directory before anything is mapped, the geometry comes from
`arch.variantGeometry` and the resident index, and each layer's experts are
read at the install's per-expert stride. `InklingQ4ParityTests` (model-gated on
`TINYTITAN_INKLING2_Q4_INSTALL`): 96/96 greedy tokens agree with llama.cpp's
`q4kxl-reference.json`, and every logit of those 96 teacher-forced steps,
chunked prefill included, is bit-identical to the GGUF route on the source file.

Experts stream (`InklingQ4ExpertResidency.slots(n)`): `n` experts per layer in
the process's own memory, read with `F_NOCACHE`, least recently used evicted.
A visit's router ends its command buffer, the CPU reads the routed ids, loads
the missing experts and continues on slot ids; a prefill chunk that routes to
more experts than a layer has slots runs in rounds, each finished before the
next one's reads overwrite its slots. At `n` = 128 every expert is read once at
load and no visit stops. Logits are bit-identical to the mapped install at 16
slots (4,959 rounds over 4,608 visits, 74 GiB read) and at 64.

Decode, release build, 24 GB M3, 64 tokens after a 5-token prompt, two rounds
(74% of memory free, so the mapped run had the whole file in page cache — its
best case, not what it gets under pressure):

| Experts | tok/s | Expert RAM | Per token |
| --- | --- | --- | --- |
| mapped | 14.2 / 17.3 / 13.6 | page cache (≈ 16 GB) | GPU 67.8 ms |
| 64 slots | 6.38 / 6.30 / 6.22 | 7.5 GB | GPU 77.6, reads 56.2, syncs 27.1 ms; 89% hits |
| 32 slots | 4.36 / 4.27 / 4.26 | 3.8 GB | GPU 93.3, reads 111.8, syncs 29.7 ms; 80% hits |

**Served, 2026-11-23.** `TinyTitanServer --model models/inkling-2-mini_GGUF-Q4`
serves the variant through the Inkling backend (`Inkling2Q4ServerLoad`; model id
`inkling-2-mini-gguf-q4`): its decoder sits behind `Inkling2Decoding` in the
same `Inkling2GGUFSession` the GGUF route uses, and the tokenizer comes from the
install's copy of the GGUF header (`GGUFModelReader.readHeader`). `--ram-budget`
(or `--expert-cache-slots`) sizes the plan (`InklingQ4LoadPlan`): weights and a
512 MiB reserve first, then the window's KV, then as many expert slots as the
rest holds; with no budget the experts get 8 GiB. `tools/api-feature-check.py`:
**10/10** at `--ram-budget 12G` (76 slots, 32K window; footprint 10 GB). The
shared session (prompt cache, several sequences, KV bits) is still phase 4.

4 GiB budget, for an 8 GB M2 Mac mini — **measured on the 24 GB M3, not on an
M2**: the budget caps what the process allocates, but the M3's page cache kept
the 1.8 GB of mapped resident weights warm, and an M2's GPU and SSD differ.
Streaming client, temperature 0, 128-token answers to 25-32-token prompts,
three prompts x two rounds, release build:

| Launch | Plan | Decode tok/s | TTFT | Footprint |
| --- | --- | --- | --- | --- |
| `--ram-budget 4G --max-context 4096` | 14 slots, 4K | 2.66-2.97 | 4.5-5.0 s | 2.37 GB + 1.8 GB mapped weights |
| `--ram-budget 4G --max-context 32768` | 8 slots, 28K | 2.24-2.35 | 4.9-5.7 s | 2.44 GB + 1.8 GB mapped weights |

At 4 GiB a long window costs speed: its KV takes what would be expert slots,
and at 8 slots (one token's worth) nearly every expert is a miss.

The gap was the cost of the 48 router stops a token: the reads did not overlap
GPU work, and each stop is a commit-and-wait.

**Prefetch, 2026-11-23.** History predicts nothing LRU does not already keep:
on a decode trace of the six short reference prompts (96 tokens each, one of them code), prefetching the
previous token's experts for a visit, or the previous loop pass's for the same
layer, removes no blocking miss at 32 or 64 slots (every such expert is still
resident) and 3% at 14. So each decode visit runs the *next* visit's router on
its own MoE input (`lookaheadPrefetch`, on by default): 73.8% of the experts it
names are the ones that visit then chooses. The experts' command buffer is
committed at once, the cache starts the prediction's reads on a background
queue into least-recently-used slots of the next layer, and the next visit
waits for them before it resolves — so the logits stay bit-identical (checked at
16 and 64 slots). Decode, release build, paired and interleaved, prefetch off
-> on:

| Slots | Rounds | tok/s off | tok/s on | Blocking misses |
| --- | --- | --- | --- | --- |
| 64 | 2 | 6.12, 3.99* | 7.16, 7.13 | 2,781 -> 1,663 |
| 32 | 5 | 4.25, 4.20, 4.23, 4.26, 4.00 | 4.76, 4.33, 4.52, 4.64, 4.16 | 5,071 -> 2,915 |
| 14 | 5 | 2.53, 2.61, 2.46, 2.25, 2.42 | 2.17*, 2.68, 2.68, 2.56, 2.67 | 9,523 -> 5,040 |

\* GPU time per token doubled in that run alone (161 and 279 ms against 74-147):
other work on the shared machine, not the change. Prefetch won every other pair.

**Rejected: evict by recent count.** Keeping the experts routed most often in
a layer's last 250 routings (least recently used breaking ties) cut blocking
misses 7% / 4% / 1% at 14 / 32 / 64 slots in a no-prefetch simulation of the
trace, but in the engine, with prefetch, it had 5.5% *more* blocking misses at
14 slots (5,316 against 5,040) and lost all three paired rounds there (1.90,
2.64, 2.23 against 2.42, 2.72, 2.79 tok/s); at 32 slots it was mixed (one win,
two losses, misses level) and at 64 it won three rounds only while it always ran
second — with the order swapped it was level (7.12 / 7.37 against 7.21 / 7.35).
Not committed; LRU stays. The measurement covers these short general prompts only: a coding workload (long contexts, repeated identifiers, tool turns) is not measured, and is where a frequency window could still behave differently.

Served at `--ram-budget 12G --max-context 32768` (76 slots) with prefetch:
`api-feature-check` 10/10; streaming client, three prompts x two rounds, decode
5.8-10.2 tok/s, TTFT 1.5-2.6 s, footprint 10 GB. Prefill still stops at every
router without a prediction (a chunk routes to most experts, so there is little
to guess) — that, and the shared session (`EngineModel`/`EngineRunner`, KV
bits, prompt cache), are what remains.

**Swift coding workload and the 250-record database, 2026-11-23.** Quiet M3
(24 GB), release build. `InklingQ4CodingWorkloadTests` (opt-in,
`TINYTITAN_INKLING2_CODING=1`) runs five TinyTitan-style Swift requests through
one decoder, 256 greedy tokens each, 1,280 decode tokens in all; the windows
are full after the first request's prefill.

*The database, live.* Pinning each layer's top experts of its last 250
routings (loading at most two missing ones per visit), against plain LRU, both
with lookahead prefetch, order swapped, 76 slots (the 12G plan): decode 9.84 /
9.86 against 9.88 / 9.63 tok/s, prefill 15% slower. Blocking misses fell 30%
(17,111 -> 11,768), but the pin loads (5,846) are reads too, and total reads
were level. Rejected and removed.

*The database, replayed.* The decode trace of the workload (every routed set
and every prediction, `TINYTITAN_INKLING2_ROUTING_TRACE`) replays through a
model of the cache that reproduces the live counters exactly (17,111 misses,
21,057 prefetches). Reads per decode token at 76 slots:

| Policy | Blocking | Background | Total |
| --- | --- | --- | --- |
| LRU + lookahead (shipped) | 9.0 | 16.5 | 25.4 |
| LRU, no prefetch | 16.8 | 0 | 16.8 |
| pin 32 / 48 / 64 of 250 + lookahead | 8.8 / 8.7 / 8.6 | 16.6 / 17.0 / 17.9 | 25.5 / 25.8 / 26.6 |
| evict by 250-window count + lookahead | 8.5 | 16.8 | 25.3 |
| lookahead + 1 / 4 / 8 top-count experts | 8.8 | 26.3 / 43.7 / 54.9 | 35.1 / 52.4 / 63.7 |
| evict by count, window 250 / 1000 / 4000, no prefetch | 16.6 / 16.1 / 18.7 | 0 | 16.6 / 16.1 / 18.7 |

The clairvoyant floor (Belady, demand reads only) is 6.8 a token at 76 slots,
against 16.8 for LRU: a policy could save that much, but what Belady uses is
*when* an expert returns, which a frequency window does not see. As a
predictor, the window's top 8 for a layer name 37.7% of the experts that visit
routes; the same visit one token earlier names 39.1%; the next visit's router
run on this visit's input names 73.7%, and further ahead 66.0 / 59.1 / 54.9 /
51.4 / 48.7% at 2-6 visits. Deeper prediction removes 0.4 blocking reads a
token at depths 1-2 for 8 more reads; the database's extras add reads and
remove none. **The database does not help prefetch on this model:** its
routing is spread too evenly for counts to say more than LRU already keeps.

*Where the time goes.* At 76 slots a token took ~100 ms against 50.7 ms
all-resident (128 slots, 19.7 tok/s; mapped 18.6-19.7). With the reads skipped
(wrong data, timing only) the same 76-slot run is 17.0 tok/s: the whole gap is
the GPU waiting for expert reads at the routers, not round trips or encoding.
What closed part of it, each order-swapped and bit-identical (16 and 64 slots,
both routing modes; 96/96 against llama.cpp):

| Change | Measured | Kept |
| --- | --- | --- |
| One command buffer per decode token; each router signals a shared event and the GPU waits for the CPU's slot resolution (`eventRouting`) | 9.30 / 9.06 / 9.28 against 8.30 / 8.58 / 8.59 tok/s (short prompts) | yes |
| The decoding thread polls the event instead of a listener queue waking (`spin`) | 9.56 / 9.64 against 9.29 / 9.26 | yes |
| A visit waits only for the in-flight reads of the experts it routed, not every prefetch (`waitPerExpert`) | 10.73 / 10.49 against 10.15 / 10.17 (coding) | yes |
| Each expert read as four concurrent 16 KiB-aligned parts | 10.94 / 10.98 against 10.98 / 10.95 | no |

The split read looked like 2.5x in a microbenchmark that ran after the mapped
benchmarks had warmed the page cache; cold and in the engine it is level.

*What does scale: memory.* The coding workload against slots per layer, one
run each, a 4K window, 76 repeated last (10.72 then 10.66, no drift):

| Slots | Budget (32K window) | Decode tok/s | Blocking misses | Prefetches |
| --- | --- | --- | --- | --- |
| 76 | 12G | 10.72 | 17,111 | 21,057 |
| 90 | ~13.6G | 12.84 | 10,503 | 10,539 |
| 100 | ~14.8G | 12.86 | 7,149 | 5,838 |
| 110 | ~15.9G | 14.68 | 4,667 | 2,831 |
| 128 | ~18G | 19.7 (short prompts) | 0 | 0 |

128 slots is above the 16 GB this 24 GB machine may plan and was not run on the
coding workload. A bare `--ram-budget 12G` gives the window priority and, with
no `--max-context`, plans 15 slots and a 262K window; `--max-context 32768` is
what gives 76.

### 3. Variant `Engines/Inkling2NVFP4`

- The same graph, copied (duplication is the 7.0 rule), with the NVFP4/MXFP8
  expert kernels and BF16 projections from `Engines/Inkling2`.
- Oracle: the checkpoint's own reference. The phase-1 work validated experts and
  layers against a Python reference (`d7183e4a`, `4afdedc1`); the variant's
  gate is greedy agreement with that reference on the parity prompts, plus the
  same golden, API and long-context checks. Where no independent oracle covers
  a path, say so in the qualification record rather than calling it verified.

### 4. Wiring

Selection table (`EngineFactories.gpuSelection` on family `inkling2` + scheme),
registry and catalog entries, `ModelProfile` sampling (the template's defaults),
the server reasoning profile (off / low / medium / high / xhigh), the Inkling
chat format and output parser moved onto the shared session's template path,
installer and launcher rows, OpenCode entries for both, golden targets in
`tools/golden-baseline.sh` and `release.sh`.

### 5. The comparison

Paired, interleaved rounds per the benchmarking guide, same machine and commit:
decode, prefill (64 / 512 / 8K / 27K), TTFT, resident memory at
`--ram-budget` 4 / 8 / 12 GB and unbounded, plus llama.cpp on the GGUF.
Reported as measurements with the protocol deviations named.

### 6. Retire the side routes

Remove `sources/TinyTitan/Inkling2GGUF/` (its decoder now lives in the
variant), `Inkling2ServerBackend` and the `Engines/Inkling2` generator path;
the server's GGUF sniffing goes. README, wiki Model-Support, release notes and
the handover say what 7.x actually serves.

## Risks and open questions

- **Expert streaming granularity.** GGUF stores `gate_up` and `down` as 3-D
  tensors per layer; the repacker must cut them per expert at block boundaries.
  A K-quant super-block is 256 values, which divides the expert row width; to
  be confirmed per tensor in phase 0.
- **The looped core and streaming.** Layers 6–17 run three times per token, so
  each pass re-reads that layer's experts; the slot cache sees three touches a
  token there. The prefetcher's layer order must follow the unrolled visits.
- **Memory today.** The machine was swapping (12 GB swap, 13 GB compressed)
  during the last measurements; phase 5 needs a quiet machine or its numbers
  are not comparable.
- **NVFP4 oracle.** There is no llama.cpp path for the safetensors checkpoint;
  the Python reference is the oracle and its coverage must be stated.

## Chat benchmark at `--ram-budget 12G`, 2026-11-24

Commit `fd29609c`, MacBook Pro M3 (Mac15,3) 24 GB, macOS 27.2.1, Swift 6.4.2, release
build. One server at a time on loopback, prompt "Compare Swift vs C++ in detail.",
`temperature 0`, `max_tokens 1024`, streamed; decode is measured by the client
from the first token to the last. Run 1 is cold (page cache and prompt cache
empty), runs 2–3 warm.

| Model / route | Plan | RSS | Out | TTFT warm | Decode tok/s (runs) |
|---|---|---|---|---|---|
| Inkling 2 Q4 `.ssdai`, `--ram-budget 12G` (default 262K window) | 15 slots, 1769 MiB cache | 10.0 GB | 380 | 4.35 s | 3.07 / 3.12 |
| Inkling 2 Q4 `.ssdai`, `--ram-budget 12G --max-context 32768` | 76 slots, 8963 MiB cache | 10.1 GB | 380 | 1.23 s | 12.30 / 12.48 / 12.34 |
| Inkling 2 NVFP4, raw-checkpoint side route | fixed 48 slots, budget ignored | 7.0 GB | 64 (route cap) | — | ≈1.5 (64 tok in 42.3 s wall) |
| Ornith 1.5 4-bit, `--ram-budget 12G` | 128 slots, 8.44 GB cache | 10.4 GB | 1024 | 1.00 s | 16.48 / 17.31 / 17.48 |

- The two Q4 plans produced byte-identical text; only the slot count differs.
  The default plan gives the 262K window priority and leaves 15 slots, which
  costs 4x decode at the same budget.
- The NVFP4 route does not stream (all 64 tokens arrive at the end), caps answers
  at 64 tokens (TT-118) and does not honour `--ram-budget`; its figure is wall
  time including prefill. The NVFP4 `.ssdai` install has no engine variant yet
  (phase 3), so this is the only way it runs today.

## Remaining work, 2026-11-24

Phases 0-2 are done (TT-121): both installs, the Q4 variant, streamed experts,
`--ram-budget`, the full API through `Inkling2GGUFServerBackend`. What is left is
for every TinyTitan feature and server parameter to apply to both Inklings, the
way they apply to Ornith. The work is split into tracker rows, in this order:

| Order | Row | Outcome | Gate |
|---|---|---|---|
| 1 | TT-128 | Every served route streams token by token | backend audit + a test that fails a buffering backend + streamed-chunks check |
| 2 | TT-122 | Inkling 2 Q4 on `ServerModelSession` through `variantGPUInkling2Q4()` | api-feature-check 10/10 through the shared session; greedy text identical to the current route at 16-bit KV; prompt-cache hit on a repeated prompt; `LibraryContractTests` on the install |
| 3 | TT-123 | `--kv-bits 4/8/16` honoured (8 is the server default) | 16-bit bit-identical; 8- and 4-bit measured against 16 on the parity prompts, kept only if they meet the quality bar |
| 4 | TT-125 | `Engines/Inkling2NVFP4` over its `.ssdai`, on the shared session | greedy agreement with `Engines/Inkling2`, per-layer parity, api-feature-check 10/10 at 12G |
| 5 | TT-124 | `--max-concurrent-sequences` for both | two concurrent requests equal two serial ones |
| 6 | TT-126 | Selection table, catalog, launcher, goldens, parameter matrix, budget default | `/v1/models` from `--models-dir`, `golden-baseline.sh --check` for both, every flag applies or is refused by name |
| 7 | TT-127 | Side routes deleted | no source references them; raw folders proposed to the owner |
| 8 | TT-129 | MTP speculative decode from the NVFP4 install's MTP head | greedy text identical with and without MTP, paired speed |

**Design of the shared-session step (TT-122).**

- `Engines/Inkling2GGUFQ4` gains a `Model: EngineModel` and a
  `Runner: EngineRunner` over `InklingQ4Decoder`. `prefillChunked` and
  `produce` call the decoder's existing prefill and decode, so numerics don't
  change. `captureInferenceState`/`restoreInferenceState` copy the per-layer K/V
  rings (to the live length), the convolution states and the position.
- `ArchConfig` is built from the install's `arch` block. The session only
  reads `vocabSize`, `numExperts`, `finalLogitSoftcap`, `sparseIndexer` and `ple`.
- The tokenizer becomes a `GFTokenizer` with an Inkling dialect, following the
  K2 precedent:
  - `applyChatTemplate` renders `Inkling2GGUFChatFormat`;
  - `StructuredAssistantDecoder` learns the thinking and
    `content_invoke_tool_json` markers;
  - stop tokens are `<|end_message|>` and `<|content_model_end_sampling|>`.
- The Q4 install carries its own `tokenizer/`, exported from the GGUF header at
  install time and covered by the receipt. Today it is written on first load to
  a cache under Application Support.
- `variantGPUInkling2Q4()` in `TinyTitanVariants` supplies `prepareLoad` from
  `InklingQ4LoadPlan`, so `--ram-budget`, `--expert-cache-slots` and
  `--max-context` keep their current meaning.

**Parameter matrix (TT-126).** Every `TinyTitanServer` flag either applies to
both Inklings or is refused with the reason. No flag is accepted and ignored,
which is the defect this work closes (`--ram-budget` was, and `--kv-bits`
still is). The expected refusals:

- `--host-kv-window`: needs a QSA model;
- `--cpu`: dense Qwen only;
- `--rope-scaling yarn`: only if the model card gives no YaRN factor.

**Quality bar.** As everywhere: no step may change greedy text or logits beyond
the measured noise floor. Speed figures are paired against the current default.
Model runs stay under the owner's 16 GB cap.

### TT-122 and TT-123 as built, 2026-11-24

- **Shared session.** `variantGPUInkling2Q4()` (`TinyTitanVariants`) is the
  install's `EngineFactory`; the selection paths (`gpuFactory(directoryURL:)`,
  `routingFactory(directoryURL:)`) return it for an Inkling 2 Q4 install. Its
  pieces:
  - `InklingQ4EngineModel` (the `EngineModel`);
  - `InklingQ4Runner` (the `EngineRunner`, over the decoder's own prefill and
    step);
  - `runInklingQ4Completion` (the completion loop: GPU argmax when greedy, the
    copied `InklingQ4Sampler` otherwise);
  - the variant's `InklingQ4LoadPlan`, so `--ram-budget`, `--expert-cache-slots`
    and `--max-context` keep their meaning.

  The session's window is now `min(asked, runner.maxContext)`, for an engine
  whose plan grants less than was asked.
- **Tokenizer dialect.** `TokenizerDialect.inkling` follows the K2 precedent:
  - `InklingChatTemplate` (moved from the 6.x route into Core, one renderer)
    renders the prompt;
  - `InklingAssistantParser` (moved likewise) runs inside
    `StructuredAssistantDecoder`;
  - the turn stops on `<|content_model_end_sampling|>` or on a turn the model
    must not write;
  - a JSON-schema answer opens the text message itself and ends on
    `<|end_message|>` (`documentEndTokenIDs`).
- **Revised: the tokenizer is not in the install.** The plan said the Q4
  install would carry `tokenizer/`. Writing into an existing install changes it,
  so instead the factory's `tokenizerFolder` exports it once from the verified
  GGUF header into the user cache directory, as the old route did, keyed by
  model id, snapshot hash and header size.
- **Prompt cache.** Snapshots hold each visit's written K/V rows (and scales
  below 16 bits) plus the convolution history; the direct prefix hit is
  dialect-free. Interior checkpoints locate messages through
  `PromptMessageMarkers` (ChatML: `<|im_start|>`, the last start being the
  generation header; Inkling: the four role tokens, with a header only when
  the prompt opened the text message). The ChatML continuation bridges are
  skipped for Inkling.
- **`--kv-bits` (TT-123): built, measured, rejected.**
  - 8- and 4-bit K/V kernels (quantize on write in `kv_finish`/`kv_finish_chunk`,
    dequantize in both relative-attention kernels, symmetric absmax) were run
    on the llama.cpp reference cases: 96 teacher-forced tokens, 16 slots.
  - Top-1 agreement with llama.cpp, largest logit difference from 16 bits:

    | K/V | one scale per head | one scale per 32 channels |
    |---|---|---|
    | 16-bit | 96/96 | 96/96 |
    | 8-bit | 94/96, 1.96 | 95/96, 1.46 |
    | 4-bit | 90/96, 6.42 | 91/96, 3.32 |

  - A memory saving that changes a token is rejected, so the kernels were not
    committed. Inkling keeps 16-bit K/V: an explicit `--kv-bits 8/4` is refused
    with these numbers, and the server's default of 8 is served at 16 with a
    logged notice rather than ignored silently.

### TT-125: the NVFP4 install against the Q4 graph, measured 2026-11-24

The NVFP4 plugin runs the Q4 variant's graph (the same llama.cpp-derived
decoder) over the checkpoint install. What that takes was established
tensor by tensor from the two installs on disk, which hold the same weights
(`scratchpad` scripts `tensor_map.py`, `expert_repack.py`):

- **Resident tensors.** All 646 tensors the Q4 graph reads map to checkpoint
  tensors.
  - *Identity copies:* every projection, norm, clip, router weight and bias,
    controller, convolution and residual query.
  - *Value types:* BF16 in the checkpoint, F32 or Q8_0 in the GGUF.
  - *Transforms*, each verified on every layer:
    - `attn_rel_bias` = `rel_logits_proj.proj` transposed;
    - in the four dynamic-convolution layers (5, 11, 17, 23), `attn_{k,v}_conv`
      = `{k,v}_sconv.static_w` transposed with its four taps reversed (exact,
      error 0);
    - `blk.L.attn_res_query` = `attn_res.attn_queries.(L-1)`;
    - `blk.L.ffn_res_query` = `attn_res.mlp_queries.L`;
    - the globals map by name (`embed`, `unembed`, `norm`, `embed_norm`,
      `attn_res.output_query`, `middle_cycle_cross_pass_{,kv_}queries`).
- **NVFP4 experts (layers 0–19): byte-exact.** A planar checkpoint expert
  repacks into the GGUF NVFP4 blocks the Q4 kernels read, and every byte is
  identical to the Q4 install's (9 experts in 3 layers tested; `down` and
  `gate_up`). The repack:
  - a block of 64 values is 4 E4M3 scale bytes, then per 16-value group
    8 bytes holding value j in the low nibble and value j+8 in the high
    nibble;
  - `gate_up` rows go from the checkpoint's interleaved (gate, up, gate, …)
    order to concatenated (all gate, then all up);
  - the per-expert scale is `amax / outer_scale_denominator`, equal to the Q4
    install's scale to 1e-9.

  So these layers compute the same bits as the Q4 route.
- **MXFP8 experts (layers 20–23)** have no exact GGUF form (the GGUF re-quantized
  them to Q8_0). They need their own expert kernels: an MXFP8 GEMV-ids and a
  GEMM-tiled decoder. The repack writes a 34-byte block per 32 values (E8M0 scale,
  one pad byte, 32 E4M3 bytes), which keeps the Q8_0 stride, so slot and
  budget arithmetic is unchanged.
- **Kernels the Q4 graph lacks:** the MXFP8 expert pair above, and a BF16
  embedding row (the Q4 graph's embedding is Q8_0 only). BF16 projections and
  the BF16 head already run (`dense_gemv_f32_bf16`, the tiled GEMM).
- **Oracle.** `Engines/Inkling2` (the raw-checkpoint route) runs the
  checkpoint's own arithmetic; the plugin's gate is greedy agreement with it
  on the parity prompts.
