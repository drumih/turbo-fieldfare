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
Remaining: streaming both sources from Hugging Face (registered, pinned
sources; ranged header fetch for the GGUF).

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
