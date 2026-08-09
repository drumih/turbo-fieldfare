<p align="center">
  <img src="docs/assets/turbofieldfare-logo-rounded.png" alt="TurboFieldfare logo: a fieldfare inside a segmented cache ring" width="280">
</p>

<h1 align="center">TurboFieldfare</h1>

<p align="center">
  <strong>Gemma 4 26B-A4B in about 2 GB of RAM — Kimi K3 (2.78T MoE) streamed from SSD on a 128 GB Mac</strong><br>
  A custom Swift + Metal runtime that scales from 8 GB Apple Silicon laptops to 128 GB workstations.
</p>

<p align="center">
  <img alt="Swift 6.2" src="https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white">
  <img alt="Metal 4" src="https://img.shields.io/badge/Metal-4-5E5CE6">
  <img alt="macOS 26 or later" src="https://img.shields.io/badge/macOS-26%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache%202.0-2ea44f"></a>
</p>

<p align="center">
  <a href="#try-it">Quick start</a> ·
  <a href="#kimi-k3-gturbo-v2">Kimi K3</a> ·
  <a href="docs/OPENAI_SERVER.md">Local server</a> ·
  <a href="docs/BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/COMMUNITY_BENCHMARKS.md">Contribute results</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="docs/OPTIMIZATION_JOURNEY.md">Experiments</a> ·
  <a href="docs/IMPLEMENTATION_REFERENCES.md">References</a> ·
  <a href="#about-this-fork">About this fork</a>
</p>

![TurboFieldfare Mac app generating text with Gemma 4 26B-A4B](docs/assets/turbofieldfare-app.webp)

Memory got expensive. So I gave a 26-billion-parameter model a ~2 GB budget —
and then gave a 2.78-trillion-parameter model a 128 GB one.

TurboFieldfare runs the instruction-tuned
**[Gemma 4 26B-A4B](https://ai.google.dev/gemma/docs/core/model_card_4)**
without loading the entire 14.3 GB model into memory. It keeps the shared
1.35 GB core and FP16 KV cache in memory, then streams only the experts needed
for each token from SSD. This is what lets the model run on Macs with 8 GB of
RAM.

The same streaming approach also runs Moonshot AI's **[Kimi K3](docs/KIMI_K3_EVALUATION.md)**
(2.78T total parameters, ~104B active per token, 93 layers of Kimi Delta
Attention and NoPE-MLA over a 896-expert LatentMoE) on a 128 GB Apple Silicon
Mac. The ~32 GB quantized trunk stays resident; the 1.447 TB of MXFP4 routed
experts stream from SSD on demand, at a measured 0.512 tok/s on a real
checkpoint — a batch/research pace, not interactive chat, for a model too
large for any consumer GPU rig to load at all. See
[Kimi K3](#kimi-k3-gturbo-v2) below.

The runtime, streaming installer, CLI, and native Mac app are written in Swift
and Metal. TurboFieldfare is model-specific rather than a wrapper around MLX or
llama.cpp. The curated [experiment record](docs/experiments/EXPERIMENT_INVENTORY.md)
summarizes 103 measured results across kernels, caching, I/O, prefill, and
decode.

## Try it

```bash
git clone https://github.com/zephyranthes03/turbo-fieldfare-kimi-k3.git
cd turbo-fieldfare-kimi-k3
swift build -c release
.build/release/TurboFieldfareMac
```

On the first run, Swift Package Manager downloads and builds the Swift packages
required by the tokenizer. The complete release build includes the foreground
Mac app and its sibling decode-service executable.

When the app opens, choose **Download** and let TurboFieldfare fetch and repack
the pinned model (about 15 GB). Once it is ready, choose **Load Model**, type
your prompt, and press **Generate**.

### Kimi K3 quick start

K3 has no Mac app yet — it needs a 128 GB Mac and runs through the CLI or
server:

```bash
swift run -c release TurboFieldfareRepack \
  --model-family kimi-k3 --output scratch/kimi-k3.gturbo
# ~1.5 TB, resumable: rerun with --resume if interrupted

swift run -c release TurboFieldfareCLI \
  --model scratch/kimi-k3.gturbo \
  --prompt "The capital of France is" --max-new 64
```

See [Kimi K3](#kimi-k3-gturbo-v2) below for the full flag set, disk-runtime
controls, and a measured real-checkpoint run.

## At a glance

| Metric          | Value                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Model           | Gemma 4 26B-A4B IT, 26B total parameters, about 3.88B active per token                                                   |
| Weights         | MLX affine 4-bit, group 64; 8-bit router; 4-bit shared and routed experts                                                |
| Memory          | ~2 GB of weights and 4K KV cache                                                                                         |
| Storage         | About 14.3 GB for the installed text-only model                                                                          |
| Hardware        | Apple Silicon Mac; 8 GB of RAM                                                                                            |
| Platform        | macOS 26, Metal 4, Swift 6.2                                                                                             |
| M2 measured decode | [5.1-6.3 tok/s](docs/BENCHMARKS.md#m2-measured-decode) on an 8 GB M2 MacBook Air |
| M5 measured decode | [31-35 tok/s](docs/BENCHMARKS.md#m5-measured-decode) on a 24 GB M5 Pro |
| Community Reports | [Here](docs/COMMUNITY_BENCHMARKS.md#community-results) |

The measured result is a reference point, not a performance ceiling. Prompt
length, generated length, page-cache state, and hardware all affect throughput.
See [community benchmark results](docs/COMMUNITY_BENCHMARKS.md#community-results)
from other Macs, or follow the
[community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) to add your own.

### Kimi K3 at a glance

| Metric          | Value                                                                                    |
| --------------- | ----------------------------------------------------------------------------------------- |
| Model           | Kimi K3, 2.78T total parameters, about 104.2B active per token, 93 layers                 |
| Weights         | Native MXFP4 routed experts (verbatim, no re-quant); int4/int8 affine quantized trunk      |
| Memory          | ~38 GB wired (int4 trunk profile) plus an optional 0–64 GiB direct-resident expert cache   |
| Storage         | About 1.5 TB for the installed model (1.447 TB of routed experts + quantized trunk)        |
| Hardware        | Apple Silicon Mac; 128 GB of RAM                                                           |
| Platform        | macOS 26, Metal 4, Swift 6.2                                                               |
| M5 Max measured decode | [0.512 tok/s](docs/K3_DISK_RUNTIME.md#measured-example-real-checkpoint) on the real checkpoint, 262144-token context |

K3 is a batch/research runtime: decode is SSD-bound by the 25.8 GB of routed
experts read per token, not interactive-chat speed. See
[Kimi K3](#kimi-k3-gturbo-v2) below for the full picture.

## Using TurboFieldfare

TurboFieldfare provides a native Mac app, a command-line interface, and an
experimental loopback OpenAI-compatible server. They use the same `.gturbo`
model directory, but only one model-owning product should run at a time.

The Swift package exposes six products:

| Product | Purpose |
| --- | --- |
| `TurboFieldfare` | Swift library containing the runtime and Metal kernels |
| `TurboFieldfareMac` | Native Mac app for installation and generation |
| `TurboFieldfareDecodeService` | One-shot local model and Metal owner used by the Mac app |
| `TurboFieldfareCLI` | Command-line instruction chat and raw completion |
| `TurboFieldfareServer` | Loopback OpenAI-compatible Chat Completions server |
| `TurboFieldfareRepack` | Streaming model installer and install verifier |

### Requirements

- An Apple Silicon Mac; the validated target is an 8 GB M2 MacBook Air
- macOS 26 with Metal 4
- Xcode 26, or a swift.org 6.3+ toolchain on `PATH` if Xcode isn't installed
- Enough free storage for the ~14.3 GB model installation
- An internet connection for the first model install

The package is arm64-only. Older macOS and Metal versions are not supported.
Without Xcode, `swift build`/`swift test` may fail to compile the package
manifest or `OutputPaneView.swift`; see [AGENTS.md](AGENTS.md#test-rules) for
the toolchain fix.

### Prompting the model

The Mac app treats what you type as an instruction and handles Gemma's chat
formatting automatically. Just describe the task and include any context the
model needs.

Generation defaults to temperature `0.2`, Top-K `64`, and Top-P `0.95`. Set
temperature to `0` for deterministic greedy output. The model can still repeat
itself or give incorrect answers, so check important results.

TurboFieldfare is text-only. The app and CLI support user and model messages
plus optional system guidance; they do not expose or execute tools. The
loopback server accepts function-tool declarations and returns
model-produced tool calls for the client to authorize and execute. Images,
audio, and video are not supported.

### Mac app

Clone the repository, then run the app from its root:

```bash
swift build -c release
.build/release/TurboFieldfareMac
```

Build the complete package so the app and its sibling decode service are both
available. When launched from this checkout, the app stores the model in
`scratch/gemma4.gturbo`.

#### Install the model

On first launch, the app checks the available storage and shows the download
and installed sizes. Choose **Download** to begin.

The installer never materializes the full source checkpoint. It streams the
required byte ranges from the pinned Hugging Face revision and repacks them
directly into the `.gturbo` layout as they arrive. This avoids a second full
checkpoint on disk and keeps scratch memory bounded.

The first installation transfers about 15 GB through bounded Hugging Face
range requests. Network speed and Hugging Face response times vary, so it can
take a while. The completed `.gturbo` installation occupies about 14.3 GB and
is accepted only after its manifest and file hashes have been validated.
Installation does not load the model into memory.

#### Load and generate

After installation:

1. Choose **Load Model**.
2. Enter a prompt in the composer.
3. Choose **Generate**, or press <kbd>Command</kbd>+<kbd>Return</kbd>. Use **Settings > Send Message With** to choose Return or Command-Return.
4. Use the stop button or <kbd>Escape</kbd> to end generation early.

The status bar shows generation progress, decode speed, and memory use. Use the
right pane to configure sampling, context length, expert-cache slots, and
runtime options. See [Runtime controls](docs/RUNTIME_CONTROLS.md) for details
and defaults.

### Command-line interface

The CLI uses an existing `.gturbo` installation. If you installed the model
through the Mac app, it is already available at `scratch/gemma4.gturbo`.
Otherwise, install it from the command line:

```bash
swift run -c release TurboFieldfareRepack \
  --output scratch/gemma4.gturbo \
  --overwrite
```

Continue a cancelled or interrupted download:

```bash
swift run -c release TurboFieldfareRepack \
  --output scratch/gemma4.gturbo \
  --overwrite \
  --resume
```

Remove saved download state:

```bash
swift run -c release TurboFieldfareRepack \
  --discard-partial \
  --output scratch/gemma4.gturbo
```

The runtime accepts only a completed `.gturbo` directory with a final
`manifest.json`.

Verify an existing installation without loading the model:

```bash
swift run -c release TurboFieldfareRepack \
  --verify-install \
  --input-gturbo scratch/gemma4.gturbo
```

#### Instruction chat

Put chat messages in a JSON array and pass it with `--messages-file`:

```json
[
  {"role": "user", "content": "Explain why chunked prefill reduces time to first token while keeping memory bounded."}
]
```

```bash
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --messages-file messages.json
```

This formats messages in the same way as the Mac app. The CLI response limit
is set with `--max-new`, which defaults to 1,024 tokens. The Mac app can
generate until the selected context window is full.

Common generation options include `--max-context`, `--temperature`, `--top-k`,
`--top-p`, `--repetition-penalty`, `--seed`, and repeatable `--stop` strings.
The public CLI uses production runtime defaults. Run the following command for
the complete option list:

```bash
swift run -c release TurboFieldfareCLI --help
```

Generated text goes to standard output. Timing statistics go to standard error;
add `--quiet` to suppress that footer in scripts.

### Local OpenAI-compatible server

Build the server and point it at an installed model:

```bash
swift build -c release --product TurboFieldfareServer
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo
```

It listens on `http://127.0.0.1:8080/v1` and supports Chat Completions,
streaming, function tools, and single-prefix prompt reuse. The client must
authorize and run every tool call. Keep the server on loopback; it has no
remote authentication or TLS.

See [Local server](docs/OPENAI_SERVER.md) for a test request, Python and
OpenCode setup, prompt reuse, tool handling, and the supported API subset.

## Test and contribute

Run the public test suite serially:

```bash
Scripts/test.sh
```

Before starting a model run, close memory-heavy apps and check
`memory_pressure -Q`. If it reports little free memory, postpone the run. Run
only one TurboFieldfare app, decode service, CLI, server, test, or other
local-model process at a time.

To contribute a comparable performance result, follow the
[community benchmark guide](docs/COMMUNITY_BENCHMARKS.md).

## How the inference engine works

At each transformer layer, Metal computes attention and the router from
resident weights. The CPU uses the router's top-8 expert IDs to plan against
the layer's 16-slot LFU cache, then fills misses with bounded parallel `pread`
calls into Metal-visible buffers. Metal computes the resident shared-expert
branch while those reads run, then combines the shared and routed outputs.

Prompt prefill uses chunks of up to 128 tokens so one fetched expert can serve
multiple rows. Generation repeats the routed layer loop one token at a time.
The installer applies the same bounded-memory rule: it repacks remote ranges
directly into `.gturbo` without staging a full shard or tensor.

For a video overview of TurboFieldfare, see Better Stack's
[Local AI On Apple Silicon uses 7X Less RAM](https://youtu.be/vHhephsP6vU).

For a visual introduction to the model architecture, see Maarten Grootendorst's
[A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4).

[System design](docs/SYSTEM_DESIGN.md) explains the `.gturbo` layout, memory
ownership, prefill, router handoff, `cb1`/`io`/`cb2` phases, Metal kernels, and
correctness invariants.

## Status and scope

TurboFieldfare currently includes:

- Remote streaming repack into the `.gturbo` model format
- Instruction-tuned Gemma 4 26B-A4B with verified text-only chat formatting
- 4-bit MLX affine embedding, attention, shared-expert, and routed-expert
  weights, with an 8-bit router
- Custom Metal kernels for quantized GEMV, attention, MoE, normalization,
  RoPE, sampling, and production fusions
- SSD-backed routed-expert streaming with optional direct-resident layer banks
- Chunked single-prompt prefill and token-by-token generation
- FP16 KV storage with bounded circular storage for 25 sliding-window layers
  and linear storage for 5 full-attention layers
- Exact split-K/V decode attention with distinct normalized K and V paths
- A Swift library, streaming installer, command-line interface, loopback
  OpenAI-compatible server, and native SwiftUI/AppKit Mac app with a one-shot
  local decode service

Current scope is text-only inference from the pinned Gemma 4 26B-A4B
instruction checkpoint on Apple Silicon Macs with at least 8 GB of RAM, plus
the second engine below for Macs with 128 GB of RAM.

### Kimi K3 (`.gturbo` v2)

This fork adds a second engine: **Kimi K3** (2.78T-parameter MoE — 69 Kimi
Delta Attention layers + 24 NoPE MLA layers, 896 experts with top-16 routing
through a 3584-wide latent bottleneck, Attention Residuals) running on a
128 GB Apple Silicon Mac. The 1.447 TB of routed experts stay in their native
MXFP4 format on SSD and stream through bounded, throughput-tuned,
whole-expert `pread` jobs (Darwin `F_NOCACHE` in the production K3 auto
profile). Temporal expert prediction remains an explicit experiment and is
off by default; the ~32 GB trunk (int4/int8 affine) stays
resident; prefill uses the M5 Neural Accelerator through Metal 4 tensor ops.
Expect roughly 0.3–1 tok/s decode — a batch/research runtime, not interactive
chat. The server can retain one exact K3 prefix by snapshotting its recurrent
and MLA state; cache hits are reported through `cached_tokens`. Read
[docs/KIMI_K3_EVALUATION.md](docs/KIMI_K3_EVALUATION.md) for the
feasibility math and the MLX-vs-GGUF evaluation,
[docs/K3_DATAFLOW.md](docs/K3_DATAFLOW.md) for the exact architecture
contract, and `AGENTS.md` for the repack/CLI/server commands.

K3 defaults differ from Gemma: greedy sampling, a 64-token CLI response cap,
32-token chunked prefill, on-demand whole-expert reads, and an online worker
choice among 1/2/4. `--expert-predict on` keeps the former temporal-prefetch
experiment available for measured A/B runs. `--expert-predict selective`
limits speculative reads to four experts stable across two tokens;
`--expert-cache-gib N` instead slices one aligned arena into exact 16-slot
resident RAM banks (24 GiB covers 91 of 92 MoE layers; drop to 16 GiB at the
262144-token context ceiling, where the MLA cache and int8 trunk leave less
headroom — a real-checkpoint 8/16/24 GiB A/B at that context confirmed 16 GiB
as the sweet spot: 24 GiB reads the least from SSD but runs ~51% slower under
the extra memory pressure, see
[docs/K3_DISK_RUNTIME.md](docs/K3_DISK_RUNTIME.md#cache-size-sweep-8--16--24-gib-real-checkpoint-262144-context)).
Only the currently executing layer is mapped to
Metal; hits stay in place and misses read directly into it, so neither path
performs a second RAM copy. Override the operational
controls with `--temperature`, `--max-new`, `--prefill-chunk`,
`--expert-io-splits`, `--expert-io-workers`, and `--expert-io-cache`. A
completed bundle with `verified-install.json` may use
`--model-verification trusted-install` to skip per-layer 15.7 GB hashes; the
default remains `full-sha256`. A real-checkpoint run with this profile
measured 0.512 tok/s decode — see
[docs/K3_DISK_RUNTIME.md](docs/K3_DISK_RUNTIME.md#measured-example-real-checkpoint).

The int8 trunk needs a Metal wired-memory ceiling above the macOS post-reboot
default seen on the validated 128 GB M5 Max. Check
`sysctl iogpu.wired_limit_mb` before loading it; a 32 GiB value is below the
57.8 GiB resident file. See [K3 disk runtime](docs/K3_DISK_RUNTIME.md) for the
temporary 80 GiB setup and its system-memory tradeoff.

For offline queues, `--batch-file jobs.jsonl` loads K3 once, resets sequence
state per row, and retains the exact expert banks across rows. Each JSONL row
contains either `{"id":"job-1","prompt":"...","max_new":64}` or an OpenAI-like
`messages` array. Output is one result JSON object per line. This is sequential
batch throughput, not approximate multi-sequence decoding.

Prepared multi-SSD expert stripes can be supplied by repeating
`--expert-shard-root`. Every root carries an `expert-shard.json`; within each
layer the runtime maps logical experts directly to per-shard file descriptors,
so the top-16 reads can span disks. The original `.gturbo` v2 bundle remains
unchanged. See [K3 disk runtime](docs/K3_DISK_RUNTIME.md) for the contract and
current limitations.

To move an existing verified K3 bundle between the int4 and int8 trunk
profiles without allocating or downloading a second 1.447 TB expert payload:

```bash
swift run -c release TurboFieldfareRepack \
  --output scratch/kimi-k3.gturbo \
  --model-family kimi-k3 \
  --trunk-quant int8 \
  --reuse-existing-experts
```

This mode requires APFS cloning. It validates the existing manifest and
`verified-install.json`, clones the immutable MXFP4 layer files copy-on-write,
downloads only the BF16 trunk ranges, validates the completed sibling bundle,
then uses an atomic directory swap. Cancellation before promotion keeps the
old bundle intact; resume with the same command plus `--resume`. There is no
byte-copy fallback when cloning is unavailable because that would invalidate
the bounded disk-space guarantee.

### Future work

- Build iPhone and iPad apps, then measure inference speed and memory use on
  mobile hardware.
- Benchmark more Apple Silicon Macs, especially the base 16 GB M4 Mac mini and
  other 8 GB models.

## Experiments and technical documentation

The [experiments that shaped TurboFieldfare](docs/OPTIMIZATION_JOURNEY.md)
explain the largest wins, the plausible ideas that failed, and the early
results that reversed under stronger validation. The detailed
[experiment record](docs/experiments/EXPERIMENT_INVENTORY.md) keeps all 103
audited entries as optional evidence.

Useful entry points:

- [Local OpenAI-compatible server](docs/OPENAI_SERVER.md)
- [System design](docs/SYSTEM_DESIGN.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [The experiments that shaped TurboFieldfare](docs/OPTIMIZATION_JOURNEY.md)
- [Experiment inventory and summaries](docs/experiments/EXPERIMENT_INVENTORY.md)
- [Implementation references](docs/IMPLEMENTATION_REFERENCES.md)

## License and model terms

TurboFieldfare's source and documentation are licensed under the
[Apache License 2.0](LICENSE).

Model weights are not included. The installer downloads them separately from
the pinned Hugging Face checkpoint, and the weights remain governed by their
source terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the model
and Swift package license review.

TurboFieldfare is an independent research project. It is not affiliated with,
sponsored by, or endorsed by Google.

## About this fork

This repository is a fork of Andrey Mikhaylov's
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare), which designed
and built the original ~2 GB Gemma 4 26B-A4B runtime, the `.gturbo` v1 format,
the streaming installer, and the native Mac app — all credit for that work,
and the dedication below, is his (see
[Afterword and the project name](#afterword-and-the-project-name)).

This fork adds a second, independent inference engine: **Kimi K3**
(Moonshot AI's 2.78T-parameter MoE, one of the largest openly released
checkpoints) on a 128 GB Apple Silicon Mac. The approach mirrors the
original project's philosophy — a model-specific runtime rather than a
wrapper around MLX or llama.cpp — but at a different point in the
memory/compute tradeoff: instead of fitting a whole model into ~2 GB, it
keeps a ~32 GB trunk resident and treats a 1.447 TB pool of MXFP4 experts on
SSD as the working set, since neither MLX nor llama.cpp can load K3 at all in
128 GB as of this writing.

The work was grounded in the actual target machine rather than an assumed
spec: `sysctl`/`sw_vers` confirmed the development machine as an M5 Max with
128 GB unified memory, macOS 26.6, and Swift 6.3 before any design math was
written down, so the memory budget and feasibility estimates in
[docs/KIMI_K3_EVALUATION.md](docs/KIMI_K3_EVALUATION.md) are calibrated to
real hardware. Two scope calls were made deliberately up front rather than
discovered by accident: use the official full-fidelity MXFP4 checkpoint
(1.56 TB total) instead of a pruned or heavily requantized alternative, and
validate the engine against synthetic tiny-K3 fixtures and a CPU-reference
oracle *before* any real-checkpoint run, so correctness bugs surface in
seconds against known-good outputs rather than after a multi-hour cold-cache
decode. The measured real-checkpoint numbers elsewhere in this README came
after that synthetic validation was green.

The K3 work, in scope order:

- **Feasibility research before code**: format evaluation (MXFP4-native
  experts vs. re-quantizing to GGUF k-quants), a byte/bandwidth budget for
  SSD-bound decode, and a literature/prior-art review of two reference
  projects (`flash-moe`, `kimi-k3-in-c`) to borrow measured techniques instead
  of guessing — written up in
  [docs/KIMI_K3_EVALUATION.md](docs/KIMI_K3_EVALUATION.md).
- **A `.gturbo` v2 wire format and repacker** covering Kimi Delta Attention,
  NoPE-MLA, Attention Residuals, and 896-expert Stable LatentMoE, streaming the
  ~1.5 TB checkpoint into place without staging a second full copy, plus an
  int4-to-int8 trunk conversion path that reuses cloned expert files instead of
  re-downloading them.
- **An SSD-backed expert-streaming engine**: bounded `pread` scheduling with
  an online worker-count search, page-aligned split reads, `F_NOCACHE` vs.
  buffered I/O policy, an opt-in exact direct-resident RAM bank cache with no
  cache-to-compute copies, and multi-SSD expert sharding — detailed in
  [docs/K3_DATAFLOW.md](docs/K3_DATAFLOW.md) and
  [docs/K3_DISK_RUNTIME.md](docs/K3_DISK_RUNTIME.md).
- **CLI, server, and batch integration**: chunked NAX-accelerated prefill,
  reasoning-effort and tool-call handling, single-prefix recurrent-state reuse
  for the server, and a JSONL batch mode that amortizes model load across
  many jobs.
- **Validation**: synthetic tiny-K3 fixtures and CPU-reference kernel
  cross-checks for every change, plus a real-checkpoint run recorded end to
  end (0.512 tok/s decode, 262144-token context — see
  [docs/K3_DISK_RUNTIME.md#measured-example-real-checkpoint](docs/K3_DISK_RUNTIME.md#measured-example-real-checkpoint)),
  so the numbers throughout this README are measured, not projected.

Every decision above that trades one thing for another (on-demand reads vs.
prediction, int4 vs. int8 trunk, cache sizing at large contexts) is written
down with the measurement that motivated it, not asserted — that discipline
carries over from the original project's own
[experiment record](docs/experiments/EXPERIMENT_INVENTORY.md).

This fork's Kimi K3 engine — design, implementation, and documentation — was
done by Yongjin Chong
([LinkedIn](https://www.linkedin.com/in/yongjin-chong-482610146/)).

## Afterword and the project name

Thanks for checking out this project!

My name is Andrey Mikhaylov. You can find me on
[LinkedIn](https://www.linkedin.com/in/andrey-mikhaylov-ios-dev/).
I am the author of TurboFieldfare and an iOS and Metal engineer. Most of my
work is with images, video, and on-device AI.

I dedicate this project to my wife, Sasha, the most supportive person I know.
She stands by me even through the hardest times. She loves wildlife, goes
birdwatching, and volunteers with our local birding community. Because of her,
I have also grown closer to birds and nature.

TurboFieldfare is named after the fieldfare, a member of the thrush family and
my favourite bird. It is not the most noticeable or brightly coloured bird, but
it definitely has a character and unique features of its own. I think the same
is true of this project: it may not be the most practical, but I built it with
my favourite tools, especially Metal, in my favourite field, on-device ML
inference. It definitely has its own character and unique features.

Next time you are outside, touch the grass and listen to the birds. Sometimes
it is the most beautiful thing you can do. And if you can, support your local
wildlife community. They do important work.

Thank you!
