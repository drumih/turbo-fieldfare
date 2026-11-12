<p align="center">
  <img width="1254" height="1254" alt="TinyTitan" src="https://github.com/user-attachments/assets/2a990de5-af24-48cd-b673-e875755741cc" />
</p>

# TinyTitan

[![Stars](https://img.shields.io/github/stars/Pummelchen/TinyTitan?style=flat-square&logo=github&label=Stars&color=e3b341)](https://github.com/Pummelchen/TinyTitan/stargazers)
[![Views (14d)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Pummelchen/TinyTitan/main/.github/traffic.json)](https://github.com/Pummelchen/TinyTitan)
[![Last Commit](https://img.shields.io/github/last-commit/Pummelchen/TinyTitan?style=flat-square&logo=git&label=Last%20Commit&color=2ea44f)](https://github.com/Pummelchen/TinyTitan/commits/main)

**TinyTitan runs Qwen-family MoE and dense text models, and Poolside's
Laguna-S-2.1, locally on Apple Silicon, streaming the routed experts from SSD
so a model larger than your RAM still runs.** Decode is native Swift over hand-written Metal kernels; the prompt runs
on the GPU or, optionally, the Apple Neural Engine. It ships a CLI, an
installer/repacker, and a loopback OpenAI-compatible server with no MLX and no
GGUF dependency.

**Who it is for.** People on an M-series Mac who want to run a 35B — or a
122B/125B — model locally on a machine that could never hold it in RAM, and
developers who want a loopback OpenAI- or Anthropic-compatible endpoint for
Codex, Claude Code, Qwen Code, OpenCode, Zed or the official SDKs. It is a
terminal-first tool: the engine and its server are the product, and the
DeepSeek Harness plugins are an optional client of that server. It is **not** a
fine-tuning toolkit, a vision model, a GUI app, or a way to expose a model to
your network — the server is `127.0.0.1`-only and has no authentication.

## What this repository delivers

1. **The LLM engine as a library — [`TinyTitanLib`](sources/TinyTitanLib).** Embed
   the engine in your own Swift program: an `Engine` and a `Session`, streaming
   tokens through a callback. Same kernels, format reader and sampler as the
   engine, in your process — no subprocess, no HTTP, no model reimplementation.
2. **The complete local LLM engine — `TinyTitanCLI` + `TinyTitanServer`.** Run a
   35B, 122B or 125B model from the terminal and serve it on a loopback
   OpenAI-/Anthropic-compatible endpoint for Codex, Claude Code, Qwen Code,
   OpenCode, Zed or the official SDKs. Native Swift and Metal: no MLX, no GGUF
   library. Experimental: the Unsloth Qwen3.8-Flash-Next UD-IQ3_XXS GGUF
   shards also run as-is through `--model` (CLI only, same top-1/top-5 as
   llama.cpp, decode at the converted model's speed — no conversion needed,
   no speed gain).
3. **The DeepSeek Harness plugin — [`plugins/dsh-tinytitan`](plugins/dsh-tinytitan).**
   The harness finds the models you have installed and keeps its route current,
   with a quiet compaction backend.
4. **The `.ssdai` model format and its converter — `TinyTitanRepack` +
   [`tools/`](tools).** Stream a checkpoint into a hash-verified install that
   keeps the routed experts on SSD, so a model larger than your RAM still runs —
   and a truncated, swapped or edited install is refused rather than served.

Those four ship from one branch (`main`) and one release tag. The first two are
the **products** — a library and an engine — and the rest are the plugin and
tooling that ship beside them; the split is spelled out under
[Library and engine](https://github.com/Pummelchen/TinyTitan/wiki/Library-and-Engine).

## Quick start

Requires macOS 26+, Xcode 27 (Swift 6.4) and Apple Silicon.

```bash
git clone https://github.com/Pummelchen/TinyTitan.git
cd TinyTitan
swift build -c release
.build/release/TinyTitanCLI --help
```

Install a model with `tools/install_models.sh --choose` (the dense 2B row is
about 2.3 GB), then run a prompt:

```bash
.build/release/TinyTitanCLI --model models/qwen3.5_2B_4Bit \
  --prompt "What is the capital of Paris ?" --max-new 96
```

Start the server with `tools/server_launcher.sh`; it prints the base URL and
writes the client configuration. The full walkthrough is
[Getting Started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started).


## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB, after a warm-up pass.

| Model | Quantization | GPU | CPU | Terminal-Bench 2.x | Terminal-Bench 4.0 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.5 tok/s** | NA | 86% (2.1) | 25% |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.1 tok/s** | NA | 86% (2.1) | 25% |
| Ornith 1.5 122B | 4-bit | **2.9 tok/s** | NA | 81% (2.1) | NA |
| Ornith 1.5 122B | 8-bit | **1.2 tok/s** | NA | 81% (2.1) | NA |
| Laguna-S-2.1 118B-A8B | 4-bit | **1.5 tok/s** | NA | 70% (2.1) | NA |
| Laguna-S-2.1 118B-A8B | 8-bit | **0.7 tok/s** | NA | 70% (2.1) | NA |
| Ornith 1.5 35B-A3B | 4-bit | **21.7 tok/s** | NA | 68% (2.1) | NA |
| Ornith 1.5 35B-A3B | 8-bit | **11.9 tok/s** | NA | 68% (2.1) | NA |
| K2-Horizon-MoVA-36B-A4B | 4-bit | **12.4 tok/s** | NA | 59% (2.1) | NA |
| K2-Horizon-MoVA-36B-A4B | 8-bit | **2.0 tok/s** | NA | 59% (2.1) | NA |
| Qwen 3.6 35B-A3B | 4-bit | **21.4 tok/s** | NA | 53% (2.1) | NA |
| Qwen 3.6 35B-A3B | 8-bit | **12.4 tok/s** | NA | 53% (2.1) | NA |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B | 4-bit | **6.7 tok/s** | **3.6 tok/s** | 51% (2.1) | NA |
| NVIDIA Nemotron 3.5 Bolt 74B-A9B | 8-bit | **0.6 tok/s** | **0.9 tok/s** | 51% (2.1) | NA |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.7 tok/s** | NA | 40% (2.0) | NA |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.3 tok/s** | NA | 40% (2.0) | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 4-bit | **17.9 tok/s** | NA | 33% (2.1) | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 8-bit | **6.9 tok/s** | NA | 33% (2.1) | NA |
| Qwen 3.5 4B (dense) | 4-bit | **22.8 tok/s** | **7.5 tok/s** | 26% (2.1) | NA |
| Qwen 3.5 4B (dense) | 8-bit | **14.1 tok/s** | **7.9 tok/s** | 26% (2.1) | NA |
| Qwen 3.5 9B (dense) | 4-bit | **12.9 tok/s** | **4.4 tok/s** | 9% (2.0) | NA |
| Qwen 3.5 9B (dense) | 8-bit | **7.8 tok/s** | **4.4 tok/s** | 9% (2.0) | NA |
| Qwen 3.5 2B (dense) | 4-bit | **44.4 tok/s** | **17.4 tok/s** | 3% (2.1) | NA |
| Qwen 3.5 2B (dense) | 8-bit | **29.6 tok/s** | **18.1 tok/s** | 3% (2.1) | NA |


## Documentation

Everything is in the [Wiki](https://github.com/Pummelchen/TinyTitan/wiki).

## Credits

TinyTitan is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), whose
bounded-memory runtime, installer, CLI and local server this project builds on.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).

**Project Maya:** the 6.0 checkpoint, calibration and machine-size cache work was
carried under that name. The live ledger is
[docs/maya-task-status.md](docs/maya-task-status.md); the closeout is
[docs/maya-closeout-2026-10-22.md](docs/maya-closeout-2026-10-22.md).

**Kapitsa:** the cross-session shared system-prefix cache and the M4+ tensor-ops
prefill direction were informed by [nace-ai/kapitsa](https://github.com/nace-ai/kapitsa).
No code is copied from it.

**Related research:** [TinyTitan Datacenter](https://github.com/Pummelchen/TinyTitan_Datacenter)
runs large MoE models across a cluster of Mac minis and Studios, keeping them on
SSD/NVMe for near-linear scale of decode throughput.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright (c) 2026 André Borchert.
