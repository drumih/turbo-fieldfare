# TurboFieldfare

Swift and Metal inference for Gemma 4 26B-A4B on Apple Silicon.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`Sources/TurboFieldfareFormat/` owns the Foundation-only `.gturbo` v1 wire
contract. `Sources/TurboFieldfare/` is the runtime; `Sources/TurboFieldfareRepack/`,
`Sources/TurboFieldfareCLI/`, `Sources/TurboFieldfareServer/`, and
`Sources/TurboFieldfareApp/` contain the installer, CLI, loopback server, and
Mac app.
`Tests/` contains focused public tests; `docs/` contains design, benchmark, and experiment notes.

```bash
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo --resume
swift run -c release TurboFieldfareRepack --model-family kimi-k3 --output scratch/kimi-k3.gturbo
swift build -c release
.build/release/TurboFieldfareMac
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The download is about 15 GB. Cancellation preserves verified completed ranges; continue them with `--resume` or remove them with `--discard-partial --output scratch/gemma4.gturbo`.

## Kimi K3 (`.gturbo` v2)

This fork also runs Kimi K3 (2.78T MoE: 93 layers = 69 KDA + 24 NoPE-MLA,
896 experts top-16 in a 3584-wide LatentMoE, AttnRes residual blocks) on a
128 GB Apple Silicon Mac by streaming the MXFP4 experts from SSD; design
basis and feasibility math live in
[docs/KIMI_K3_EVALUATION.md](docs/KIMI_K3_EVALUATION.md) and the exact engine
dataflow contract in [docs/K3_DATAFLOW.md](docs/K3_DATAFLOW.md). The v2 wire
format is pinned by golden fixtures in
`Tests/TurboFieldfareFormatCompatibility/Fixtures/v2/`; v1 behavior and
fixtures must keep passing unchanged.

```bash
# Install (streams the pinned moonshotai/Kimi-K3 revision; ~1.5 TB out, resumable)
swift run -c release TurboFieldfareRepack \
  --model-family kimi-k3 --output scratch/kimi-k3.gturbo
swift run -c release TurboFieldfareRepack \
  --model-family kimi-k3 --output scratch/kimi-k3.gturbo --resume
# `--trunk-quant int8` trades ~28 GB more residency for trunk quality headroom.

# Run (CLI auto-detects v1 vs v2 bundles)
swift run -c release TurboFieldfareCLI \
  --model scratch/kimi-k3.gturbo \
  --prompt "The capital of France is" --max-new 64
# Chat: --messages-file messages.json --reasoning-effort low|high|max
# Prefill: --prefill chunked --prefill-chunk 32 (K3 default; NAX on M5)
# I/O: whole-expert reads, auto workers 1/2/4, F_NOCACHE for production K3
# Overrides: --expert-io-splits 1|2|4|8 --expert-io-workers auto|1...32
#            --expert-io-cache auto|buffered|uncached
# Fast startup after a completed verified install: --model-verification trusted-install

# Serve
swift run -c release TurboFieldfareServer --model scratch/kimi-k3.gturbo
```

Reality check: decode reads ~25.8 GB of experts per token from SSD, so expect
roughly 0.3–1 tok/s — this is a batch/research runtime, not an interactive
chat engine. Resident budget is ~38 GB wired at the default int4 trunk
profile; the OS page cache absorbs expert reuse. The server retains one exact
prompt prefix by snapshotting KDA/conv, active MLA rows, last logits, and
routing predictions; hits report `cached_tokens`, and any token mismatch
falls back to full prefill. The `stop` request field is rejected for K3
(token-level stop only). All K3 code lives in K3-namespaced files
(`Runtime/K3/`, `Metal/K3/`, `Tokenization/K3*`, repack `Core/K3/`); tests are
`Tests/**/K3*/` and run with the same `Scripts/test.sh` rules below — K3 tests
use synthetic tiny bundles, never the real checkpoint.

## Local server

Follow the [server guide](docs/OPENAI_SERVER.md) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
Apply the model-process checks below first; never start a second model process
or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do
not proxy, tunnel, or expose it. A tool call from the local model never bypasses
the client's normal permission policy. Keep the execution session alive while
the server is needed, and stop only a server you launched.

## Test rules

Before a model run, require macOS 26+, Swift 6.2+, enough disk, acceptable `memory_pressure -Q`, a completed `scratch/gemma4.gturbo`, and no process from `pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests through `Scripts/test.sh`. Run only one app, CLI, or model-using test at a time.

For performance results, build release once and follow the [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable experimental controls or profiling.

Do not download a full checkpoint, duplicate the `.gturbo` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app sends prompts through the pinned Gemma 4 IT chat format. It
exposes context length, temperature, Top-K, Top-P, expert-cache slots, prefill,
and RDADVISE. The defaults are temperature `0.2`, Top-K `64`, and Top-P `0.95`.
Responses can use the context space left after formatting the prompt, and FP16
is the runtime KV format. The HUD shows generation rate, token count, and
decode-service memory; Last run also shows time to first token and I/O. Build
the app with its sibling `TurboFieldfareDecodeService`; it never loads a second
in-process model. See [README](README.md) and [Runtime controls](docs/RUNTIME_CONTROLS.md).
