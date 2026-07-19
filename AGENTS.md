# TurboFieldfare

Swift and Metal inference for Gemma 4 26B-A4B on Apple Silicon.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`Sources/TurboFieldfare/` is the runtime; `Sources/TurboFieldfareRepack/`, `Sources/TurboFieldfareCLI/`, and `Sources/TurboFieldfareApp/` contain the installer, CLI, and Mac app.
`Tests/` contains focused public tests; `docs/` contains design, benchmark, and experiment notes.

```bash
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo
swift build -c release
.build/release/TurboFieldfareMac
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The download is about 15 GB.

## Test rules

Before a model run, require macOS 26+, Swift 6.2+, enough disk, acceptable `memory_pressure -Q`, a completed `scratch/gemma4.gturbo`, and no process from `pgrep -fl 'TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests through `Scripts/test.sh`. Run only one app, CLI, or model-using test at a time.

For performance results, build release once and follow the [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable experimental controls or profiling.

Do not download a full checkpoint, duplicate the `.gturbo` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app exposes generation controls plus cache slots, LFU/LRU, prefill, FP16 or K4/V4 KV, RDADVISE, and verification. Interactive generation defaults to 1,024 new tokens, temperature 1.0, Top-K 64, Top-P 0.95, and repetition penalty off at 1.0. Start with production defaults. The HUD reports rate, tokens, and decode-service memory; Last run adds TTFT and I/O. The app must be built with its sibling `TurboFieldfareDecodeService`; it never falls back to loading a second in-process model. See [README](README.md) and [Runtime controls](docs/RUNTIME_CONTROLS.md).
