# TinyTitan 7.1 — the CPU engine reaches the library and the CLI

This release finishes the surface work around the 7.0 engine: the dense CPU
engine is now available to embedding programs and to the CLI, the variant
factory table ships in the library product, and the dense installs that were
found dead during the 7.0 work are healthy again. The shipped engine set is
unchanged by policy: Ornith 1.5 35B-A3B 4-bit remains the one qualified
variant in the registry; the six dense Qwen 3.5 installs run on 7.0 engines at
runtime, and every family without a variant keeps its 6.x path.

### The CPU engine is part of the library and the CLI

`CPUModelBackend` moved from the server into `TinyTitanCPUEngine`, which the
`TinyTitanLib` product now carries beside the facade, the legacy engine and the
variant factories. `Session` runs over any `ServerInferenceBackend` — the
engine owns the backend, the session holds it weakly, so `unload()` still
reports `engineShutDown` — and raw prompts use one path for both engines
(`tokenizeRawPrompt` + `renderedPromptIDs`). The new public `CPUEngine` gives
an embedder the same `ChatMessage`/`GenerationOptions`/events surface as
`Engine`.

The CLI selects it with `--engine cpu|gpu`; no flag keeps the automatic 7.0
GPU routing. `EngineFactories.routingCPUFactory` picks the per-width dense
variant and refuses a GPU-only family by name. The CPU backend reports real
prefill/decode seconds, so the CLI footer is no longer zero.

Verified on the 4B install: `--engine cpu` answered with
`prefill=5tok/0.64s new=4tok decode=0.42s tok/s=9.614` in the verifier run;
`--engine gpu` on the same prompt answered at 25.3 tok/s; greedy CPU and GPU
agree for the first tokens and then diverge, as the two engines' numerics
differ (CPU text is not held to the GPU golden). The library contract suite
passes 5/5, and new session-backend tests pin the raw path and the refusal of
a backend without a tokenizer.

### The 7.0 factory surface ships in the library product

`TinyTitanVariants` is now one of the product's targets, so an embedder and the
CLI construct a variant engine with `EngineFactories.routingFactory` instead of
reaching into the server. Server, CLI and library share one selection table:
Ornith 35B 4-bit on the GPU, dense Qwen 3.5 on `DenseGPU`, and the 6.x factory
for everything else. `tools/release.sh` and `tools/build_library.sh` assert that
the new modules travel with the library archive.

### The dense installs are healthy again

- **2B 4-bit and 8-bit** were installed from a size-correct but fully zeroed
  payload and answered with repeated `!`; the converter now probes content
  before accepting a download (`tools/prepare_qwen35.py`, with its own test
  suite), and both widths were rebuilt from clean conversions. Registry sizes
  follow the rebuilt payloads.
- **4B 8-bit** was 99.98% zero blocks; it was rebuilt, and the 4/4 and 5/5
  parity suites plus the dense goldens pass.
- The dense decode matrix (2B/4B/9B, 4- and 8-bit, GPU and CPU) was sampled on
  2026-11-20; the host was busy (load 5.79), so the absolute figures are
  depressed and only the ordering/ratios are stable. The wiki Benchmarks page
  records it with that caveat.

### Startup aliases and dense routing

`--model <id>@cpu` / `@gpu` now resolves to the base catalog entry at startup;
before the fix, the launcher's engine alias reached the server as a model id
the catalog could not serve. The server routes single-model GPU launches and
catalogue launches through the shipped variants, and the dense CPU path uses
the per-width variant engine.

### A local Apple M3 build (not shipped)

`tools/build_m3.sh` builds every product with `-target-cpu apple-m3` /
`-mcpu=apple-m3` into `.build/m3`, verifies the flags in the compiler log and
every artifact's arch, and refuses a non-M3 host. It is not part of the
release. A 3-pair 4B decode measured 22.44–22.50 tok/s portable against
22.45–22.54 native: no difference outside the noise. The shipped artifacts
remain **generic arm64**: both release builders now refuse a CPU-specific flag
in `Package.swift` or their own build commands.

### The DeepSeek Harness plugin follows 0.2.2-rc.5

`dsh-tinytitan` moved its exact pin — version, peer dependencies, lockfile and
support gate — from 0.2.2-rc.1 to 0.2.2-rc.5. The rc.5 removal of
`parentAgent` is handled through the session's `parentSession`/`delegationDepth`
with the old field as fallback; the shipped standard preset still parses and
builds, and a throwaway rc.5 web profile booted and wrote the TinyTitan route.
The plugin tests (157) and eslint/prettier pass. The plugin refuses a different
harness by design, so the host must be on 0.2.2-rc.5.

### Also in this release

- The gate set now covers the four-target product: `stdout-clean` walks the
  whole `TinyTitanLib` product closure, `library-facade` allowlists
  `CPUEngine`, `verify_cpu_models.sh` exercises `TinyTitanCLI --engine cpu`,
  and `build_library.sh` no longer states a stale module count.
- The server's generated library archive carries `TinyTitanVariants` and
  `TinyTitanCPUEngine` modules.

### Performance

No new performance claim is made for this cut. The 4B dense CPU CLI measured
9.6 tok/s in a short verification run, and the GPU 25.3 tok/s on the same
8-token prompt; both are single-run host observations, not a benchmark table.
The dense 2B/4B/9B matrix and the Ornith post-7.0 investigation are recorded
on the wiki/repo with their busy-host and single-machine caveats. The release
gate re-runs the golden baselines; no benchmark was re-run for this document.

### Checks not run

The installed models on the release machine are `ornith-1.5_35B_A3B_4Bit`,
`qwen3.5_2B_4Bit`, `qwen3.5_2B_8Bit`, `qwen3.5_4B_4Bit`, `qwen3.5_4B_8Bit`,
`qwen3.5_9B_4Bit` and `qwen3.5_9B_8Bit`; their seven golden targets are the
ones the release gate can check. No model may be fetched to fill a row. The
following baselines are **not checked** — no install under `models/`:
`ornith-8`, `qwen36-4`, `qwen36-8`, `qwen38-4`, `qwen38-8`, `k2-4`, `k2-8`,
`agentworld-4`, `agentworld-8`, `katcoder-4`, `katcoder-8`, `ornith122-4` and
`ornith122-8`.

### Checksum

`tinytitan-7.1-macos-arm64.tar.gz` sha256: `SHA256_PENDING`
`tinytitan-7.1-macos-arm64.tar.gz` size: `ARCHIVE_BYTES_PENDING` bytes
`tinytitan-lib-7.1-macos-arm64.tar.gz` sha256: `LIBRARY_SHA256_PENDING`
`tinytitan-lib-7.1-macos-arm64.tar.gz` size: `LIBRARY_BYTES_PENDING` bytes
`tinytitan-7.1-tools.tar.gz` sha256: `TOOLS_SHA256_PENDING`
`tinytitan-7.1-tools.tar.gz` size: `TOOLS_BYTES_PENDING` bytes
