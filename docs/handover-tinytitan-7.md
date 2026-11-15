# Handover: the TinyTitan 7.0 conversion (2026-11-15)

Read `AGENTS.md` first, then this file, then the plan
(`docs/plan-tinytitan-7-engine.md`, tracking the wiki Project-Tracker rows
TT-072…TT-087 and issue #32, closed with the work tracked in the tracker).
The task list lives in the tracker and the git log; this file carries the
state a fresh session cannot discover from those quickly.

## What is done and verified (all pushed to origin/main)

- **Generic 7.0 foundation in `TinyTitanCore`** — registry
  (`registry/registry.json`, 9 models / 19 pairs), `core-purity` and
  `variant-isolation` lint gates (tools/lint.sh is now 20 checks),
  `EngineVariant`/`QuantSpec`/`QuantRole`/`GeometryContract`/profiles
  contracts, `tinytitan catalog --check`, and the **complete generic
  install-reading stack**: `ResidentIndex`, `ModelError`,
  `SSDAIModelDirectory`, `BoundedMetadataRead`, manifest types + wire
  conversion, and `SSDAIManifestReader` (the runtime's `ManifestReader.read`
  delegates to it). `MetalContext` + the shader bundle also moved to Core with
  an injectable shader-module list (Core default names no model; the runtime
  adds `nemotron` via `MetalContext.runtimeDefault`).
- **The dense family (4B/9B × 4/8-bit) is ported to 7.0 and parity-verified**:
  four variant targets (`sources/Engines/Qwen35Dense*`) run the config-driven
  dense CPU engine and produce greedy output **identical to the 6.x runtime
  engine** on their own installs (DenseParityTests, all four passed). The
  4B-4bit variant was the template; the other three replicate it with their
  own geometry/quant statics. `VariantError.notPorted` is what an unwired
  variant throws.
- The matrix test (role resolution + the 8-bit-as-nibbles trap) is in
  `tests/Engines/Qwen35Dense4B4Bit/MatrixTests.swift`.
- All four dense installs verified on disk under `models/`.

## What is in progress — the `ArchConfig.family` decoupling (WIP)

The GPU runtime lift into Core requires `ArchConfig` to be model-free, so
`ArchConfig.family` was changed from `ModelFamily` to a raw `String` (the
runtime resolves it via a `modelFamily` computed property;
`ModelFamily`/the presets stay in the runtime). The source tree builds green
in release. **The test target does not compile yet**: ~29 sites across five
test files (`ANEPrefillAttentionTests`, `RoleUniformityTests`,
`ReasoningControlTests`, `LagunaSurfaceTests`, `LagunaToySynthetic` — see
`git show HEAD` and `swift build --build-tests`) pass `family:` as a
`ModelFamily` member where `ArchConfig.family` is now a String, or compare
`config.family == .member`.

**Trap learned the hard way: a scripted/regex fix does NOT converge.** Each
conversion shifts type inference and creates new sites (error counts
oscillated 23 → 83 → 10 → 29 across attempts), and `.family` exists on
several types (`ManifestIdentity`, `ServerReasoningProfile`, `ArchInfo` with
its own `RepackModelFamily`) that a regex wrongly renames. Fix each site
manually with full context: sites that feed `ModelProfile.resolve`,
`apiModelID`, `TensorSchema.schema(for:)`, or `ServerReasoningProfile`
expect `ModelFamily` — use `X.modelFamily` or `.member`; sites that build an
`ArchConfig` keep the String. Then run the full serial suite
(`swift test --no-parallel`).

## Running processes (detached from any editor session)

The Qwen3.8-Flash-Next **8/8/16 8-bit snapshot conversion** is running
detached (PPID 1, survives editor closure):
`python3 tools/prepare_qwen38.py --bits 8 --rest-8bit --reuse-ngram-table
models/qwen3.8-flash-next_125B_A6B_4Bit --output .build/qwen38-8bit-snapshot
--work .build/qwen38-8bit-shards` (log `/tmp/qwen38-8bit-convert2.log`, PID
8118 at handover time). When it completes: repack with
`TinyTitanRepack --input-snapshot .build/qwen38-8bit-snapshot --output
models/qwen3.8-flash-next_125B_A6B_8Bit --consume-snapshot
--share-ngram-table --model-id qwen3.8-flash-next-8bit`, then
`--verify-install` and a smoke generation. Disk-tight: ~11 GB margin, so do
not raise `--fetch-workers` above 1 while the install is being written.

## Remaining work after the WIP lands green

1. Dense **GPU** path for the variants (they serve CPU only today).
2. **Qwen3.8 GPU port** — `RealForwardRunner` is ~11K lines over 45 files,
   built on `Model`, `MetalContext`, the kernel objects and expert streaming.
   With `MetalContext`, the geometry configs and the manifest layer already
   in Core, the remaining lift is `Model`, the kernel objects, and the
   runner itself — a major refactor, not a copy.
3. Qualification records per variant (golden + benchmark), registry
   `shipped` flags, `catalog --check` green.
4. Version bump to 7.0, release notes, tag, dry run, publish per
   `docs/release-process.md`.

## Machines and state

- This Mac (24 GB M3) is shared with the FXAI project — check the top CPU
  users before calling a number a measurement.
- The 8-bit conversion was started with the owner's approval; no other model
  may be downloaded without it.
- Lint invocations need `PATH=".build/shims:$PATH"` (pinned ruff; recreated
  after any `swift package clean`).
