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
in release and the test conversions are DONE: the full serial suite passes
(2,371 tests across all eight targets; one transient TinyTitanServerTests
runner crash re-ran green). The conversions touched ~29 sites across five
test files (`ANEPrefillAttentionTests`, `RoleUniformityTests`,
`ReasoningControlTests`, `LagunaSurfaceTests`, `LagunaToySynthetic` — see
`git show HEAD` and `swift build --build-tests`) pass `family:` as a
`ModelFamily` member where `ArchConfig.family` is now a String, or compare
`config.family == .member`.

**VERIFIED 2026-11-15: full serial suite green.**
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

## Scope decision (owner, 2026-11-15): 7.0 ships ONE variant

**Ornith 1.5 35B-A3B 4-bit** (`ornith-1.5-35b-a3b-4bit`, the qwen36-family
MoE on the GPU engine with SSD expert streaming). The dense 4B/9B variant
ports stay in the tree as code for future releases; the Qwen3.8 8/8/16
conversion was stopped mid-run (staging discarded, the converter resumes
from finished output shards if the 8-bit install is wanted later). The
generic Core foundation (manifest layer, MetalContext, geometry configs)
serves every future variant.

## Remaining work after the WIP lands green

1. **Ornith 1.5 35B 4-bit variant** — the install is verified on disk
   (`models/ornith-1.5_35B_A3B_4Bit`, 47 files, receipt written 2026-11-15;
   it required fixing a repacker bug — see below). Port the qwen36-family
   GPU engine into an `Ornith35B4Bit` variant: the manifest layer,
   `MetalContext` and the geometry configs are already in Core; the
   remaining lift is `Model`, the kernel objects, expert streaming and the
   runner — a major refactor, not a copy. Parity gate: golden baseline
   `ornith-4` exists in `benchmark/golden/`.
2. Qualification record for the variant (golden + benchmark), registry
   `shipped` flag, `catalog --check` green.
3. Version bump to 7.0, release notes, tag, dry run, publish per
   `docs/release-process.md`.

## Repacker bug fixed here (17529070)

A conversion-site regex stripped `.rawValue` from
`archDict["family"]` in `SSDAIJSON.swift`, putting the `RepackModelFamily`
enum into the JSON dict — NSJSONSerialization crashed
("Invalid type in JSON write (__SwiftValue)") at the finalize stage of every
local import. Fixed; the Ornith install completes and verifies.

## Machines and state

- This Mac (24 GB M3) is shared with the FXAI project — check the top CPU
  users before calling a number a measurement.
- The 8-bit conversion was started with the owner's approval; no other model
  may be downloaded without it.
- Lint invocations need `PATH=".build/shims:$PATH"` (pinned ruff; recreated
  after any `swift package clean`).

## 7.0 release verification (2026-11-15)

Verified at commit `c3e65db9` on `main`, Apple M3 MacBook Pro 24 GB,
macOS 27.2.1, Swift 6.4.2 (Xcode 27 toolchain). No model was run in this
session (no benchmark, golden or server); the ruff shim was recreated
(`.build/shims/ruff` → pipx ruff 0.16.7) before lint.

**Result: NOT all green.** Release build and `catalog --check` pass; the
serial suite has one deterministic failure and lint fails 2 of 20 checks.
All three defects pre-exist on `main` and sit in files this verification
session does not own (tests/registry/sources) — they are recorded here for
the owner to fix before the release can go out.

### Serial suite — `swift test --no-parallel` (exit 1)

2,372 tests across nine targets; 2,371 passed, 1 failed:

| Target | Tests | Suites | Result |
| --- | --- | --- | --- |
| TinyTitanTests | 1307 | 271 | **FAILED (1 issue)** |
| TinyTitanServerTests | 434 | 49 | passed |
| TinyTitanRepackTests | 196 | 31 | passed |
| TinyTitanMemoryTests | 189 | 23 | passed |
| TinyTitanLibTests | 129 | 17 | passed |
| TinyTitanCoreTests | 15 | 2 | passed |
| Qwen35Dense2B4BitTests | 3 | 1 | passed |
| Ornith35B4BitTests | 1 | 1 | passed |
| ContinuityCoreTests | 98 | 10 | passed |

The failure is **not** a transient runner crash:
`RegistryInventoryTests.registryParsesAndCoversTheShippedPairs`
(`tests/TinyTitan/Registry/RegistryInventoryTests.swift:52`) asserts
`variant["shipped"] as? Bool == false` for **every** registry variant.
Commit `87d1dad5` (Ornith qualification) set `ornith-1.5-35b-4bit` to
`shipped: true` without updating this invariant; the test last changed in
`63297e4e`/`1944cacd`, before qualification. Re-ran
`swift test --no-parallel --filter RegistryInventoryTests`: same failure,
exit 1 — deterministic. Fix belongs to the test/registry owner: the
assertion must accept the shipped Ornith pair (e.g. pin the shipped set
explicitly) rather than demanding `false` everywhere.

### Lint — `PATH=".build/shims:$PATH" tools/lint.sh` (exit 1, 18/20 ok)

ok: force-cast, unbounded-read, file-length, converter, arch-path,
stdout-clean, silent-test-skip, test-hollow, library-facade, docs,
shell-portability, shellcheck (0.11.0), swiftlint (0.65.1 --strict),
swift-format, javascript (eslint 10.11.0 / prettier 3.9.9), python
(ruff 0.16.7), core-purity, variant-isolation.

FAIL:
- **func-length** — 4 new functions over 120 lines, all in the Ornith
  variant: `sources/Engines/Ornith35B4Bit/Kernels/Attention/Attention.swift:encodeSplit`,
  `Runtime/Generation/RawCompletionHelpers.swift:runStreamingMTPCompletion`,
  `Runtime/Inference/Model+SchemaValidation.swift:validateLayerTensors`,
  `Runtime/Inference/RealForwardRunner+PrefillProjection.swift:encodeAffineProjection`.
  Shorten/split them or add audited rows to `tools/func-length-baseline.txt`.
- **unchecked-sendable** — `sources/TinyTitanCore/Format/TensorView.swift:TensorView`
  has an `@unchecked Sendable` without an `unchecked-invariant:` note.

### Build and catalog

- `swift build -c release` — exit 0, Build complete! (344.97 sec).
- `swift run -c release TinyTitanCLI catalog --check` — exit 0,
  "catalog: ok (registry ↔ target ↔ qualification consistent)".

### Ornith 1.5 35B 4-bit qualification (registry, read-only)

`ornith-1.5-35b-4bit`: `shipped: true`, target `Ornith35B4Bit`, gpu engine;
qualification dated 2026-11-15 on Apple M3 MacBook Pro 24 GB: golden
SHA256 `363ba710eef09deaeb3d2fe6c060b88b658bb392d767da0b4021ed6fc01500a7`,
decode **11.55 tok/s**. Golden `ornith-4` passed at qualification time
(commit `87d1dad5`); it was **not re-run** in this session (no model runs).

### Publish status (updated 2026-11-15, commit b3a16747)

The three failures above are **fixed** on `main` (`b3a16747`): the registry
test now asserts a shipped variant carries a target + qualification; the four
copied >120-line engine functions are baselined as verbatim AUD-024 copies;
`TensorView` in Core carries its `unchecked-invariant` note; CONTRIBUTING's
baseline count is corrected to 17. Re-verified: **full serial suite green
(2,372 tests, 9 targets)** and **all 20 lint gates green**; `catalog --check`
exit 0. The annotated tag **v7.0** is created and pushed on `b3a16747`.
Publish remains **pending CI green** on the tag commit (`tools/ci-green.sh`),
then `tools/release.sh v7.0` (dry run) and `--publish`.

### Final 7.0 audit — independent code review (2026-11-15)

A read-only reviewer audited the 7.0 changes against the spec gates. Verdicts:
**variant-isolation PASS** (all 368 files; Ornith35B4Bit imports only
Core/Format/KernelsC/Tokenizers + system frameworks; the four func-length
baseline copies are byte-identical to the runtime originals), **library-facade
PASS** (169 `public`, all on the baseline). Three concerns, handled as follows:

- **FIXED — `Qualification` nullability.** `TinyTitanCore/Variant/Qualification.swift`
  required non-optional `prefillTokS`/`ttftS`/`memoryCeilingBytes`, but the
  shipped registry record stores them `null` (unmeasured fields stay null by
  design). Any `Codable` decode of the record would throw. Made the three
  fields optional (`Double?`/`Int?`, init defaults `nil`) and added
  `qualificationDecodesRegistryRecordWithNullUnmeasuredFields` to
  `VariantContractsTests`, decoding the registry shape and asserting the nulls.
  Verified: `VariantContractsTests` 14/14 green (the new decode test included).
  Sibling audit (read-only): no sibling defect — `Qualification` is the only
  `Codable` type in `Core/Variant/` (the rest are plain `Sendable` structs,
  never decoded); `CatalogCommand` parses the registry with raw
  `JSONSerialization` and never decodes `Qualification`, so the bug was latent
  for a future Codable consumer; no production decode site exists and the
  install manifest carries no nulls.
- **FOLLOW-UP — core-purity gate scope.** The gate scans `*.swift` identifiers
  only; model names survive in Core *comments* (`Format/ArchConfig.swift`,
  `Format/ModelTypes+Configs.swift`) and in `.metal` sources, and the regex
  misses dotted/hyphenated spellings (`Qwen3.6`, `K2-Horizon`). Gate is green as
  written; strengthening its scope is future work, not a 7.0 blocker.
- **FOLLOW-UP — variant serial-use invariant.** `Ornith35B4Bit` is documented
  serial-use; the `Bridge` semaphore ordering is correct, but concurrent
  `generate`+`decode` would race and `runSync` blocks on a semaphore. This is
  the inherited 6.x engine design (parity-verified, driven serially by the
  session); actor isolation is future work. Also noted: `CatalogCommand`
  returns `[]` on a `swift package describe` failure (fail-open findings) and
  its orphan check is CWD-relative; `decode` is one-ahead vs `generate`. All
  inherited semantics, parity-verified — recorded for the phase-4 cleanup.
