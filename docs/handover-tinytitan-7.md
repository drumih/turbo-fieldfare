# Handover: the TinyTitan 7.0 conversion (2026-11-15)

Read `AGENTS.md` first, then this file, then the plan
(`docs/plan-tinytitan-7-engine.md`, tracking the wiki Project-Tracker rows
TT-072…TT-087 and issue #32, closed with the work tracked in the tracker).
The task list lives in the tracker and the git log; this file carries the
state a fresh session cannot discover from those quickly.

## TT-097 COMPLETE — the server runs the shipped 7.0 variant engines

Completed 2026-11-18. Ornith 1.5 35B-A3B 4-bit and all six dense Qwen 3.5
installs route to their shipped variant engines on CPU and GPU through eager,
managed and catalogue launches. The engine contracts and factory boundary live
in Core; `TinyTitanLib` no longer owns concrete engine construction. Other
supported models retain their qualified 6.x path, so **6.x is not deleted**:
TT-087 remains Won't Do, with its prerequisite now reachable rather than done.

DenseGPU is token-identical to the 6.x GPU oracle on all six installs, and the
weather-tool prompt regression probe is also byte-identical on 2B 4/8-bit and
4B 8-bit. The feature matrix passed streaming, explicit sampling overrides,
reasoning separation and prompt-cache reuse on every installed identity. Ornith,
4B 4-bit and both 9B widths also passed the tool-call probe; the three identities
above inherit the reference model's repeated-`!` response and are not migration
regressions. The 2B models remain unsuitable for instruction/tool validation.
No benchmark was run on the busy host; the paired variant-vs-6.x performance
comparison remains deferred to a quiet host, so the specification's performance
acceptance criterion is not closed.

Final golden checks passed for Ornith server agreement and dense 4B 4-bit plus
both 9B widths. The 2B widths and 4B 8-bit mismatch their older fluent raw-
completion baselines by producing repeated `!`. This is not a variant routing
regression: DenseGPU and 6.x GPU are token-identical, the 2B 4-bit CPU variant
matches 6.x CPU, and the independent `TinyTitanBench cpu35gen` 6.x CPU path also
produces 96 `!` tokens on the current 2B 4-bit install. Forcing the real-logits
head did not change the result. The stored baselines are unchanged; recapture is
not justified by this evidence.

**The four authoritative artifacts (read in this order):**
1. `AGENTS.md` — the working rules (toolchain, gates, model-run discipline, rule 2
   "one generation path").
2. **Spec** `docs/superpowers/specs/2026-11-16-tt097-engine-agnostic-session-design.md`
   (approved 2026-11-16).
3. **Plan** `docs/superpowers/plans/2026-11-16-tt097-engine-agnostic-session.md`
   (16 tasks, 4 phases, TDD-shaped; each task is an independently-green commit).
4. Task table rows **#69–#72** (Phase 1–4); this handover for state not in git.

**The design in one paragraph:** the server already dispatches to
`ServerInferenceBackend` with two impls (`ServerModelSession` GPU in `TinyTitanLib`,
`CPUModelBackend` dense-CPU in `TinyTitanServerCore`), both `import TinyTitan` (6.x).
The forward path already runs through the existential `any LogitProducer`, but each
module defines its **own** copy of `LogitProducer`/`ChunkedPrefillRunner`/etc. The
fix: promote that engine-protocol family into `TinyTitanCore` as **one** shared,
model-free definition that 6.x and every variant conform to, type the two sessions
against a Core `EngineRunner`/`EngineModel` protocol, move concrete engine
construction behind an injected **`EngineFactory`** so `TinyTitanLib` imports no
engine module for inference, and have `TinyTitanServerCore` (which may import the
variants) select the 6.x or variant factory per catalog identity. Dense GPU needs a
new **shared config-driven** variant engine (`sources/Engines/DenseGPU/`, the
`numExperts==0` runner from `plan-dense-gpu-engine.md`). 6.x is **never deleted**
(it is the regression engine all the way; retired only by TT-087 after Phase 4).

**Landed so far (all pushed to `main`):**
- Spec `f4be7a78`; plan `01c1df59`; plan refinement `248d6efc`.
- **Phase 1 Tasks 1–4** `71ce076b`, `7b8f6a29`, `effe7e07`: diagnostic counter
  pins, the shared Core engine-protocol family + `EngineStatistics`, and 6.x
  `Model`/`RealForwardRunner` conformance.
- **Phase 1 Task 5** `267b8829`: `CPUEngineRunner` +
  `CPUEngineConfiguration`, Ornith variant GPU conformance, and 6.x + all six
  dense-variant CPU conformances. The synthetic CPU contract test drives
  `reset`/`prefill`/`step` for real. Verified with a release build, 2,389
  model-free tests across nine targets, all required lint gates, and golden
  `ornith-4` byte-identical. Golden `ornith-8` was **not checked** (no install;
  none fetched).
- **Phase 2 Task 6** `57b4fed2`: Core `EngineFactory` and `SessionBuildOptions`
  now define the injection boundary; KV-cache precision and RoPE scaling types
  are shared through Core aliases in both runtime copies. `promptCacheMode`
  deliberately remains session policy. Verified with a release build, 2,390
  model-free tests across nine targets, all required lint gates, and golden
  `ornith-4` byte-identical. Golden `ornith-8` was **not checked** (no install;
  none fetched).
- **Phase 2 Task 7** `c5bdf926`: `ServerModelSession` now holds Core
  `EngineModel`/`EngineRunner` contracts and loads through an injected factory;
  the package-private default still constructs 6.x, including MTP. Multi-slot
  batching and diagnostics use the Core contracts. Benchmark probes and expert
  warm-up remain explicitly 6.x-only capabilities. Verified with a release
  build, 2,391 model-free tests, all required lint gates, golden `ornith-4`
  byte-identical, and Ornith greedy parity against 6.x (37.83 s). Golden
  `ornith-8` was **not checked** (no install; none fetched).

- **Phase 2 Task 8** `a6731816`: `CPUModelBackend` now stores any Core
  `CPUEngineRunner` and builds through `CPUEngineFactory`; the zero-argument
  factory lets each variant capture its own snapshot representation, while the
  server-side default still constructs 6.x `CPUQwen35`. Verified with a release
  build, 2,392 model-free tests, all required lint gates, and golden `ornith-4`
  byte-identical. Golden `ornith-8` was **not checked** (no install; none
  fetched).

**Remaining acceptance work:**
- Run the paired variant-vs-6.x performance comparison on a quiet host; the busy-
  host deferral is explicit and is not evidence of no regression.
- Preserve the three dense raw-golden mismatches above as open baseline findings,
  not as failures of the migrated engines. Do not recapture them without evidence
  of a deliberate numerics or checkpoint change.
- Optional follow-up: compose unlisted `@cpu`/`@gpu` aliases with the `-fast`
  suffix in HTTP validation; direct engine aliases and listed `-fast` IDs work.

**Invariants this work must never break:** zero numerics change until a model is
routed to a variant (then golden/parity byte-identical or it is a finding, not a
re-baseline); one generation path (reuse the session, do not fork it); every commit
green (never leave `main` mid-migration); no model fetched to satisfy a gate;
Apple Silicon only; 6.x preserved (on `main` live + `archive/6.x-final`).
Model runs are gated: require macOS 26+, Swift 6.4, `memory_pressure -Q` ok, no
competing model process, valid install receipt; run tests serially, one at a time.

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

## Measured integration gap (2026-11-16) — what actually blocks "main = 7.0 only"

Re-derived from the tree, not assumed:

- **The production server/library still run entirely on the 6.x `TinyTitan`
  runtime.** `Package.swift`: `TinyTitanServer → TinyTitanServerCore →
  { TinyTitanLib, TinyTitan }` and `TinyTitanLib → TinyTitan`. The GPU server
  session and the dense GPU path both call 6.x `Model.load`/`RealForwardRunner`
  (e.g. `K2ProbeBackend.swift:84`). **No product target imports any variant.**
- **`EngineVariant` exists and is clean** (`TinyTitanCore/Variant/EngineContext.swift`:
  `prefill`/`decode`/`qualify`, `EngineContext`/`LoadedModelHandle`/`SessionState`),
  and every variant implements it — but **there is no `VariantRegistry` and no
  consumer**: the spec's "registry resolves a variant id to a concrete factory"
  is not written yet. `grep EngineVariant sources/` hits only Core + the variants.
- **Phase (d) "dense GPUEngine port" is NOT a feature port.**
  `docs/plan-dense-gpu-engine.md` is marked implemented+verified: dense 2B/4B/9B
  (×4/8) already run on the **GPU** — via the config-driven 6.x `RealForwardRunner`
  (`numExperts == 0`), gated by `TINYTITAN_DENSE_GPU_EQUIV=1` + both golden
  baselines. So the remaining work is *retargeting* the server onto variant
  engines, not building new inference.
- **Type-conflict note for any future "one variant dir per model with both CPU+GPU
  in one target" idea:** the dense `CPUEngine` and the GPU engine share 9 top-level
  type names (`GenerationDefaults`, `Int8AffineGEMV`, `ModelFamily`,
  `ModelReasoningControl(+Error)`, `Quantization`, `ReasoningLevel(+Error)`,
  `RequestReasoning`), so strict separation means **separate variant targets**, not
  co-located directories.

The minimal sequence to reach "server runs 7.0, then retire 6.x" — all serial, all
model-gated (a real install + `tools/golden-baseline.sh` per step), and none of it
parallelizable to agents:

1. `VariantRegistry` (id → factory) + `EngineVariant` qualification hook, in Core.
2. A server-side consumer that drives a variant's `prefill`/`decode` through the
   streaming session (rule 2: ONE generation path — move the session onto it, do
   not fork it), selectable per model; keep 6.x for not-yet-ported models.
3. Wire Ornith first (TT-082): parity + golden on the variant engine (TT-083).
4. Wire the dense trio (CPU + GPU) the same way.
5. Only once every served model runs on a variant: drop `TinyTitan` from
   `TinyTitanLib`/`TinyTitanServerCore`/CLI, retire 6.x off `main` (it stays on
   `archive/6.x-final`), sync registry/scripts/wiki.

Step 2 is the load-bearing one and the risk: it rewrites the generation path the
whole product sits on, so it must land behind a real model-run gate or not at all.

**Measured cost of step 2 (2026-11-16, after the user chose "wire Ornith now,
verified").** It is not a session — it is a multi-step migration, because the
server inference contract a variant would implement is entangled with 6.x:
- `ServerInferenceBackend` + `ServerInferenceEvent` + `ServerCompletion` live in
  **`TinyTitanLib`**, and a variant cannot import `TinyTitanLib` (variant-isolation).
- The protocol's `samplingDefaults` → `GenerationDefaults.Sampling` (runtime/6.x),
  and `generate(_ request: ValidatedChatRequest)` → `ValidatedChatRequest`
  (`OpenAIRequestValidation.swift` imports `TinyTitan`).
- `ServerModelSession` is **2,236 lines across 8 files** and drives concrete 6.x
  engine members (`runner.totalPrefetchIssued`, `model.routedExpertWeightBits`,
  `runner.expertStreamingStatistics`, prompt-cache, checkpoints). It is the
  generation path itself, not an adapter over `prefill`/`decode`.

So a variant can only serve the server after: (i) the server-inference contract
(and the sampling type) is decoupled from 6.x into a variant-importable module,
(ii) the 2,236-line session is re-hosted onto the variant engine, (iii) Ornith is
routed to it, (iv) model-gated. (i) is a bounded phase-a/b/c-style move; (ii) is
the large one. **Do (i) incrementally and keep main green at each commit; do not
start (ii) unless it can be finished + model-verified in the same effort** — a
half-migrated session leaves `main` untested, which the rules forbid. The bare
`GenerationDefaults.Sampling` struct (plain Float/Int fields) is the first piece
of (i): put `GenerationDefaults` (bare constants + `Sampling` + `house`) in Core
and leave the family rows + `forFamily(_:)` in the runtime as a same-named
extension (Swift allows cross-module extensions to add static methods), so
`GenerationDefaults.Sampling` stays callable by name everywhere and Core carries
no model knowledge.

6.x stays the server's engine until this chain is complete; `archive/6.x-final`
remains the fallback. The user's "one LLM on 7.0 = project complete" goal is met
by the 7.0 **variant** passing its own golden + parity (already true), NOT by the
server process importing it — those are separate claims and the tracker rows
TT-082/083/087 should say so until (ii)+(iii) actually land.

## CLOSURE — decided 2026-11-16 (owner): 7.0 closed at the variant bar

The owner chose to **close 7.0 on the variant-qualification bar**, not to attempt
the server re-host in this effort. Outcome, all recorded + published:

- **Re-verified, not cited stale:** preconditions clean (Swift 6.4.2 / macOS
  27.2.1 / 204 GB free / 51% mem / no competing model process), and
  `OrnithParityTests.greedyParityAgainst6x` was **re-run against the real
  `models/ornith-1.5_35B_A3B_4Bit` install and passed in 82.1 s** — the 7.0
  variant engine is greedy-byte-identical to 6.x on the real model. (Run it with
  `TINYTITAN_ORNITH_PARITY_MODEL=models/ornith-1.5_35B_A3B_4Bit swift test
  --no-parallel --filter Ornith35B4BitTests`; without the env var it self-skips.)
- **Tracker (wiki, pushed `a8812e6`):** TT-082 → **Done** at the variant bar;
  TT-087 → **Won't Do** (6.x NOT deleted from main — it still serves every model;
  preserved on `archive/6.x-final`); **new TT-097** (Open, deferred) carries the
  server-serves-variant migration + its measured blocker + staged plan. Footer
  count corrected to 39 rows; the two claims (variant-qualified vs
  server-serves-variant) are stated apart everywhere so neither is overclaimed.
- **What a future session owes if it wants the server on 7.0:** TT-097 only.
  First bounded step is moving the model-free `GenerationDefaults.Sampling`
  struct into Core (Core = bare constants + `Sampling` + `house`; family rows +
  `forFamily` stay runtime as a same-named extension of computed `static var`s),
  then untangle `ValidatedChatRequest`, then re-host the 2,236-line session, then
  route + model-gate. Do NOT start the re-host unless it can be finished +
  model-verified; a half-migrated session leaves `main` untested, which the rules
  forbid. The Core unification phases (a/b/c) + the parallelFor deadlock fix +
  the dense-trio variants are all landed regardless and stand on their own.

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

## 2026-11-16 — one-model scope reset; prefill deadlock fixed; dense 2B/4B added

Owner reset the 7.0 completion scope: **7.0 ships Ornith 1.5 35B-A3B 4-bit (GPU)
plus dense Qwen 3.5 2B/4B 4-bit (CPU); all other models are deferred to user
request; once these work perfectly on the variant engine, retire 6.x and 7.0 is
complete.** Tracker: TT-082 (wire variant as server engine), TT-087 (retire 6.x),
TT-095/TT-096 (this session's fixes).

Done and pushed this session:
- **TT-095 — full-chunk prefill deadlock** (`61f92ba1`). A ≥4096-token MoE
  prefill chunk hung at 0% CPU mid-layer (OpenCode agent turn). Root cause
  (isolated by forcing serial I/O, which completed): `executeCachedPreads` ran
  `DispatchQueue.concurrentPerform` *on the expert-I/O scheduler's 4 fixed worker
  threads*; concurrentPerform counts the caller as a worker and blocks it,
  starving the GCD pool under concurrent fetches. Fix: generic dispatch-group
  `parallelFor` in `TinyTitanCore/Concurrency/ParallelFor.swift`, used at the 6
  scheduler-context sites (executeCachedPreads/decodeBlocks/readRanges) in the
  runtime AND the Ornith variant. RANSDecode (layer-open) / NgramTableReader
  (main path) are not on the worker pool — left. Verified: ParallelForTests 3/3,
  5,818-token prompt completes, golden ornith-4 byte-identical, 20 gates green.
- **TT-096 — dense 2B/4B 4-bit** (`77638454`). Ported Qwen35Dense2B4Bit CPU
  forward (copied the config-driven CPUEngine from 4B; 2B statics). Both installed
  via `tools/install_models.sh qwen35-2b`/`qwen35-4b` (installer proven: streams,
  converts, hashes, writes receipt, refuses snapshot mixing). dense2B4BitParity
  and dense4B4BitParity passed vs the 6.x CPU engine.

### The remaining migration (TT-082 → TT-087), scoped

The variants own the full engine (runRawCompletion/tokenizer/sampler/decoder are
byte-identical copies of 6.x) and are parity-verified, but the SERVER still
generates through 6.x via TinyTitanLib `ServerModelSession`. **Key finding: the
server-adapter layer is ~10,377 lines of TinyTitanLib, nearly all `import
TinyTitan`; `ServerCompletion` cascades into OpenAIUsage / ParsedToolCall /
StopReason / WatchdogSet, so the backend contract is deeply 6.x-coupled.** The
variant duplicates every engine type nominally, so the adapter cannot drive the
variant's `runRawCompletion` without being re-typed.

Staged plan (each gated on the previous; NO destructive deletion before stage 3):
1. Move the generic backend contract (ServerInferenceBackend / ServerInferenceEvent
   / ServerCompletion + AssistantOutput / watchdogs / prompt-cache, and the cascade
   data types) into shared TinyTitanCore, decoupled from 6.x. `defaultPrefillChunkTokens`
   stays in Lib (it switches on ModelFamily).
2. Build a variant-typed server session over the variant engine (a spec-sanctioned
   copy of ServerModelSession re-typed for the variant; ModelProfile.resolve →
   variant statics, ModelFamily → variant family).
3. Server selects the variant for Ornith + dense; prove golden byte-parity +
   streaming/tools/sampling/prompt-cache/reasoning/stop-sequences/cancellation/
   262144 context + OpenCode end-to-end agent turn + decode benchmark.
4. Retire 6.x (TT-087): ModelFamily (62 files), ModelProfile (19), the 8
   tools/prepare_*.py converters, and the 6.x paths in Lib/CLI/Bench/Server/
   Validation; adjust installer/launcher/catalog/golden/release scripts to the
   shipped set. Keep arm64-only/release discipline; catalog --check is the model list.

This is a multi-thousand-line refactor — execute as a focused phase, verifying the
build + gates at every stage and keeping the tree green between stages.

### Progress 2026-11-16 (data-vocabulary stage done)

- **Done:** the shared backend data vocabulary now lives in `TinyTitanCore` and is
  used by both 6.x and the Ornith variant — `StopReason` (`Generation/StopReason.swift`),
  `JSONValue` + `ParsedToolCall` + `ToolCallParserError` (`Tokenization/`),
  `OpenAIUsage` (`Server/OpenAIUsage.swift`). Commits `f8bc3de3`, `d6df22d6` (6.x side),
  `87f92877` (Ornith variant side). The Ornith variant's duplicate copies were removed;
  `greedyParityAgainst6x` still passes (37.5s). The dense variants carry none of these
  types (their CPU engine is simpler), so nothing to unify there.
- **Consequence:** the Ornith variant's tool decoder now emits `Core.ParsedToolCall`,
  which is exactly the type `ServerInferenceEvent.toolCall` carries — the event boundary
  is already type-compatible.
- **Remaining engine-integration crux (measured, not guessed):** the backends still
  import the engine from 6.x. `CPUModelBackend` (341 lines) needs `CPUQwen35`,
  `AffineSnapshot`, `CPUSampler`, `GenerationDefaults`, `RequestReasoning` (all present
  in each dense variant) **plus `GenerationConfig`, `GFTokenizer`, `ModelThinkingMode`
  (absent from the dense variants — they live in the 6.x generation/tokenizer
  subsystems).** `ServerModelSession` (~2,234 lines) similarly needs the GPU engine +
  tokenizer + `GenerationConfig` (which cascades into the `JSONConstraint` grammar
  subsystem) + `WatchdogSet` (cascades into the 6.x-coupled detectors) +
  `ValidatedChatRequest`. So before a backend can drive a variant, those shared
  engine/tokenizer/config types must either move to Core (large cascade) or the adapter
  must be duplicated per variant. Recommended: a variant backend inside TinyTitanLib
  (which may import the variant) translating into the existing contract, lifting only
  the genuinely shared types — not the whole grammar/watchdog/validation subsystems.
- **Dense trio shipped (TT-096 Done):** all six dense variants (2B/4B/9B × 4/8-bit)
  installed via the installer and parity-verified vs 6.x (2B4 183.9s, 2B8 57.6s,
  4B4 430.2s, 4B8 164.5s, 9B4 230.2s, 9B8 677.8s). Installer resume-after-kill race in
  `prepare_qwen35.py`'s threaded downloader noted as a follow-up (fresh download avoids it).

### Owner directive 2026-11-16: strict CPU/GPU code separation, dedicated dense paths

The owner requires CPU and GPU paths **strictly code-separated**, and each dense model
to own **its own dedicated CPU and GPU code paths** — the whole point of 7.0 (no shared
family switch). Target structure per dense variant:

    sources/Engines/Qwen35Dense<N><Q>Bit/
        CPUEngine/      # the dense CPU forward (CPUQwen35) — already present, dedicated
        GPUEngine/      # the dense GPU forward (RealForwardRunner, qwen3_5_dense) — TO ADD
        Variant.swift   # exposes both, engine selectable (CPU default for dense; @gpu opt-in)

The GPU engine is config-driven (`denseFFN = cfg.numExperts == 0` in
`RealForwardRunner+DecodeLayers.swift`), so the same runner code serves the dense family;
the MoE-streaming/QSA/speculation paths are inert for a dense install. The dense GPU port
is therefore a spec-sanctioned **copy** of the GPU engine (the Ornith variant's
Kernels/Metal/Runtime/ModelIO/Streaming, ~284 files) into each dense variant's `GPUEngine/`,
with dense statics — NOT a cross-variant import (variant-isolation forbids that).

**Ordering constraint:** the GPU engine copy includes tokenizer/generation files that are
being unified into Core (stages 1–3 done: data vocabulary + reasoning types). Complete the
Core unification of `GenerationConfig`/`GenerationDefaults` (+ the `JSONConstraint` grammar
cascade) and `GFTokenizer`/`GFDetokenizer` FIRST, so the copied GPUEngine imports those from
Core instead of re-duplicating them. Then port GPUEngine per variant, wire the engine choice
into the variant surface + the server backend, and parity-check each against the 6.x GPU
engine (`TINYTITAN_DENSE_GPU_EQUIV=1` oracle exists per plan-dense-gpu-engine.md).

**Measured entanglement (2026-11-16, attempt reverted to keep the tree green):** a naive
`git mv` of the grammar subsystem + `SamplerTypes.swift` into Core does NOT compile, because:
- `JSONTokenTable` references `GFTokenizer`/`GFDetokenizer` (still in 6.x Tokenization) — so
  the tokenizer (~979 lines, needs Core → Tokenizers dep) must move first or together;
- `SamplerTypes.swift`'s `GenerationDefaults` switches on `ModelFamily` (model knowledge —
  core-purity forbids it in Core). `GenerationConfig` (generic struct) and `GenerationDefaults`
  (family-specific) must be SPLIT: the config struct to Core, the family defaults stay in the
  runtime/variants;
- `SamplerTypes.swift` also throws `GeneratorError` (6.x Generation) — move that to Core too.
Correct sequence: (a) `GeneratorError` + `GFTokenizer`/`GFDetokenizer`/`TokenizerTypes` → Core;
(b) split `GenerationConfig` (Core) from `GenerationDefaults` (runtime); (c) grammar subsystem
→ Core; (d) then the dense GPUEngine port. Each step build + parity + gate verified.

**BLOCKER measured 2026-11-16 (phase (a) attempted twice, reverted both times to keep the
tree green): Core → Tokenizers breaks TEST-target builds.** Adding `Tokenizers` (and even an
explicit `yyjson` product) to `TinyTitanCore` makes the RELEASE products build, but every
test target that depends on Core fails with `clang dependency scanning failure … unable to
resolve module dependency: 'yyjson'` (yyjson is the transitive C dep of swift-transformers).
Clearing the module cache and release-config test builds do NOT fix it. The variant tests
built fine before because Tokenizers was a DIRECT dep of the variant; once it becomes
transitive-through-Core for the many Core-dependent test targets, the explicit-module
scanner cannot resolve yyjson for them. Until this SwiftPM issue is resolved (candidate
workarounds: declare yyjson/Tokenizers directly on each Core-dependent test target, or a
build setting that disables explicit clang module scanning for tests), the tokenizer cannot
move to Core, and phases (a)–(d) are blocked. Stages 1–3 (data vocabulary + reasoning types,
which need NO Tokenizers) are landed and green. Also measured: moving GFTokenizer to Core
requires elevating a web of internal members/types to `package` (thinkStartMark/EndMark,
tokenizer, generationSuffix, GFByteLevelDecoderConfiguration + nested AddedToken,
GFDetokenizer.@usableFromInline members, byteLevelScalarToByte) — doable but broad.

The server-level CPU/GPU choice already works today via the 6.x engines (catalog `<id>` GPU
default + `<id>@cpu`; launcher `--engine cpu|gpu` + interactive menu) — verified on 2B/4B/9B
both engines. The 7.0 goal is to move that choice into the dedicated variant code paths.

## 2026-11-19 — ANE-for-Ornith diagnosis surfaced a severe install defect (TT-098)

While diagnosing why the new Ornith ANE prefill sidecar fails the live identical-greedy
gate (`benchmark/ane_prefill_correctness.py`: `on` vs `off` identical=False,
common_prefix=0; logits 5.55% vs the 2.0% bar; per-layer `max_absolute` exactly 0.0625
= the fp16 lattice step), two root causes were separated:

1. **Ornith (the actual ANE goal): the sidecar's whole CoreML program is
   `compute_precision=FLOAT16` — fp16 softmax, additive −30000 mask, no online
   rescale — against the GPU reference's fp32 flash-style online softmax.** The
   error is numeric-precision accumulation (~0.5–0.7%/layer → 5.55% logits → greedy
   argmax flip at token 1), not a wiring bug (pre-ANE layers are exactly 0.0; the
   bridge is a zero-conversion fp16 pass-through). Also on record: `tools/verify_ane_sidecar.py`
   computes its fp32 reference "from the graph's own definition" and derives scale/mask
   from the exporter's own constants, so Task-1's "0.555% NumPy PASS" can never detect
   GPU-parity divergence. And the recorded Sept-2026 "2B ANE 1.3–1.4×" baselines are
   **speed-only by the project's own README** — one greedy token, empty-string digests.
2. **Both dense 2B installs are dead (tracker TT-098, P1): the payload is 0.52%
   nonzero** (4B/9B/Ornith ~97.5%), manifest hash == receipt hash == sha256 of the
   zeros — the converter wrote them and receipt/parity/golden all self-consistently
   passed on a dead model. The 2B has never worked since the 2026-11-16 installer run;
   the AGENTS.md "2B fails instruction-following" lore is very likely this defect
   (zeroed weights → constant logits → repeated `!`). The planned "2B as ANE control"
   experiment is moot until both widths are re-installed (operator decision; the
   converter's zero-write root cause is a separate investigation — 4B/9B went through
   the same path healthy).

Fix landed: the sidecar verifier refuses vacuous comparisons (`nan > 2.0` is False in
Python — the old gate passed 0/0=nan): `compare_outputs` in
`tools/verify_ane_sidecar.py`, six `VerdictTests` first (RED on AttributeError), the
zeroed 2B now exits 1 with a named refusal, healthy Ornith still exits 0 at 0.555%.
Commit `b2cc6b5f`; tracker TT-098 + TT-097 root-cause note published on the wiki.

Next for the ANE goal: mixed-precision softmax inside the fp16 graph (cast scores →
fp32 `mb.softmax` → cast back) in `tools/export_ane_prefill.py` (currently hardcoded
`compute_precision=FLOAT16`, line ~670, coremltools 9.0), iterated with
`ane_prefill_layer_diff` per-layer error on Ornith (~34 min export per iteration), then
the identical-greedy + ≤2%-logits gate + paired benchmark. Owner's direction: ANE
worked for the Qwen 3.8 family prefill and is wanted for Ornith at the same quality bar.

## 2026-11-19 (later) — Ornith ANE rejected; the fp32 exporter changes were inert and are reverted

The staged gates ran on a full fp32-core re-export (32m14s, 10 layers, all
histories, `aneCompileVerified: true`, same weights SHA). They reproduced the
pre-fix failure exactly, and the fp32 core made the graph slower:

- verify: synthetic 0.555%, real captured normed 4.245% (both unchanged), exit 1;
- live layer diff: `on` logits 5.55%, linear control 0.000% — identical to the
  fp16 baseline;
- identical greedy: `on.identical` false, common prefix 0; the ANE arm repeats
  `S` where the GPU answers, and ANE prefill is 148.03 s against 93.56 s;
- probe with `compute_precision=FLOAT32`: 59/173 ops on the ANE (fp16 74),
  382 ms against 244 ms per layer, real error 4.248%, output changed by at most
  0.00415. The probes explain the number: k_new/v_new match at 0.059%, fp16
  rounding of q/k/v/gate after rms+rope moves the block by 0.052%, and the
  block's absolute error is ~5e-4 on both inputs (0.0934 x 0.555% synthetic,
  0.0101 x 4.245% real) — the larger relative error on the real input is
  mostly its smaller output scale. The live ANE-vs-GPU difference is 0.34-0.6%
  per layer, compounding to 5.55% logits over ten layers and flipping the
  greedy first token. Decisive: the same package scores **4.245% on CPU+ANE,
  2.419% on CPU only and 0.075% on CPU+GPU** against the fp32 reference — the
  graph is right and the ANE's fp16 accumulation is the limit (manual fp16
  and fp32 softmax compile to the same result; FLOAT32 precision leaves it at
  4.248%). Stage split on the same package: attention core 0.64%, output gate
  1.74%, wo projection 3.96%, gate+wo 4.25% — fp16 accumulation/cancellation
  (gated mean 0.008 against 0.1-0.7 partial products cancelling to 0.0101).
  Splitting wo into 4/16 fp32 partials is fused back (4.25%). The fix is
  structural: sidecar returns the core plus the pre-sigmoid gate, GPU does
  sigmoid/wo in fp32; core alone is 0.64%. Not a wiring/input/KV bug.

Decision: `d750fa2c` (softmax-only) and `ba76b292` (fp32 core) are reverted —
both were numerically inert on the live model and the core cost 1.56x per
layer. The local sidecar is `models/ornith-1.5_35B_A3B_4Bit/ane_prefill-rejected/`
(the 10-layer 4,096-token export, 541 MB); `ane_prefill` is gone, so the runtime
cannot select it. `tools/ane_sidecars.sh` now refuses to export one for
`ornith-1.5_35B_A3B_{4,8}Bit` with the measured reason (TT-099), because a sidecar
already on disk is still selected and a new install would otherwise recreate it.
Golden `ornith-4` passes with the sidecar absent (GPU path, output identical to
baseline). Evidence records are committed under `benchmark/ane-prefill/`; the
plan/spec are committed under `docs/superpowers/`.

Open for TT-099: name the semantic or fp16-boundary difference with a real-input
probe before another export, then re-run gates 1-4. Two facts bound the search. This is a regression, not an inherent limit: v4.5 measured 2.31x on the same
model at 6,103 tokens with token-for-token greedy agreement (`docs/v4.5-ane-prefill.md`). And the cache hand-off is not it: `k_new`/`v_new` match the reference in
token-major layout to 0.059% on the fp32-core sidecar (probe, not the verifier —
which checks only `out`). Do not ship a sidecar for
this install until identical-greedy passes.

The tail split, its measurements and the remaining acceptance decision are in
`docs/handover-ane-ornith.md` (2026-11-19, evening).

## 2026-11-20 — the factory surface moved out of the server; the CLI routes variants

`TinyTitanVariants` is a new target holding the 7.0 selection table and the
variant factories (`EngineFactories+Variants.swift` moved out of
`TinyTitanServerCore`). It is part of the `TinyTitanLib` product, so the CLI and
an embedder can construct a variant engine; `TinyTitanLib` itself still imports
no engine module. `EngineFactories.routingFactory(directoryURL:)` is the
single-model entry point, and `sources/TinyTitanCLI/Run.swift` plus
`CalibrationMeasurement` now call it instead of `sixXGPU()`. Goldens
`ornith-4` and `qwen35-2b-4` remain byte-identical through the routed CLI,
which is the acceptance evidence; the serial suite and the embedded-dependency
check are green. The CPU dense path followed in the same session: `CPUModelBackend` moved to
`TinyTitanCPUEngine`, `Session` now runs over any `ServerInferenceBackend`, and
the new public `CPUEngine` publishes the same facade. `TinyTitanCLI --engine
cpu` serves the dense Qwen 3.5 installs; `--engine gpu` (or no flag) keeps the
7.0 GPU routing. Greedy CPU and GPU starts agree for the first tokens and then
diverge, as the two engines' numerics differ; CPU text is not held to the GPU
golden. `TinyTitanServerCore` and the `TinyTitanServer` executable now
import the new target instead of naming the eight variant modules.
