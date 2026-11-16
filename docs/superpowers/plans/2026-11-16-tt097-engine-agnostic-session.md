# TT-097 Engine-Agnostic Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the server process generate through the 7.0 `EngineVariant` engines (Ornith GPU, dense CPU, dense GPU) instead of the 6.x runtime, by promoting a shared engine protocol into `TinyTitanCore` and typing the existing sessions against it with an injected engine factory.

**Architecture:** The server already dispatches to `ServerInferenceBackend`, which already has two impls (`ServerModelSession` GPU, `CPUModelBackend` CPU) selected by `ModelRouter`. We stop binding those sessions to concrete 6.x `Model`/`RealForwardRunner`/`CPUQwen35` and type them on one Core-resident `EngineRunner`/`EngineModel` (+ `CPUEngine*`) protocol that 6.x and every variant engine conform to. The concrete engine is constructed by an injected `EngineFactory` (kept in the server, the only place variants are importable), so `TinyTitanLib` finally stops importing 6.x for inference. One generation path; reused, not copied.

**Tech Stack:** Swift 6.4 / Xcode 27 (the only supported toolchain), SwiftPM, Metal, swift-testing (`@Test`/`#expect`), `tools/lint.sh` (20 gates), `tools/golden-baseline.sh`.

**Spec:** `docs/superpowers/specs/2026-11-16-tt097-engine-agnostic-session-design.md` (approved 2026-11-16) — read it with this plan; the plan argues from the spec.

## Global Constraints

- Toolchain: Xcode 27 / Swift 6.4 only; a build from another toolchain is not supported. Run all gates with `export PATH=".build/shims:$PATH"`.
- Apple Silicon only; never `--arch x86_64`, never `lipo`, never a universal binary.
- Output must be **byte-identical** at every phase boundary; a parity or golden failure is a finding, never re-baselined to go green. A performance change that degrades quality is rejected (AGENTS: "not bought with model quality").
- Never weaken a test to pass; never rewrite a golden to clear a mismatch. Report which side is wrong.
- Model runs are a separate, gated act: before any model run require macOS 26+, the supported toolchain, disk, acceptable `memory_pressure -Q`, a completed `.ssdai` install with a valid `verified-install.json` receipt, and no competing model process. Run tests serially (`swift test --no-parallel`), one model-using test at a time.
- Never fetch a model to satisfy a gate; an unrunnable gate is reported **not checked**.
- Do not delete 6.x. `TinyTitan` stays live as the regression engine through all four phases; retirement (TT-087) only becomes reachable after Phase 4 passes and is out of this plan.
- Commit to `main`; "online" means push. `wiki`/tracker updates are a separate repo (two-push). Sync the tracker after pushing.
- A file under `sources/` stays ≤500 physical lines; split along a cohesive seam if a session file crosses it.
- `core-purity`: nothing model-named enters `TinyTitanCore`. `variant-isolation`: a variant imports only `TinyTitanCore`/`TinyTitanFormat`/`TinyTitanKernelsC`/system — it may gain conformance to Core protocols, never import another module or the 6.x runtime.

## File Structure (what is created / modified)

Create in `TinyTitanCore`:
- `sources/TinyTitanCore/Engine/EngineModel.swift` — `EngineModel` protocol (6 members).
- `sources/TinyTitanCore/Engine/EngineRunner.swift` — `EngineRunner` protocol (behavioral members + `statistics`/`resetStatistics`).
- `sources/TinyTitanCore/Engine/EngineStatistics.swift` — struct carrying the 35 diagnostic counters.
- `sources/TinyTitanCore/Engine/CPUEngine.swift` — `CPUEngineModel`/`CPUEngineRunner`.
- `sources/TinyTitanCore/Engine/EngineFactory.swift` — `EngineFactory` value type (the injected construction surface).

Modify (conformance added, behavior preserved):
- 6.x: `sources/TinyTitan/Runtime/Inference/Model.swift`, `RealForwardRunner.swift` (+`+Diagnostics`/`+State`), `sources/TinyTitan/**/CPUQwen35.swift`.
- variants: `sources/Engines/Ornith35B4Bit/Runtime/Inference/{Model.swift,RealForwardRunner.swift}`, `sources/Engines/Qwen35Dense*Bit/CPUEngine/CPUQwen35.swift`.

Modify (protocol-typed + factory-injected):
- `sources/TinyTitanLib/ServerModelSession*.swift`, `sources/TinyTitanLib/ModelSessionPlan.swift`, `sources/TinyTitanLib/OpenAIRequestValidation.swift`.
- `sources/TinyTitanServer/Core/{CPUModelBackend.swift, ModelRouter.swift, ModelSessionPlan+Arguments.swift}`.

Create (routing + new engine):
- `sources/Engines/DenseGPU/` — shared config-driven dense-GPU engine (~273 files, a copy of the config-driven runner), conforming to `EngineRunner`.
- `sources/TinyTitanServer/Core/EngineFactories.swift` — the 6.x factory + variant factories (server imports both 6.x and variants).

New tests:
- `tests/TinyTitanCore/Engine/EngineConformanceTests.swift` — protocol witness + diagnostics-aggregate-equals-counters pin.
- `tests/TinyTitanServer/EngineRoutingTests.swift` — factory selected per identity (model-free, fake engine).

---

## Phase 1 — Core engine contracts + conformance (runtime unchanged; all-6.x; goldens must not move)

### Task 1: Pin the 35 diagnostic counters before collapsing them behind one struct

**Files:**
- Create: `tests/TinyTitanCore/Engine/EngineConformanceTests.swift`
- Reference (read, do not change yet): `sources/TinyTitan/Runtime/Inference/RealForwardRunner+Diagnostics.swift:246`

**Interfaces:**
- Produces: a test that the `EngineStatistics` aggregate in Task 3 must satisfy.

- [ ] **Step 1: Write the failing test.** Enumerate the 35 diagnostic counters via `Mirror` over the current runner and assert each `total*` counter is readable individually; record the exact property names into a `knownDiagnosticCounters: [String]` array literal in the test so a later rename is caught.

```swift
import Testing
@testable import TinyTitan // 6.x runner; the variant copy is asserted in Task 4

@Test("diagnostic counters are individually readable")
func diagnosticCountersAreEnumerated() throws {
    let expected: [String] = [ /* 35 names, filled from:
      grep -rhoE 'runner\.total[A-Za-z]+|runner\.resetKernelGPUTimings' \
        sources/TinyTitanLib/ServerModelSession*.swift | sort -u */ ]
    // Assert none is empty and the count is 35 (measured this session).
    #expect(expected.count == 35)
}
```

- [ ] **Step 2: Run it.** `swift test --no-parallel --filter EngineConformanceTests` → FAIL (names not yet filled) then PASS once filled. This task adds no production code; it is the regression pin for Task 3.
- [ ] **Step 3: Commit.** `git add tests/TinyTitanCore/Engine/EngineConformanceTests.swift && git commit -m "test(tt097 p1): pin the 35 runner diagnostic counters before the statistics aggregate"`

### Task 2: Add `EngineModel` protocol to Core

**Files:**
- Create: `sources/TinyTitanCore/Engine/EngineModel.swift`

**Interfaces:**
- Produces: `public protocol EngineModel: Sendable` with exactly these witnesses (signatures copied verbatim from the 6.x decls):
  - `var config: ArchConfig { get }` (`ArchConfig` is already Core)
  - `var modelID: String { get }`
  - `var sourceSnapshotHash: String? { get }`
  - `var routedExpertWeightBits: Int { get }`
  - `func routedExpertCacheSlotCount() -> Int`
  - `func saveExpertUsageProfile()`

- [ ] **Step 1: Create the protocol** listing the 6 members above with those exact signatures (from `Model.swift:49,50,188` and `Model+ExpertProfile.swift:36`). No model names anywhere → core-purity clean.
- [ ] **Step 2: Build.** `swift build -c release` → compiles (nothing conforms yet; protocol alone is valid).
- [ ] **Step 3: Commit.**

### Task 3: Add `EngineStatistics` + `EngineRunner` protocol to Core

**Files:**
- Create: `sources/TinyTitanCore/Engine/EngineStatistics.swift`
- Create: `sources/TinyTitanCore/Engine/EngineRunner.swift`

**Interfaces:**
- Consumes: Task 1's counter list.
- Produces:
  - `public struct EngineStatistics: Sendable, Equatable` — one `let` per each of the 35 counters (exact names/types from `RealForwardRunner+Diagnostics.swift`), read-only aggregate.
  - `public protocol EngineRunner: Sendable, ChunkedPrefillRunner, ContextWindowReporting` (reuse the Core-promoted `ChunkedPrefillRunner`/`ContextWindowReporting` — see note) with members copied verbatim: `var continuationPosition: Int { get }`, `var supportsInferenceStateRestore: Bool { get }`, `func captureInferenceState(...) -> ...`, `func captureAuxiliaryState() throws -> InferenceAuxiliaryState?`, `func restoreInferenceState(...)`, `func decodeExpertIO() -> (hits: UInt64, misses: UInt64, bytes: UInt64)?`, `var prefetchRingSummary: String? { get }`, `func expertStreamingStatistics() -> ...`, `func kernelGPUGaps() -> ...`, `func kernelGPUOccupancy() -> ...`, `func kernelGPUTimingSummary() -> ...`, `func prefillTensorOpsLogLine() -> ...`, `func reset()`, `func resetKernelGPUTimings()`, `var statistics: EngineStatistics { get }`, `func resetStatistics()`, plus every prefill/run/decode entry point `ServerModelSession+Generation.swift` calls on the runner.

- [ ] **Step 1: Promote the two protocols.** Move `ChunkedPrefillRunner` + `ContextWindowReporting` (+ their value types, e.g. `InferenceAuxiliaryState`) into `TinyTitanCore/Engine/` as the single nominal definitions; have the 6.x `LogitProducer.swift` and the variant's re-export/alias them (pure code motion; keep `@_exported`/`typealias` so existing conformances resolve unchanged).
- [ ] **Step 2: Write `EngineStatistics`** with the 35 named `let`s (from the diagnostics file).
- [ ] **Step 3: Write `EngineRunner`.** To enumerate the full member set precisely, run
  `grep -rhoE 'runner\.[a-zA-Z]+' sources/TinyTitanLib/ServerModelSession*.swift | sort -u`
  and declare exactly those (minus the `total*` which now live on `statistics`).
- [ ] **Step 4: Build.** `swift build -c release`.
- [ ] **Step 5: Commit.**

### Task 4: Conform 6.x engine types to the Core protocols

**Files:**
- Modify: `sources/TinyTitan/Runtime/Inference/Model.swift` — `extension Model: EngineModel {}`
- Modify: `sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift` — `extension RealForwardRunner: EngineRunner {}` + implement `var statistics` (reads the 35 existing counters) and `resetStatistics()` (calls the existing per-counter resets).

**Interfaces:**
- Consumes: Tasks 2–3.

- [ ] **Step 1: Write the failing conformance test** in `EngineConformanceTests.swift`:
```swift
@Test("6.x RealForwardRunner exposes statistics equal to its individual counters")
func statisticsMatchCounters(...) {
    // build/stub a runner; assert runner.statistics.totalPrefetchIssued == runner.totalPrefetchIssued
    // for all 35 (loop over Task 1's names), and resetStatistics() zeroes them.
}
```
- [ ] **Step 2: Run → FAIL** (no `statistics` yet).
- [ ] **Step 3: Add the conformances** + implement `statistics`/`resetStatistics` by forwarding to existing properties (zero recomputation).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Full suite + both goldens.** `swift test --no-parallel` then `tools/golden-baseline.sh --check 8` and `... --check 4` → byte-identical (runtime unchanged; this proves the refactor is behavior-preserving).
- [ ] **Step 6: Lint gates.** `tools/lint.sh core-purity && tools/lint.sh variant-isolation && tools/lint.sh swift-format`.
- [ ] **Step 7: Commit.**

### Task 5: Conform the variant engines + add the CPU protocols and CPU conformance

**Files:**
- Create: `sources/TinyTitanCore/Engine/CPUEngine.swift` (`CPUEngineModel`, `CPUEngineRunner`: members copied from what `CPUModelBackend.swift` calls — `var configuration`, `var threads`, `func step(...)`, `func prefill(...)`, `func reset()`, `CPUSampler` surface, `CPUModelFamily`-free).
- Modify: `sources/Engines/Ornith35B4Bit/Runtime/Inference/{Model.swift,RealForwardRunner.swift}` (+ their `+Diagnostics`/`+State`) — conform to `EngineModel`/`EngineRunner`.
- Modify: `sources/TinyTitan/**/CPUQwen35.swift` and each `sources/Engines/Qwen35Dense*Bit/CPUEngine/CPUQwen35.swift` — conform to the CPU protocols.

- [ ] **Step 1: Write the failing test** asserting `Ornith35B4Bit.RealForwardRunner` and `Qwen35Dense4B4Bit.CPUQwen35` conform (a generic function `func serve<R: EngineRunner>(_ r: R)` called with each type must compile).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Add conformances** mirroring Task 4 (members already exist on the copies; add `statistics`/`resetStatistics`).
- [ ] **Step 4: Run → PASS + full suite + goldens** (`tools/golden-baseline.sh --check 8|4`). Variant code is not on the server path yet, but the suite compiles all variants; goldens unchanged.
- [ ] **Step 5: Gates:** `variant-isolation`, `core-purity`, `swift-format`, `file-length`.
- [ ] **Step 6: Commit.**

---

## Phase 2 — Protocol-typed sessions + injected factory (still the 6.x factory; goldens must not move)

### Task 6: Define the `EngineFactory` injection surface

**Files:**
- Create: `sources/TinyTitanCore/Engine/EngineFactory.swift`
- Reference: `sources/TinyTitanLib/ModelSessionPlan.swift:149` (current `makeSession` → `ServerModelSession.load`), `sources/TinyTitanLib/ServerModelSession+Loading.swift:222,319,339` (`Model.load`, sidecar, `RealForwardRunner(...)`).

**Interfaces:**
- Produces: `public struct EngineFactory: Sendable` with
  `let buildModel: @Sendable (URL, ArchConfig, MetalContext?) throws -> any EngineModel`
  and `let buildRunner: @Sendable (any EngineModel, SessionBuildOptions) throws -> any EngineRunner`,
  plus a `SessionBuildOptions` struct carrying the ~18 non-engine params `makeSession` threads today (maxContext, slots, prompt-cache fields, prefillChunkTokens, kvCachePrecision, thinkingMode, reasoningEffort, expert-cache, …).

- [ ] **Step 1–3:** write the type, build, commit.

### Task 7: Type `ServerModelSession` on the protocols; move construction into the factory

**Files:**
- Modify: `sources/TinyTitanLib/ServerModelSession.swift:57,72,171,175` — `let model: any EngineModel`, `let runner: any EngineRunner`, init params accordingly.
- Modify: `sources/TinyTitanLib/ServerModelSession+Loading.swift` — `load(...)` now takes an `EngineFactory` and calls it instead of `Model.load`/`RealForwardRunner(...)` directly.
- Modify: `sources/TinyTitanLib/ModelSessionPlan.swift:makeSession` — accept + forward the factory.
- Modify: `sources/TinyTitanLib/OpenAIRequestValidation.swift` — drop the now-unused `import TinyTitan` (its only 6.x use was `GFTokenizer.*`, already Core).

- [ ] **Step 1: Write the failing test** in `EngineRoutingTests.swift`: a `ServerModelSession` built from a fake `EngineFactory` returning stub `EngineModel`/`EngineRunner` loads and reports `maximumContext` (proves the session no longer needs a concrete engine).
- [ ] **Step 2: Run → FAIL** (session still concrete-typed).
- [ ] **Step 3: Re-type the session + init + load + makeSession** to the protocols/factory. If any `ServerModelSession+*.swift` file crosses 500 lines during the edit, split along its existing seam.
- [ ] **Step 4: Provide the 6.x factory** in `TinyTitanLib` as a package-private default that closes over 6.x `Model.load`/`RealForwardRunner` (so Lib still builds a working session pre-Phase-3). NOTE: this is the one place Lib still references 6.x construction; it is removed in Phase 3 when the server supplies the factory and the 6.x default factory moves to the server target.
- [ ] **Step 5: Run → PASS + full suite + both goldens byte-identical + parity vs 6.x.**
- [ ] **Step 6: Gates** incl. `stdout-clean` over the Lib closure (the diagnostics aggregate logging must go to stderr).
- [ ] **Step 7: Commit.**

### Task 8: Type `CPUModelBackend` on `CPUEngineRunner` with injection

**Files:**
- Modify: `sources/TinyTitanServer/Core/CPUModelBackend.swift:117,185` (`CPUQwen35(snapshot:)`, `generate`) → build via an injected `CPUEngineFactory`; store `any CPUEngineRunner`.

- [ ] **Step 1: failing test** (backend builds from a stub CPU engine). **Step 2: FAIL. Step 3: implement + 6.x default factory.** **Step 4: PASS + suite + golden.** **Step 5: commit.**

### Task 9: Move the 6.x default factory out of Lib into the server; Lib imports no engine module for inference

**Files:**
- Create: `sources/TinyTitanServer/Core/EngineFactories.swift` — `EngineFactories.sixXGPU`, `EngineFactories.sixXCPU` (close over 6.x `Model`/`RealForwardRunner`/`CPUQwen35`).
- Modify: remove the Phase-2 6.x default factory from `TinyTitanLib`; `makeSession` now requires an explicit factory.
- Verify: `Package.swift` — confirm `TinyTitanLib` target dependencies no longer require `TinyTitan` for the inference path (keep only if a value type still needs it; if so, move that type to Core or the factory).

- [ ] **Step 1: grep proves the coupling.** `grep -rn 'import TinyTitan' sources/TinyTitanLib/*.swift | grep -v '@testable\|Tests'` — enumerate remaining 6.x uses; each must be either moved to the factory (engine construction) or to Core (value type). List them in the commit message.
- [ ] **Step 2: relocate each.**
- [ ] **Step 3: build + full suite + both goldens + parity.** This is the checkpoint where the embeddable facade is genuinely 6.x-free for inference.
- [ ] **Step 4: `stdout-clean`, `library-facade` gates** (the facade's public surface must not grow; `FACADE_UPDATE=1` only with a decision recorded in `docs/plan-embedded-library.md`).
- [ ] **Step 5: Commit.**

---

## Phase 3 — Variant factories + per-identity routing (models begin switching to the variant; gate each flip)

### Task 10: Variant GPU + CPU factories in the server

**Files:**
- Modify: `sources/TinyTitanServer/Core/EngineFactories.swift` — add `variantGPUOrnith` (imports `Ornith35B4Bit`, builds its `Model`+`RealForwardRunner`) and `variantCPUQwen35` (imports the `Qwen35Dense*` targets, builds `CPUQwen35`).
- Modify: `Package.swift` — add those variant targets to `TinyTitanServerCore` dependencies (server→variant import is allowed).

- [ ] **Step 1: failing routing test** — given identity `ornith-1.5-35b-4bit`, `EngineFactories.for(...)` returns the variant GPU factory (compare factory identity, not a model run). **Step 2: FAIL. Step 3: implement factories + selection table.** **Step 4: PASS (model-free).** **Step 5: build + suite + gates.** **Step 6: commit.**

### Task 11: Route Ornith-4bit to the variant GPU engine (model-gated flip #1)

**Files:**
- Modify: `sources/TinyTitanServer/Core/ModelRouter.swift` + `ModelSessionPlan+Arguments.swift` — for `.gpu` entries whose identity is a shipped variant, select `variantGPUOrnith`.

- [ ] **Step 1: Model-run preconditions** (macOS 26+, toolchain, disk, `memory_pressure -Q`, receipt valid, no competing process). If any fails, report not-checked and stop.
- [ ] **Step 2: run the server against `models/ornith-1.5_35B_A3B_4Bit`, issue a chat + a streaming request + a tool request; confirm output.**
- [ ] **Step 3: `tools/golden-baseline.sh --check 4` (Ornith 4-bit) → byte-identical.** If the variant path diverges, that is a finding — stop and report, do not re-baseline.
- [ ] **Step 4: unit parity** `TINYTITAN_ORNITH_PARITY_MODEL=models/ornith-1.5_35B_A3B_4Bit swift test --no-parallel --filter Ornith35B4BitTests` → PASS.
- [ ] **Step 5: paired benchmark** vs the 6.x server path (per paired-benchmarking skill); adopt only if not slower/quality-degrading.
- [ ] **Step 6: Commit** the routing for Ornith.

### Task 12: Build the shared config-driven dense-GPU engine (spec D4)

**Files:**
- Create: `sources/Engines/DenseGPU/**` — a spec-sanctioned copy of the config-driven GPU engine (273 files) for family `qwen3_5_dense` (`numExperts == 0`), geometry read from the manifest, **no runtime family switch**. Conform its `Model`/`RealForwardRunner` to `EngineModel`/`EngineRunner` (as Task 5).
- Modify: `Package.swift` — new `DenseGPU` variant target importing only `TinyTitanCore`/`TinyTitanFormat`/`TinyTitanKernelsC`; `DenseGPUTests` test target.

- [ ] **Step 1: copy** the config-driven runner from the Ornith variant into `DenseGPU/` (pure copy), then set its dense constants (no routed experts).
- [ ] **Step 2: gates** `variant-isolation` (no 6.x / no cross-variant import), `core-purity`, `file-length`.
- [ ] **Step 3: conformance test** (generic `serve<R: EngineRunner>` compiles with `DenseGPU.RealForwardRunner`).
- [ ] **Step 4: build + commit.**

### Task 13: Route dense CPU + dense GPU to their variant engines (model-gated flips #2–N)

**Files:**
- Modify: `EngineFactories.swift` — add `variantDenseGPU` (imports `DenseGPU`); the CPU factory already exists (Task 10).
- Modify: `ModelRouter.swift` — `@cpu` → `variantCPUQwen35`; `@gpu`/bare dense → `variantDenseGPU`; bare dense default = GPU.

- [ ] **Step 1: preconditions.**
- [ ] **Step 2–4: for each dense install × engine** (2B/4B/9B × {CPU, GPU} × {4,8-bit}): golden `--check`, parity vs 6.x (`dense*BitParity`, `TINYTITAN_DENSE_GPU_EQUIV=1` oracle), a streamed + tool + prompt-cache + reasoning-split request. Gate each; a mismatch is a finding. (Use 4B first when one suffices; run serially, one at a time.)
- [ ] **Step 5: paired benchmark** per engine vs 6.x; adopt only if not regressing.
- [ ] **Step 6: commit.**

---

## Phase 4 — Acceptance bar, records, and reachable retirement

### Task 14: OpenCode queued long-prompt prefill on the variant path

- [ ] **Step 1: preconditions.** Run the previously-deadlocked 5,818-token prompt through the server on the Ornith variant path → completes, golden byte-identical (confirms the TT-095 `parallelFor` fix is carried into the variant, which Task 5 conformed but must still hold under the session).
- [ ] **Step 2: commit any variant parity gap found.**

### Task 15: Full feature-surface end-to-end matrix

- [ ] Per model: streaming deltas, tool calls, explicit `temperature`/`top_p`/`top_k` overriding family defaults, prompt-cache reuse across turns, reasoning split (`<think>`), and the `--engine cpu|gpu` + `<id>@cpu/@gpu` selection. Record commands + exit codes + outputs. Report any not-checked (no install) honestly.

### Task 16: Update records and flip tracker rows

- [ ] Registry: add/qualify dense-GPU rows if new installs are goldened; keep `catalog --check` consistent (`tools/lint.sh converter`, `docs`).
- [ ] Tracker (wiki, two-push): TT-082 "server-serves-variant" moves from deferred → Done; TT-097 → Done; TT-087 becomes reachable (only flip it to Done if 6.x is actually removed — otherwise leave Won't Do with a note that the prerequisite is now met).
- [ ] README/wiki: state that the server runs the 7.0 variant engines for Ornith + dense (CPU+GPU), with the exact commands.
- [ ] `docs/handover-tinytitan-7.md`: record the completed migration + any deviations.
- [ ] Run `tools/release.sh`-adjacent gates + full suite + all goldens; commit + push both repos.

---

## Self-Review (author checklist, completed inline)

1. **Spec coverage:** §4-D1 (Core protocols) → Tasks 2–3; conformance 6.x+variants → Tasks 4–5; §4-D2 (protocol-typed sessions + injection + Lib 6.x-free) → Tasks 6–9; §4-D3 (routing) → Tasks 10–13; §4-D4 (dense-GPU) → Task 12; §6 phasing → phases as grouped; §7 acceptance → Tasks 11,13–15; §8-R1 diagnostics pin → Task 1; R2 dynamic dispatch → measured in the Phase 2/3 paired benchmarks (Tasks 7,11,13), with the generic-specialization fallback named there. No spec requirement lacks a task.
2. **Placeholders:** every protocol member list is sourced from an exact `grep`/file:line the implementer runs; the 35-counter list is pinned in Task 1. No TBD/TODO.
3. **Type consistency:** `EngineModel`/`EngineRunner`/`EngineStatistics`/`EngineFactory`/`CPUEngineRunner` names and the 6 `Model` members are used identically across Tasks 2–13; factory member names (`buildModel`/`buildRunner`) are defined once in Task 6 and consumed in Tasks 7–13.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-11-16-tt097-engine-agnostic-session.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. **Constraint from memory: agents are unavailable in this session** (`sonnet`/`opus` model errors, general-purpose disabled), and Phase 4 tasks are paired-benchmark/model-gated runs that must NOT be delegated to a concurrent subagent (single-purpose machine). So subagent-driven would fall back to inline for any task needing a model run.
2. **Inline Execution** — execute task-by-task in this session with checkpoints (executing-plans), which the model-gated and benchmark phases require anyway.
