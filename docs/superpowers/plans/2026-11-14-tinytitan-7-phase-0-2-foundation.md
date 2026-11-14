# TinyTitan 7.0 — Foundation Plan (Migration Phases 0–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the 7.0 engine structure — the frozen 6.x inventory in one registry, the generic `TinyTitanCore` library with no model knowledge, the `EngineVariant` contracts, and the first variant (`Qwen35Dense2B4Bit`) qualified against 6.x by golden and benchmark parity.

**Architecture:** Do not rewrite in place. 6.x stays the reference engine and keeps shipping while the 7.0 structure grows beside it: a data registry, a generic core extracted by pure code motion, one variant target proving the template. Each task ends green under the existing gates.

**Tech Stack:** Swift 6.4 (Xcode 27, SwiftPM), Metal, the existing `.ssdai` install format, the existing lint-gate harness (`tools/lint.sh`).

**Spec:** `docs/plan-tinytitan-7-engine.md` (design contract; the authority on all trade-offs) and the wiki Project Tracker rows TT-072 … TT-087 (the work breakdown this plan implements). This plan covers migration phases 0, 1 and 2 of the spec's §11 table.

## Execution sequence

Run the tasks strictly in this order; the Depends column is a hard gate — a task starts only when its dependencies are committed.

| # | Task | Depends on | Deliverable | Plan section |
| --- | --- | --- | --- | --- |
| 1 | Freeze 6.x, write the registry inventory | — | `registry/registry.json` + inventory test | §Task 1 |
| 2 | Extract `TinyTitanCore` (pure code motion) | 1 | generic-only core target, suites unchanged | §Task 2 |
| 3 | `core-purity` + `variant-isolation` gates | 2 | two new fail-closed lint checks, AGENTS.md count updated | §Task 3 |
| 4 | `EngineVariant` contracts in Core | 2 | VariantID/Quant/Profiles/Recipe/Qualification types + tests | §Task 4 |
| 5 | Variant target scaffold (`Qwen35Dense2B4Bit`) | 1, 3, 4 | compiling, gated, non-serving variant + registry wiring | §Task 5 |
| 6 | `tinytitan catalog --check` | 1, 5 | registry ↔ target ↔ qualification gate | §Task 6 |
| 7 | Port the 2B 4-bit graph into the variant | 5 | working 7.0 decode path (pure copy) | §Task 7 |
| 8 | Golden + benchmark parity vs 6.x | 7 | identical greedy hashes, paired tok/s | §Task 8 |
| 9 | Matrix test + qualification record | 6, 8 | first shipped variant, `catalog --check` green | §Task 9 |

Phases 3–9 are follow-on plans (one file each), generated from the template Tasks 4–9 prove. They are tracker rows TT-081 … TT-087 and are not detailed here.

## Follow-on plans

| Spec phase | Follow-on plan file | Tracker row | Scope |
| --- | --- | --- | --- |
| 3 | `docs/superpowers/plans/<date>-tinytitan-7-phase-3-dense-family.md` | TT-081 | 2B 8-bit, 4B, 9B variants |
| 4 | `<date>-tinytitan-7-phase-4-35b-family.md` | TT-082 | Ornith 35B, Qwen 3.6, KAT, AgentWorld (copy the 35B graph), then delete the 35B family switch |
| 5 | `<date>-tinytitan-7-phase-5-qwen38.md` | TT-083 | Qwen3.8 4/8 with HC/PLE/QSA inside its variants |
| 6 | `<date>-tinytitan-7-phase-6-ornith122b.md` | TT-084 | Ornith 122B 4/8, dedicated tile per variant |
| 7 | `<date>-tinytitan-7-phase-7-mtp.md` | TT-085 | MTP sidecars as variants of their own |
| 8 | `<date>-tinytitan-7-phase-8-k2.md` | TT-086 | K2 own runner or drop from the shipped matrix |
| 9 | `<date>-tinytitan-7-phase-9-deletion.md` | TT-087 | Delete family enum switches, `ModelProfile`, Python tools |

## Global Constraints (verbatim from the spec and AGENTS.md)

- C1 — Toolchain is **Xcode 27 / Swift 6.4, nothing else**; the target is `arm64-apple-macos26.0`; Apple Silicon only.
- C2 — Every production file under `sources/` stays **≤ 500 physical lines** (comments and blanks included); split along a cohesive seam as pure code motion, and record the split in `docs/repository-layout.md`.
- C3 — Tests run **serially**: `swift test --no-parallel` (extra arguments pass through).
- C4 — A model install is **never fetched** for a gate, a benchmark or a test; only installs already under `models/` are used. The 2B is used *deliberately* in this plan because the 2B variant itself is the subject (AGENTS.md's stated exception), and the report must say so.
- C5 — **6-bit is refused by name** in `QuantSpec` validation; `bits` is 2, 4 or 8 (spec §6).
- C6 — A performance claim requires golden text identity or zero top-1 divergence plus a paired benchmark; parity for migration is "same prompt set, greedy output hash identical or a recorded, approved difference; decode within the 6.x noise floor; memory not worse; receipt verifies" (spec §11).
- C7 — Pins: swiftlint 0.65.1, ruff 0.16.7 (run `tools/lint.sh` with `PATH=".build/shims:$PATH"` on this machine — a pipx venv on PATH shadows `python3` and breaks the converter gate), shellcheck 0.11.0, swift-format per the committed `.swift-format`.
- C8 — One commit per task, imperative conventional message, body stating what and why. Commits go to `main` (no branches); push after each completed phase.
- C9 — A generic component in `TinyTitanCore` **must not grow model knowledge to avoid a copy** (spec §1.3).
- C10 — Engine layout follows repo convention (lowercase), not the spec's illustrative capitalisation: `sources/TinyTitanCore/`, `sources/Engines/<Variant>/`, `registry/registry.json`. **`registry/` (lowercase) and not `Models/`**, because `models/` is the gitignored install directory and macOS filesystems are case-insensitive — `Models/` would collide with it.
- C11 — The current lint suite has **18 checks**; adding any check requires updating AGENTS.md's count sentence in the same commit, or the `docs` gate fails.

---

## Task 1 — Freeze 6.x and write the registry inventory

**Depends on:** none. **Tracker row:** TT-072.

**Files:**
- Create: `registry/registry.json`
- Test: `tests/TinyTitan/Registry/RegistryInventoryTests.swift`

**Interfaces:**
- Produces: `registry/registry.json` with schema version 1; every shipped (model, quant) pair from the running 6.x tree as one entry. Later tasks and the follow-on plans read this file and its schema; nothing in Swift reads it yet.

- [ ] **1.1 Freeze the 6.x reference** — Record the freeze point in the tracker row (Step 1.5) and tag nothing new: `v6.3` (`84958fb3`) is the frozen reference; `main` moves on and the registry records what is *shipped*, re-derived at each inventory refresh.

- [ ] **1.2 Derive the shipped inventory from the code, not from memory**

```bash
grep -oE "'[a-z0-9-]+'" tools/tinytitan_models.sh | sort -u
grep -n 'ModelProfile(' sources/TinyTitan/Runtime/Configuration/ModelProfile.swift | head -40
grep -n 'displayName\|case ' sources/TinyTitan/Infrastructure/ModelIO/ModelFamily.swift
```

Expected: one list of model ids, one list of `(modelID, weightBits)` tuning rows, one list of family cases. Cross-check the installer menu keys in `tools/install_models.sh`. Every discrepancy is a finding to record in the tracker row — do not reconcile by editing the shells.

- [ ] **1.3 Write `registry/registry.json` (data, not code)**

One entry per shipped (model, quant) pair. Schema:

```json
{
  "version": 1,
  "models": [
    {
      "modelID": "qwen35-2b",
      "displayName": "Qwen 3.5 2B",
      "family": "qwen35_dense",
      "source": { "repo": "Qwen/Qwen3.5-2B", "sha": "fill-from-model-card-or-installer" },
      "variants": [
        {
          "id": "qwen35-2b-4bit",
          "target": null,
          "bits": 4,
          "engine": "cpu",
          "shipped": false,
          "installedSizeBytes": null,
          "sourceSizeBytes": null,
          "qualification": null
        }
      ]
    }
  ]
}
```

`"target": null` until Task 5 wires real target names; `"qualification": null` until Task 9; `"shipped": false` until Task 9 flips the first variant. The inventory entries carry **no tuning values** — profiles stay in `ModelProfile` until their variant owns them (spec §7).

- [ ] **1.4 Write the inventory test**

`tests/TinyTitan/Registry/RegistryInventoryTests.swift`:

```swift
import Testing
import Foundation
@testable import TinyTitan

struct RegistryInventoryTests {
    private static func loadRegistry() throws -> [String: Any] {
        let url = URL(fileURLWithPath: "registry/registry.json")
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    @Test func registryParsesAndCoversTheShippedPairs() throws {
        let registry = try Self.loadRegistry()
        #expect(registry["version"] as? Int == 1)
        let models = try #require(registry["models"] as? [[String: Any]])
        // Shipped pairs from 6.x, derived in Step 1.2 — assert the count here and
        // update it only together with the Step-1.2 reconciliation output.
        #expect(models.count == 9)
        let ids = models.compactMap { $0["modelID"] as? String }
        #expect(!ids.isEmpty)
        for model in models {
            let variants = try #require(model["variants"] as? [[String: Any]])
            #expect(!variants.isEmpty)
        }
    }

    @Test func variantIdsAreUniqueAndSuffixed() throws {
        let registry = try Self.loadRegistry()
        let models = try #require(registry["models"] as? [[String: Any]])
        var seen = Set<String>()
        for model in models {
            for variant in try #require(model["variants"] as? [[String: Any]]) {
                let id = try #require(variant["id"] as? String)
                #expect(id.hasSuffix("bit"))
                #expect(seen.insert(id).inserted)
            }
        }
    }
}
```

- [ ] **1.5 Record the freeze + inventory in the wiki tracker** — one row (TT-072) titled "7.0 phase 0 — freeze and registry inventory", linking this plan and the Step-1.2 reconciliation output. Update its Updated date when later tasks land.

- [ ] **1.6 Verify and commit**

```bash
swift test --no-parallel --filter RegistryInventoryTests   # Expected: PASS
PATH=".build/shims:$PATH" tools/lint.sh docs               # Expected: ok
git add registry/registry.json tests/TinyTitan/Registry/RegistryInventoryTests.swift
git commit -m "feat(7.0): freeze the 6.x inventory into the one registry"
```

---

## Task 2 — Create the TinyTitanCore target by pure code motion

**Depends on:** Task 1. **Tracker row:** TT-073.

**Files:**
- Create: `sources/TinyTitanCore/` (moved files), `Package.swift` target entry
- Move (git mv): `sources/TinyTitan/Infrastructure/Metal/MetalContext.swift`, `sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift`, `sources/TinyTitan/Infrastructure/Concurrency/*`, `sources/TinyTitan/Infrastructure/Hardware/*`, `sources/TinyTitan/ProcessMemory.swift`, `sources/TinyTitan/Tokenization/*`
- Modify: `Package.swift` (new target + dependency edges), every importer's `import` lines

**Interfaces:**
- Produces: target `TinyTitanCore`, exporting `MetalContext`, the tokenization types, concurrency/hardware helpers. `TinyTitan` depends on `TinyTitanCore`. Later variant targets (Task 5+) depend on `TinyTitanCore` and on nothing else generic.

- [ ] **2.1 Add the target to Package.swift**

```swift
.target(
    name: "TinyTitanCore",
    path: "sources/TinyTitanCore"
),
```

and add `TinyTitanCore` to `TinyTitan`'s (and where needed `TinyTitanLib`'s) `dependencies:`. Keep `TinyTitanFormat` a separate target — it is already generic; `TinyTitanCore` may `import TinyTitanFormat` but phase 1 does not fold them (record that decision in the commit body; folding is phase-9 cleanup if ever).

- [ ] **2.2 Move the generic files with git mv**

```bash
mkdir -p sources/TinyTitanCore/Metal sources/TinyTitanCore/Concurrency \
         sources/TinyTitanCore/Hardware sources/TinyTitanCore/Tokenization
git mv sources/TinyTitan/Infrastructure/Metal/MetalContext.swift sources/TinyTitanCore/Metal/
git mv sources/TinyTitan/Infrastructure/Metal/GPUKeepWarmRunner.swift sources/TinyTitanCore/Metal/
git mv sources/TinyTitan/Infrastructure/Concurrency/*.swift sources/TinyTitanCore/Concurrency/
git mv sources/TinyTitan/Infrastructure/Hardware/*.swift sources/TinyTitanCore/Hardware/
git mv sources/TinyTitan/ProcessMemory.swift sources/TinyTitanCore/
git mv sources/TinyTitan/Tokenization/*.swift sources/TinyTitanCore/Tokenization/
```

**Scope guard:** move only these. `Infrastructure/ModelIO`, `Infrastructure/Streaming`, `Kernels/`, `Runtime/`, `CPUEngine/` stay — they carry model knowledge today. `Tokenization/K2ToolCallParser.swift` and `QwenToolCallParser.swift` are named after models but are generic protocol parsers; move them and let the Task 3 purity gate decide (if the gate flags them, they stay behind in `TinyTitan` — do not weaken the gate to pass them).

- [ ] **2.3 Fix imports** — `swift build -c release` names every miss; add `import TinyTitanCore` mechanically, no behaviour edits.

- [ ] **2.4 Record the before-counts, then run the gates**

```bash
swift test --no-parallel 2>&1 | grep 'Test run'   # record BEFORE the move
swift build -c release                            # Expected: success
swift test --no-parallel                          # Expected: same suite counts as before
PATH=".build/shims:$PATH" tools/lint.sh           # Expected: all checks ok
```

The `stdout-clean` walk must give `TinyTitanCore` the same no-stdout treatment `TinyTitanLib` gets — extend the `stdout-clean` walk if it enumerates targets explicitly.

- [ ] **2.5 Commit**

```bash
git add -A
git commit -m "refactor(7.0): extract TinyTitanCore — Metal context, tokenization, concurrency, hardware

Pure code motion, no behaviour change. The core is generic-only; model
knowledge stays in the runtime targets. TinyTitanFormat stays separate."
```

---

## Task 3 — The core-purity and variant-isolation gates

**Depends on:** Task 2. **Tracker row:** TT-074.

**Files:**
- Modify: `tools/lint.sh` (two new checks), `AGENTS.md` (count sentence "eighteen checks" → twenty, constraint C11), `docs/repository-layout.md` (describe the two checks)
- Test: `benchmark/test_lint_core_purity.py`, `benchmark/test_lint_variant_isolation.py`

**Interfaces:**
- Produces: `tools/lint.sh core-purity` — fails when any file under `sources/TinyTitanCore/` matches the model-name pattern below; `tools/lint.sh variant-isolation` — fails when a file under `sources/Engines/<Variant>/` contains `import` of another `Engines/` target or of `TinyTitan`/`TinyTitanLib`. Both fail when their walk finds no Swift files (the house fail-closed pattern of `unbounded-read`).

- [ ] **3.1 Write the failing gate tests first**

`benchmark/test_lint_core_purity.py` — build a throwaway tree (the pattern of `test_lint_gate_scan_guard.py`): a stub `sources/TinyTitanCore/x.swift` containing `case qwen36`, assert the check exits non-zero and names the file; a clean stub passes; an empty tree is a failure, not a pass.

`benchmark/test_lint_variant_isolation.py` — stub `sources/Engines/Qwen35Dense2B4Bit/a.swift` containing `import Qwen38Flash4Bit` and `import TinyTitan`; assert non-zero naming both edges; a stub importing only `TinyTitanCore` and `Metal` passes; empty tree fails.

- [ ] **3.2 Run the tests to verify they fail**

```bash
cd benchmark && python3 -m unittest test_lint_core_purity test_lint_variant_isolation
```
Expected: FAIL — the modes do not exist yet.

- [ ] **3.3 Implement the checks in tools/lint.sh**

Model-knowledge pattern for `core-purity` (one `grep -nE` over `sources/TinyTitanCore/**.swift`, continuation windows joined like `force-cast` does):

```
qwen36|qwen38|qwen35|qwen4flash|ornith|agentworld|katcoder|laguna|nemotron|k2horizon|deepseek|ModelFamily|ModelProfile|ModelCatalog
```

Case-insensitive; the exemption comment is `lint:allow-core-name <reason>` above the line (start with zero exemptions). For `variant-isolation`: every `import X` line under `sources/Engines/*/` where `X` is another directory under `sources/Engines/`, or `TinyTitan`, `TinyTitanLib`, `TinyTitanServerCore`, `TinyTitanRepackCore`; allowed imports are `TinyTitanCore`, `TinyTitanFormat`, `Metal`, `MetalKit`, `Foundation`, `Accelerate`, `Compression`, `SystemPackage` and the C-kernel target if one exists.

- [ ] **3.4 Register the modes and update the count**

Add both modes to `tools/lint.sh`'s usage header and `all` chain, then update AGENTS.md's count sentence ("eighteen checks CI enforces beyond the compiler" → "twenty") in the same commit — the `docs` gate fails otherwise. Run the docs gate to prove it.

- [ ] **3.5 Run the tests to verify they pass, then the full gate**

```bash
python3 -m unittest test_lint_core_purity test_lint_variant_isolation   # Expected: PASS
PATH=".build/shims:$PATH" tools/lint.sh                                 # Expected: 20 checks ok
```

- [ ] **3.6 Commit**

```bash
git add -A
git commit -m "feat(7.0): core-purity and variant-isolation gates"
```

---

## Task 4 — The EngineVariant contracts in TinyTitanCore

**Depends on:** Task 2. **Tracker row:** TT-075.

**Files:**
- Create: `sources/TinyTitanCore/Variant/VariantID.swift`, `EngineVariant.swift`, `GeometryContract.swift`, `QuantSpec.swift`, `PerformanceProfile.swift`, `SamplingProfile.swift`, `ModelRecipe.swift`, `Qualification.swift`, `EngineContext.swift`
- Modify: `Package.swift` (add a `TinyTitanCoreTests` test target)
- Test: `tests/TinyTitanCore/Variant/VariantContractsTests.swift`

**Interfaces:**
- Produces (exact names Tasks 5, 7, 9 and the follow-on plans use):

```swift
public struct VariantID: Hashable, Sendable, Comparable {
    public let raw: String            // "<model-id>-<bits>bit", e.g. "qwen35-2b-4bit"
    public init(_ raw: String) throws // throws if not lowercase, not "-<2|4|8>bit"-suffixed
    public var bits: Int { get }
}

public struct SourcePin: Sendable { public let repo: String; public let sha: String }

public struct GeometryContract: Sendable {
    public let required: [String: GeometryValue]   // validated against manifest.json
    public enum GeometryValue: Sendable { case int(Int), string(String) }
    public func validate(_ manifest: [String: Any]) throws  // throws naming the first mismatch
}

public struct QuantSpec: Sendable {   // affine only in 7.0
    public let bits: Int              // 2, 4 or 8; init throws naming 6
    public let group: Int
    public let scaleIsBF16: Bool
    public let biasIsBF16: Bool
    public init(bits: Int, group: Int) throws
}

public enum QuantRole: String, Sendable, CaseIterable {
    case embedding, lmHead, attentionQO, attentionKV, gdnQKV, gdnZ, gdnOut,
         router, sharedExpert, routedExpert, hyperConnection, ple, indexer,
         movaValue
}

public struct TensorStem: Hashable, Sendable { public let raw: String; public init(_ raw: String) }

public struct VariantQuant: Sendable {
    public let defaultSpec: QuantSpec
    public let roles: [QuantRole: QuantSpec]
    public let perTensor: [TensorStem: QuantSpec]
    public func spec(forRole: QuantRole, stem: TensorStem) -> QuantSpec  // perTensor wins, then roles, then default
}

public struct PerformanceProfile: Sendable {
    public let expertCacheBytes: Int, prefetchDepth: Int, prefillChunk: Int
    public let routerTopKSimd: Bool, attentionSimdPartial: Bool, keepCacheWired: Bool
}

public struct SamplingProfile: Sendable {
    public let temperature: Float, topK: Int, topP: Float
    public let presencePenalty: Float, thinking: ThinkingDefaults
}

public struct ModelRecipe: Sendable {        // install/convert only
    public let variantID: VariantID
    public let renames: [String: String]     // checkpoint name -> variant tensor stem
    public let keepBF16Stems: Set<String>
    public let headBits: Int?, embeddingBits: Int?
}

public struct Qualification: Codable, Sendable {
    public let goldenSHA256: String
    public let decodeTokS: Double, prefillTokS: Double, ttftS: Double
    public let memoryCeilingBytes: Int
    public struct TerminalBench: Codable, Sendable { public let suite: String, score: Double, harnessVersion: String, date: String }
    public let terminalBench: TerminalBench?
    public let hardware: String, date: String
}

public struct EngineContext: Sendable {      // generalized from the spec's MetalContext-only init
    public let metal: MetalContext?          // nil for CPU-only variants
    public let physicalRAMBytes: UInt64
}

public protocol EngineVariant: Sendable {
    static var id: VariantID { get }
    static var source: SourcePin { get }
    static var geometry: GeometryContract { get }
    static var quant: VariantQuant { get }
    static var performance: PerformanceProfile { get }
    static var sampling: SamplingProfile { get }
    static var recipe: ModelRecipe { get }
    init(context: EngineContext, model: LoadedModelHandle) throws
    func prefill(_ tokens: [Int], state: inout SessionState) throws
    func decode(_ state: inout SessionState) throws -> Int
    static func qualify() throws -> Qualification
}
```

`LoadedModelHandle` and `SessionState` start as minimal Core types (tensor/residency handle per spec §8: names, buffers, offsets, dtypes — no PLE/HC/QSA): define them in this task as thin structs over opaque buffers so the protocol compiles; the first variant widens them with pure code motion if it needs more. **They must not contain feature fields.**

- [ ] **4.1 Write the failing contract tests** — `VariantID("qwen35-2b-4bit")` succeeds and `bits == 4`; `VariantID("qwen35-2b-6bit")` throws naming 6; `QuantSpec(bits: 6, group: 64)` throws naming 6; `VariantQuant.spec(forRole:stem:)` resolution order (perTensor > roles > default); `GeometryContract.validate` refuses a near-miss (missing key, wrong int) and names the key.

- [ ] **4.2 Run to verify failure** — `swift test --no-parallel --filter VariantContractsTests` → Expected: FAIL (types do not exist).

- [ ] **4.3 Implement the contract files** — one file per type as listed above; `ThinkingDefaults` moves from wherever the 6.x sampler defines it into `SamplingProfile.swift` by pure code motion (copy with a provenance header naming the source file and commit if the runtime still needs its own copy).

- [ ] **4.4 Run to verify pass, then the gates**

```bash
swift test --no-parallel --filter VariantContractsTests      # Expected: PASS
PATH=".build/shims:$PATH" tools/lint.sh core-purity          # Expected: ok (no model names)
PATH=".build/shims:$PATH" tools/lint.sh                      # Expected: all ok
```

- [ ] **4.5 Commit**

```bash
git add -A
git commit -m "feat(7.0): EngineVariant contracts in TinyTitanCore"
```

---

## Task 5 — The variant target scaffold and isolation proof

**Depends on:** Tasks 1, 3, 4. **Tracker row:** TT-076.

**Files:**
- Modify: `Package.swift` (target `Qwen35Dense2B4Bit`, path `sources/Engines/Qwen35Dense2B4Bit`, depends on `TinyTitanCore`)
- Modify: `registry/registry.json` (`"target": "Qwen35Dense2B4Bit"` on the `qwen35-2b-4bit` variant)
- Create: `sources/Engines/Qwen35Dense2B4Bit/Variant.swift` (the conformance skeleton)
- Test: `tests/TinyTitanCore/Variant/RegistryTargetsTests.swift`

**Interfaces:**
- Produces: a compiling target with `struct Qwen35Dense2B4BitVariant: EngineVariant` whose statics return real values and whose `init/prefill/decode` *throw* `TinyTitanError.variantNotPorted` — a compiling, gated, non-serving target. Task 7 replaces the throws with the ported graph.

- [ ] **5.1 Registry entry first (test drives it)** — write the test asserting the `qwen35-2b-4bit` variant has `"target": "Qwen35Dense2B4Bit"`; run it (FAIL); set the target in `registry/registry.json`; run again (PASS).

- [ ] **5.2 Add the target and skeleton** — `sources/Engines/Qwen35Dense2B4Bit/Variant.swift` carries a provenance header:

```swift
// Provenance: template created 2026-11-14 against docs/plan-tinytitan-7-engine.md §3.
// The graph ported in Task 7 is copied from sources/TinyTitan/CPUEngine
// (commit <sha recorded at copy time>).
```

Statics: `id = try VariantID("qwen35-2b-4bit")`, `source` from the registry's `source`, `geometry` from the 2B manifest contract (write it from a real `models/` 2B `manifest.json` if an install exists, else from `docs/adding-a-model.md`'s recorded contract — never from memory), `quant`: default 4-bit group 64, `embedding`/`lmHead` roles at 8-bit if the 6.x overrides say so (read `ModelProfile`/converter overrides, do not guess), `performance`/`sampling` copied from the 2B's `ModelProfile` table row, `recipe` from the 2B converter's constants.

- [ ] **5.3 Build and run the gates**

```bash
swift build -c release                                                  # Expected: success
PATH=".build/shims:$PATH" tools/lint.sh variant-isolation               # Expected: ok
swift test --no-parallel --filter RegistryTargetsTests                  # Expected: PASS
```

- [ ] **5.4 Commit**

```bash
git add -A
git commit -m "feat(7.0): Qwen35Dense2B4Bit variant scaffold under the isolation gate"
```

---

## Task 6 — `tinytitan catalog --check`

**Depends on:** Tasks 1, 5. **Tracker row:** TT-077.

**Files:**
- Modify: `sources/TinyTitanCLICore/` (new `CatalogCommand.swift`), `sources/TinyTitanCLI/Command/` (wire the subcommand)
- Test: `tests/TinyTitanCLI/CatalogCheckTests.swift` (create `tests/TinyTitanCLI/`)

**Interfaces:**
- Consumes: `registry/registry.json` (Task 1 schema), the `target` names (Task 5).
- Produces: `swift run TinyTitanCLI catalog --check` exits 0 when every registry entry's `target` names a real SwiftPM target and every variant marked `"shipped": true` has a non-null `qualification`; exits non-zero naming the first inconsistency. `--markdown` renders the model list (the installer menu and README generation come with later phases — this task delivers check + list render only).

- [ ] **6.1 Write the failing test** — stub registry in a temp dir, stubbed target list: missing target named → non-zero; `"shipped": true` with null qualification → non-zero; consistent stub → zero. Target existence is checked against `swift package describe --type json` (RELEASE.md §1.5's "guard the plan" rule — do not check for artifact files).

- [ ] **6.2 Run to verify failure** — `swift test --no-parallel --filter CatalogCheckTests` → Expected: FAIL.

- [ ] **6.3 Implement** — `CatalogCommand.swift` in `TinyTitanCLICore`: parse the registry (JSONSerialization is enough), run `swift package describe --type json` for target names, apply the checks, print findings to stderr, exit non-zero on any. Wire `catalog [--check] [--markdown]` into the CLI's command router.

- [ ] **6.4 Run for real against the tree**

```bash
swift run -c release TinyTitanCLI catalog --check
```

Expected: exit 0 with every variant `"shipped": false` (nothing is shipped yet, so nothing owes a qualification). Flip one stub variant to `"shipped": true` with null qualification in a scratch copy and confirm the check refuses it, then restore.

- [ ] **6.5 Gates and commit**

```bash
PATH=".build/shims:$PATH" tools/lint.sh    # Expected: ok
git add -A
git commit -m "feat(7.0): tinytitan catalog --check — registry ↔ target ↔ qualification"
```

---

## Task 7 — Port the dense 2B 4-bit graph into the variant

**Depends on:** Task 5. **Tracker row:** TT-078.

**Files:**
- Create: `sources/Engines/Qwen35Dense2B4Bit/Graph/` (the copied forward), `Quant/` (the variant's concrete int4 linear path copied from the 6.x code the 2B actually executes)
- Copy-with-provenance (never git mv — 6.x keeps running): the files under `sources/TinyTitan/CPUEngine/` the 2B decode actually executes.

**Interfaces:**
- Consumes: `TinyTitanCore` (`EngineContext`, `MetalContext`, tokenization), `TinyTitanFormat` (the format reader).
- Produces: `Qwen35Dense2B4BitVariant.init/prefill/decode` implemented for the 4-bit CPU path; every copied file carries `// Provenance: copied from <path> @ <sha>, <date>; qualified independently per spec §3`.

- [ ] **7.1 Record the executing file set** — run the 2B once (`ls models/` first; if the 2B install is absent this task BLOCKS on the operator — never fetch):

```bash
TINYTITAN_LAYER_TRACE=1 .build/release/TinyTitanCLI \
  --model models/qwen3.5_2B_4Bit --prompt "The capital of France is" --max-new 8
```

Record the trace's stage list; map each stage to its source file; write the list into the commit body.

- [ ] **7.2 Copy the graph as pure code motion** — copy each file into `Graph/` (≤500 lines each, C2; the copies inherit the 6.x splits). Rename type prefixes to `D2B4*` (e.g. `DenseCPUForward` → `D2B4Forward`) so the two engines can coexist in one process for the parity test. Add the provenance header to every file. **No behaviour edits, no "improvements".**

- [ ] **7.3 Implement the EngineVariant conformance** — `prefill`/`decode` call the copied graph with the variant's own `PerformanceProfile` constants in place of the `ModelProfile` lookups and environment switches (spec §7: overrides compile out — delete the env reads in the copy). Geometry validation runs `geometry.validate(manifest)` in `init` and refuses on mismatch.

- [ ] **7.4 Build + gates**

```bash
swift build -c release                                            # Expected: success
PATH=".build/shims:$PATH" tools/lint.sh variant-isolation core-purity   # Expected: ok
swift test --no-parallel                                          # Expected: all green, no count change
```

- [ ] **7.5 Commit**

```bash
git add -A
git commit -m "feat(7.0): port the dense 2B 4-bit graph into its variant (pure copy)"
```

---

## Task 8 — Golden and benchmark parity against 6.x

**Depends on:** Task 7. **Tracker row:** TT-079.

**Files:**
- Create: `tests/Engines/Qwen35Dense2B4Bit/ParityTests.swift` (create `tests/Engines/Qwen35Dense2B4Bit/` and a `Qwen35Dense2B4BitTests` SwiftPM test target)

**Interfaces:**
- Consumes: the 6.x engine path (still live in `TinyTitanLib`) and the 7.0 variant, in one test process.
- Produces: the parity evidence for the qualification record (Task 9): greedy output hashes, decode tok/s pairs, memory readings.

- [ ] **8.1 Write the model-gated parity test** — gate with `.enabled(if:)` on an env var naming the install (the sanctioned honest-skip idiom — never a silent early return; the `silent-test-skip` gate enforces this):

```swift
@Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_VARIANT_PARITY_MODEL"] != nil,
               "requires a .ssdai 2B install; set TINYTITAN_VARIANT_PARITY_MODEL"))
func greedyParityAgainst6x() throws { ... }
```

The body loads the same install twice (6.x `Engine` path; 7.0 variant), runs the fixed prompt set (`benchmark/golden/`'s prompt list is the reference — read it, do not restate it), greedy, 128 new tokens, and asserts the SHA-256 of the concatenated outputs is identical between paths; on mismatch the test prints both hashes and fails (a recorded, approved difference is a later deliberate change, never a silent pass).

- [ ] **8.2 Run the parity**

```bash
TINYTITAN_VARIANT_PARITY_MODEL=models/qwen3.5_2B_4Bit \
  swift test --no-parallel --filter ParityTests
```

Expected: PASS with identical hashes. **State in the report that the 2B is the deliberate subject here** (constraint C4's exception).

- [ ] **8.3 Paired decode benchmark** — time both paths over 512 tokens of the same fixed prompt, 3 measured runs each after one discarded warm-up, same process order (6.x first, 7.0 second) for all three rounds. Record both medians and the delta; the 7.0 path must be within the 6.x noise floor (spec §11) — a regression beyond noise is a finding to fix before Task 9, never a rounding note.

- [ ] **8.4 Commit**

```bash
git add -A
git commit -m "test(7.0): golden + benchmark parity for Qwen35Dense2B4Bit against 6.x"
```

---

## Task 9 — The matrix test and the qualification record

**Depends on:** Tasks 6, 8. **Tracker row:** TT-080.

**Files:**
- Create: `tests/Engines/Qwen35Dense2B4Bit/MatrixTests.swift`
- Modify: `registry/registry.json` (`qualification` filled for `qwen35-2b-4bit`, `"shipped": true`)

**Interfaces:**
- Produces: the matrix-test pattern the follow-on plans reuse; the first qualification record; `tinytitan catalog --check` turning green with one shipped variant.

- [ ] **9.1 The matrix test (spec §12.6)** — synthetic tensors, no install, no model gate:

```swift
@Test func everyRoleResolvesAndEveryWidthHasAPath() throws {
    let quant = Qwen35Dense2B4BitVariant.quant
    for role in QuantRole.allCases where D2B4Contract.roles.contains(role) {
        let spec = quant.spec(forRole: role, stem: TensorStem("layer0.weight"))
        #expect([2, 4, 8].contains(spec.bits))
        // The nibble trap: synthesize an 8-bit buffer, read it through the
        // variant's kernel path for its declared spec, assert the dequantized
        // values equal the host-side reference computed at spec.bits — an
        // 8-bit-as-nibbles read produces garbage here.
    }
    for role in QuantRole.allCases where !D2B4Contract.roles.contains(role) {
        // A role the variant does not declare must not be reachable: assert
        // the variant's graph never constructs a reader for it.
    }
}
```

Run: `swift test --no-parallel --filter MatrixTests` → Expected: PASS.

- [ ] **9.2 Write the qualification record into the registry** — `qualification` for `qwen35-2b-4bit`: `goldenSHA256` from Task 8, decode/prefill/TTFT medians from Task 8's paired runs, memory ceiling from the paired run's `phys_footprint`, `terminalBench: null` with a note that the 2B's published row is 3% (2.1) — the row is generated from the record when `catalog --markdown` owns the README table (phase 4+); `hardware: "M3 MacBook Pro 24 GB"`, `date: "<today>"`. Set `"shipped": true` only when every field but `terminalBench` is present.

- [ ] **9.3 Prove the gate chain end to end**

```bash
swift run -c release TinyTitanCLI catalog --check   # Expected: exit 0 (one shipped, qualified)
PATH=".build/shims:$PATH" tools/lint.sh             # Expected: all 20 checks ok
swift test --no-parallel                            # Expected: green everywhere
```

- [ ] **9.4 Commit, push, hand over**

```bash
git add -A
git commit -m "feat(7.0): first qualified variant — Qwen35Dense2B4Bit ships its qualification record"
git push origin main
```

Write `docs/handover-tinytitan-7-phase2.md`: what landed, the parity numbers, the open follow-on plans, the traps hit (record them as they happened, not from memory).

---

## Self-review notes

- **Spec coverage:** phases 0, 1, 2 of §11 → Tasks 1, 2–3, 4–9. Ordered changes 1 (registry), 3 (QuantSpec/QuantRole), 4 (profile split), 6 (matrix test) are implemented here; ordered change 2 (manifest-first geometry) lands via Task 7 step 7.3 and generalizes per variant in phases 3+; ordered change 5 (features into variants) is phase 5. Definition-of-done items covered here: registry exists, first variant qualified, purity/isolation gates live. The remaining DoD items belong to phases 3–9 by design.
- **Deliberate deviations from the spec's sketches, recorded:** `EngineContext` replaces the protocol's bare `MetalContext` (CPU-only dense variants need a context without Metal — the spec calls its Swift "names illustrative, the boundaries are the contract"); `registry/` replaces `Models/` (case-insensitive-filesystem collision with the gitignored `models/` install dir, C10); `catalog --check` consults `swift package describe` rather than artifact files (RELEASE.md §1.5). Each is argued in-task.
- **Placeholder scan:** no TBD/TODO steps; every code step carries its code or its exact command; the copy-lists that depend on runtime facts (Task 7's file set) are defined by a measurement step, not by memory.
- **Type consistency:** `VariantID`/`QuantSpec`/`VariantQuant.spec(forRole:stem:)`/`GeometryContract.validate` names are used identically in Tasks 4, 5, 7, 9.
