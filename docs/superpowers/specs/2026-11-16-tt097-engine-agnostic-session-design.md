# TT-097 design: an engine-agnostic session, so the server runs the 7.0 variant engine

**Status: approved design (owner 2026-11-16), pre-spec-review.** No code is written
from this document until the owner approves the spec and an implementation plan
exists for it.

## 1. Goal and scope

Make the **server process** generate through the 7.0 `EngineVariant` engines
instead of the 6.x `TinyTitan` runtime, for every currently-served model scenario,
using the existing multi-backend seam. Scope confirmed by the owner:

- **Ornith 1.5 35B-A3B 4-bit** -> the variant **GPU** engine (`Ornith35B4Bit`).
- **Dense Qwen 3.5 2B/4B/9B at 4 and 8 bits** -> their variant **CPU** engines
  (`Qwen35Dense*Bit`, the existing `CPUEngine`).
- **Dense Qwen 3.5 2B/4B/9B on the GPU** -> new variant **GPU** engines (added by
  owner request, "cover all model scenarios").
- Everything else (Qwen3.8, K2, Laguna, ...) **stays on 6.x** - out of scope.

Success = the server, for each in-scope model and each of its engines, produces
golden-byte-identical, streamed output through the variant engine, with the full
feature surface (tools, sampling knobs, prompt cache, reasoning split, queued
long-prompt prefill) intact and no speed/quality regression vs the 6.x server path.

## 2. What is true today (measured, this session)

- `ServerInferenceBackend` is the seam the whole server talks to, and it already has
  **two** implementations - `ServerModelSession` (GPU, in `TinyTitanLib`) and
  `CPUModelBackend` (dense CPU, in `TinyTitanServerCore`) - chosen by `ModelRouter`.
  Both `import TinyTitan` (6.x).
- The GPU session's only *concrete* 6.x engine types are `Model` (6 members:
  `config`, `modelID`, `routedExpertWeightBits`, `sourceSnapshotHash`,
  `saveExpertUsageProfile`, `routedExpertCacheSlotCount`) and `RealForwardRunner`
  (46 members, **35 of them diagnostics**: `runner.total*` / `reset` / `Statistics`).
  `GFTokenizer`, `MetalContext` and `ArchConfig` are **already in `TinyTitanCore`**.
- The CPU backend's concrete 6.x types are `CPUQwen35`, `CPUSampler`,
  `CPUModelFamily`.
- `ValidatedChatRequest` (`OpenAIRequestValidation.swift`) depends on 6.x only via
  `GFTokenizer.Message`/`GFTokenizer.FunctionDefinition`, which are now Core.
- The 6.x and variant `RealForwardRunner` are both `public` and already conform to
  the *same-named* protocols `ChunkedPrefillRunner` + `ContextWindowReporting`, but
  each defined **locally** in its own module's `LogitProducer.swift` - so they are
  different nominal protocols today.
- File volumes: a dense CPU variant is ~20 Swift files; the GPU engine is 273.

## 3. The core idea

Promote a **single shared engine protocol** into `TinyTitanCore` that both the 6.x
engine and every variant engine conform to, then type the two session
implementations against that protocol and inject the concrete engine from a
**factory**. The session code stops being bound to a specific engine module, so it
runs whichever engine the router hands it. This satisfies AGENTS rule 2 (one
generation path): the session is *reused*, not copied per engine.

## 4. Architectural decisions

### D1 - Shared engine contracts live in `TinyTitanCore`

New, model-free protocols in `TinyTitanCore/Engine/`:

- `EngineModel` - the 6 `Model` members above.
- `EngineRunner` - the ~11 behavioral `RealForwardRunner` members the session uses
  (prefill/decode entry points, raw completion, continuation position, inference
  state save/restore, supports-restore flag, expert-streaming statistics), **plus a
  single `var statistics: EngineStatistics` and `func resetStatistics()`** that
  collapse the 35 diagnostic counters.
- `EngineStatistics` - a struct carrying the 35 diagnostic counters, built by
  reading the existing per-counter properties (no recomputation).
- `CPUEngineModel` / `CPUEngineRunner` - the `CPUQwen35`/`CPUSampler` members
  `CPUModelBackend` uses.

Conformance is **added to existing types**:

- 6.x `Model`/`RealForwardRunner` declare `: EngineModel`/`EngineRunner`; the members
  already exist, so conformance is a declaration plus a `statistics`/`resetStatistics`
  implementation returning the existing counters.
- Each variant's identical `Model`/`RealForwardRunner` declare the same.
- `CPUQwen35` (6.x) and each dense variant's `CPUQwen35` declare the CPU protocols.

`ChunkedPrefillRunner`/`ContextWindowReporting` are folded into (or referenced by)
`EngineRunner` so there is **one** nominal definition in Core, not two copies.

Why Core, not a new module: Core is the only module a variant may import
(variant-isolation), and it already carries `GFTokenizer`/`MetalContext`/`ArchConfig`.
These protocols are model-free (no family switch, no names), so core-purity holds.

### D2 - The sessions are protocol-typed and the engine is injected

- `ServerModelSession` stores `let model: any EngineModel` and
  `let runner: any EngineRunner` instead of the concrete types; the construction
  (`Model.load(...)`, `RealForwardRunner(...)`) leaves `TinyTitanLib` and moves into
  an **`EngineFactory`** value the caller injects (`makeSession` receives it).
  `TinyTitanLib` therefore stops importing 6.x for inference - the same move that
  finally decouples the embeddable facade from 6.x.
- `CPUModelBackend` is typed against `CPUEngineModel`/`CPUEngineRunner` with the same
  injection idea.
- `ServerInferenceBackend` / `ServerCompletion` / `ServerInferenceEvent` /
  `ValidatedChatRequest` stay where they are; after D1 a variant no longer needs to
  import them, because the variant supplies an `EngineRunner`, not a
  `ServerInferenceBackend`. The **server** wraps the engine in the (unchanged)
  protocol-typed session.

### D3 - The server chooses the engine per model via the factory

`TinyTitanServerCore` (which may import both 6.x and the variants) provides a 6.x
factory and a variant factory, and `ModelSessionPlan`/`ModelRouter` select by catalog
identity:

- `ornith-1.5-35b-4bit` -> variant GPU factory.
- `qwen35-{2b,4b,9b}[-8bit]@cpu` -> variant CPU factory.
- `qwen35-{2b,4b,9b}[-8bit]@gpu` (and the bare id, whose dense default is GPU) ->
  variant dense-GPU factory (D4).
- everything else -> 6.x factory.

### D4 - Dense GPU engines (owner-requested; structure decision)

The dense family is served on the GPU by the **config-driven** runner with
`numExperts == 0` - exactly the mechanism `docs/plan-dense-gpu-engine.md` shipped and
verified (the sizes/widths differ only by manifest geometry; 6.x's one
`RealForwardRunner` serves all of them).

Two ways to express that as a 7.0 variant:

- **Shared config-driven dense-GPU engine (recommended).** One new GPU-engine variant
  module for the dense family (`sources/Engines/DenseGPU/`, ~273 files, a
  spec-sanctioned copy of the config-driven runner) conforming to `EngineRunner`, with
  geometry read from the manifest. It has **no runtime family switch** (geometry comes
  from the install's manifest, as the dense CPU engine already does), so it is one
  variant serving the dense-GPU family rather than a union graph branching on model
  name. This adds ~273 files total, not 6x273.
- **Per-pair dense-GPU variants.** `Qwen35Dense{2,4,9}B{4,8}BitGPU` - six self-contained
  targets, ~1,640 near-identical files. Maximally literal to "one variant per
  (model, quant, engine)" but ~6x the files for no behavioral gain, since the code is
  config-driven.

**D4 ruling (pending owner confirm at spec review):** the **shared config-driven
dense-GPU engine**, matching the proven `plan-dense-gpu-engine.md` design and avoiding
~1,370 duplicated files. The owner may choose the per-pair alternative at spec review if
strict per-pair GPU targets are required.

Dense CPU and dense GPU engines are **separate targets by construction** (`DenseGPU`
does not co-locate with the existing `CPUEngine`s), which sidesteps the 9 colliding type
names (`GenerationDefaults`, `ModelFamily`, `Int8AffineGEMV`,
`ModelReasoningControl(+Error)`, `Quantization`, `ReasoningLevel(+Error)`,
`RequestReasoning`) that appear when CPU and GPU engine code share one module.

## 5. Data flow (steady state)

```
HTTP -> ValidatedChatRequest -> ModelRouter
  -> ModelSessionPlan.makeSession(engineFactory)     // factory chosen by identity
    -> ServerModelSession(model: any EngineModel, runner: any EngineRunner)
       // or CPUModelBackend(engine: any CPUEngineRunner)
    -> generate() -> streamed ServerInferenceEvent / ServerCompletion
```

The session never names an engine module; the factory does, and the factory is imported
only where the variants are importable (the server target).

## 6. Phasing (each phase is a committed, gated green state)

Every phase ends with: warning-free release build, the full serial suite
(`swift test --no-parallel`), **both** golden baselines byte-identical, the dense
`TINYTITAN_DENSE_GPU_EQUIV` oracle where relevant, and the 20 `tools/lint.sh` gates
green. Phases 1 and 2 run **on 6.x under the hood** (the factory still hands out the 6.x
engine), so their goldens are unchanged by construction - they prove the refactor is
behavior-preserving before any model switches engines.

1. **Core contracts + conformance.** `EngineModel`/`EngineRunner`/`EngineStatistics`
   and the CPU protocols in Core; 6.x + all variant engine types conform. The
   diagnostics are pinned by a new test before the `statistics` aggregate is added.
   Runtime unchanged; goldens byte-identical; parity vs 6.x holds.
2. **Protocol-typed sessions + factory injection.** `ServerModelSession` and
   `CPUModelBackend` typed on the protocols; engine construction injected; Lib stops
   importing 6.x for inference. Still 6.x factory; goldens byte-identical; suite +
   parity green.
3. **Variant factories + routing.** The shared dense-GPU engine is built (D4) and all
   three factory kinds exist; the router selects per identity. Ornith and dense CPU
   first (engines that already exist), then dense GPU. Each newly-routed model is
   model-gated as it flips.
4. **Acceptance bar + record.** Per switched model/scenario: golden, parity vs 6.x,
   streaming, tool-calls, sampling-knob override, prompt cache, reasoning split,
   OpenCode queued long-prompt prefill (the deadlock fixed in TT-095), and a paired
   benchmark vs the 6.x server path (adopt only if not slower and not
   quality-degrading, per AGENTS). Then update TT-082/087/097, the registry
   qualification rows, README/wiki, and 6.x retirement (TT-087) becomes reachable.

## 7. Verification / acceptance (model-gated; preconditions first)

Before any model run: macOS 26+, Xcode 27 / Swift 6.4, disk, acceptable
`memory_pressure -Q`, a completed `.ssdai` install with a valid receipt, no competing
model process. Run tests serially; one CLI/model test at a time.

- **Numerics:** greedy golden text identity per model/scenario; the checked prompt set
  with zero top-1 divergence and logits within the measured noise floor; the dense
  GPU-vs-CPU oracle. A parity failure is a finding, not something to re-baseline.
- **Behavior surface:** one end-to-end run per feature (stream deltas, tool calls,
  explicit temperature/top-p/top-k overriding family defaults, prompt-cache reuse
  across turns, reasoning split, and the queued long-prompt prefill that previously
  deadlocked).
- **Performance:** a paired, interleaved benchmark vs the 6.x server path per model, on
  a quiet machine, per the paired-benchmarking method. Adopt only if not slower and not
  quality-degrading.

## 8. Risks and mitigations

- **R1 - the diagnostics aggregate changes a reported count.** If any `runner.total*`
  value is currently derived rather than stored, `EngineStatistics` could change what
  the observability line prints. Mitigation: Phase 1 pins the 35 counters with a test
  before adding the aggregate; the aggregate only reads existing properties. This is
  numerics-adjacent, so it is gated even though it does not affect tokens/logits.
- **R2 - protocol extraction forces `any` dynamic dispatch on the hot path.** A
  generic-`some` or specialized path may be needed to keep decode speed; measured in
  Phase 2's paired benchmark, and if the protocol witness costs decode throughput, the
  session is specialized per engine via a generic parameter (same source, two
  instantiations) instead of an existential.
- **R3 - `TinyTitanLib` importing a variant for the facade.** It does not: Lib stays
  protocol-typed and receives the engine through the injected factory; only the server
  imports the variant modules.
- **R4 - dense-GPU shared engine bending the per-(model,quant) variant rule.** Ruled in
  D4: config-driven, no family switch, manifest-supplied geometry, so it is one variant
  for the family, mirroring `plan-dense-gpu-engine.md`. Owner may override.
- **R5 - half-migrating breaks main.** Mitigated by the phasing: every phase is a green,
  gated commit and 6.x stays live throughout; nothing is deleted.

## 9. Out of scope

- Deleting 6.x (TT-087) - only becomes reachable after Phase 4 passes.
- Qwen3.8 / K2 / Laguna / Ornith 8-bit / other models on variants.
- Any GUI/desktop surface (none exists, none added).
- Numerics or default changes: none; output is byte-identical or the change is rejected.
