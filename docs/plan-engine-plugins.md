# Engine plugins: TinyTitan Plugin Protocol v1

Status: **design, 2026-11-24; TPP/1.0 implemented (TT-130)**: the API target,
the registry, the template plugin and the Inkling 2 Q4 plugin. The
specification is `docs/plugin-api-v1.md`. Tracker: TT-130 (protocol and
host), TT-131 (conformance kit), TT-132 (migrate the shipped engines), TT-133
(out-of-tree plugins).

## Goal

A new LLM joins TinyTitan by shipping **one engine plugin** that implements the
TinyTitan Plugin Protocol v1 (TPP/1). Nothing else in the tree changes: no edit
to the library, CLI, server, catalog, tokenizer or session. Once the plugin
passes the conformance kit, every TinyTitan feature works for that model in
every front end, because the front ends only ever talk to the protocol:

- `TinyTitanLib` (`Engine`, `Session`)
- the CLI
- the loopback server: OpenAI chat, Responses, Anthropic `/v1/messages`, and
  `--models-dir` catalog switching
- the memory layer
- `calibrate`, the golden baselines and `release.sh`

The rule this enforces is the one AGENTS.md already states for the API — *every
served model supports the whole API but vision, and a gap is fixed, not
documented* — made structural: a plugin either provides a capability through
the protocol or declares it refused, with a reason the host shows to the user.
Nothing is accepted and silently ignored.

Non-goals for v1:
- **No dynamic loading (`dlopen`) of untrusted code.** See Packaging.
- **No change to any engine's numerics.** The migration is code motion behind
  the protocol.
- **No GUI.**

## The owner's rules (2026-11-24)

- **A plugin is standalone.** Each LLM plugin is a library target compiled into
  the build. Otherwise it is independent of the rest: it imports only the plugin
  API, Core and system frameworks, and all tuning and customization for a model
  happens in its plugin's code only.
- **This holds for every supported model, 6.x included.** The template plugin,
  with the API spec, is the golden base; the other models come in as plugins
  one by one over time.
- **Phase 1:** the plugin system is clear and works with every model on disk:
  Inkling 2 Q4 and NVFP4, Ornith 1.5 4-bit, and the six dense Qwen 3.5
  installs.
- **Phase 2:** import the remaining 6.x code into 7.x, so the core is fully
  ported. What remains after that is plugins for the models not on disk.

**As built in 1.0, the plugin's entry points are static** (`static func probe`,
`static func engineFactory(host:)`), and the engine is the existing Core
`EngineFactory`. The `ChatDialect`, `LoadPlanV1` and `LoadedEngine` shapes below
are the 1.1 direction.

## Why now: the seams the Inkling integration hit

TT-122 put Inkling 2 Q4 on the shared session. Every edit outside the variant
target is a place where the host still knows about specific models, and that is
what a plugin protocol has to remove:

| Seam | Where today | What Inkling needed |
|---|---|---|
| Engine selection | `EngineFactories.gpuSelection` / `gpuFactory(directoryURL:)` — a hand-written switch on `ModelFamily` and model-id prefixes | a new `isInkling2Q4` branch in two functions |
| Family identity | the 6.x `ModelFamily` enum, read by the catalog (`ModelCatalog.Kind.gpu(ModelFamily)`), reasoning fallback, generation defaults (9 files outside the runtime) | a separate reasoning path in `singleModelReasoning`; not in the catalog at all (TT-126) |
| Front-end branches | `TinyTitanServer/Command/main.swift` and the CLI main: 18 model-specific branches (K2 probe, Inkling routes, GGUF, CPU) | its own branch until TT-122 removed it |
| Chat format | `TokenizerDialect` enum in Core with 45 `dialect ==`/`switch` sites, copied into each variant's decoder | a new enum case, edits to the renderer, the decoder (and its two copies in the Ornith and dense variants), the JSON table |
| Prompt cache | ChatML continuation bridges and checkpoint markers keyed on ChatML's message-start token in `TinyTitanLib` | a dialect guard, then `PromptMessageMarkers` |
| Tokenizer source | `GFTokenizer.tokenizerFolder` assumes `tokenizer/` in the install | a `tokenizerFolder` factory hook |
| Parameters | each server flag reaches each engine (or not) ad hoc | `--kv-bits` silently ignored until TT-123; `--ram-budget` ignored on the old route |

7.0 already has the right *runtime* contracts: `EngineFactory`, `EngineModel`,
`EngineRunner`, `LogitProducer`, `RawCompletionRequest` (Core
`Engine/`). TPP/1 keeps them and adds the four missing pieces: **identity and
probing**, **chat dialect**, **parameter declaration** and **registration**.

## The protocol

All TPP/1 types live in a new target, `TinyTitanPluginAPI`, which depends only
on `TinyTitanCore`. It is `public` and versioned: it is the promise to a plugin
author, the way `TinyTitanLib` is the promise to an embedder. A plugin target
imports `TinyTitanPluginAPI` and Core only (the existing variant-isolation gate,
extended to plugins).

### 1. The plugin

```swift
public protocol EnginePlugin: Sendable {
    /// Stable, reverse-DNS id: "ai.tinytitan.inkling2-gguf-q4".
    static var pluginID: String { get }
    /// The protocol this plugin was built against (1 for TPP/1).
    static var protocolVersion: ProtocolVersion { get }
    /// What the plugin serves, for the catalog and `/v1/models`.
    static var descriptor: PluginDescriptor { get }

    /// Whether this plugin serves the install at `directory`, read from the
    /// manifest alone (bounded read, no weights, no Metal). Several plugins may
    /// match; the host takes the highest `MatchStrength`, and a tie is a
    /// load-time error naming both plugins, never a silent pick.
    static func probe(_ install: InstallProbe) -> MatchStrength?

    init(install: URL, host: HostServices) throws

    var metadata: ModelMetadata { get }          // id, family label, bits, sampling, reasoning
    var dialect: any ChatDialect { get }         // §3
    var parameters: ParameterSupport { get }     // §4
    func plan(_ request: LoadRequest) throws -> LoadPlanV1      // §5
    func load(_ plan: LoadPlanV1, context: MetalContext) throws -> LoadedEngine  // §6
}
```

`InstallProbe` gives the plugin the parsed `manifest.json` (family, model id,
quant, `variantGeometry`) and the receipt state; a plugin never walks the
directory itself. `HostServices` is the one door back into the host:
- logging, through `ServerLog.diagnostic`, which keeps stdout clean;
- the shared Metal context;
- the user cache directory;
- the device's working set.

### 2. Identity

`PluginDescriptor` and `ModelMetadata` replace the host's switches on
`ModelFamily`:
- the family is an opaque string the plugin owns;
- `ReasoningControl` (already in Core) says which thinking and effort levels
  exist;
- `SamplingProfile` (already in Core) gives the model card's defaults;
- `capabilities` lists what the engine implements (§4).

The catalog, `/v1/models`, the reasoning fallback, the `--reasoning` mapping and
the golden-target list read these instead of `ModelFamily`.

### 3. Chat dialect

```swift
public protocol ChatDialect: Sendable {
    var tokenizer: TokenizerSource { get }       // folder, or a derivation (GGUF header)
    func render(_ request: RenderRequest, tokenizer: GFTokenizer) throws -> RenderedPrompt
    func makeParser(for prompt: RenderedPrompt, offeredTools: Set<String>,
                    tokenizer: GFTokenizer) -> any AssistantStreamParser
    func stopTokens(_ tokenizer: GFTokenizer) -> StopTokens   // turn end, document end, foreign turns
    var messageMarkers: PromptMessageMarkers { get }          // interior checkpoints
    func continuationBridge(...) -> [Int32]?                  // optional: multi-turn cache splice
}
```

`RenderRequest` carries the messages, tools, the resolved reasoning
(thinking and effort) and `structuredAnswer`. `RenderedPrompt` returns the ids
and the decoder's starting state (in text, in thought, or waiting for the model
to open a turn). `AssistantStreamParser` is the shape `StructuredAssistantDecoder` and
`InklingAssistantParser` already share: `consume(token)`, `tail()`, `finish()`
→ `[StructuredAssistantEvent]`.

Core ships **reference dialects**, not an enum: `ChatMLDialect` (the Qwen
families), `K2Dialect`, `InklingDialect`. A plugin picks one or brings its own;
adding a dialect touches no host file. The `TokenizerDialect` enum and its 45
switch sites go away, and so do the copies of the decoder in the variant
targets, because the decoder is the dialect's.

### 4. Parameters and capabilities: declared, never ignored

```swift
public struct ParameterSupport: Sendable {
    public var kvBits: Support<KVCachePrecision>          // e.g. .only([.fp16], reason: "…TT-123")
    public var ropeScaling: Support<RuntimeRoPEScalingMode>
    public var hostKVWindow: Support<Int>
    public var concurrentSequences: Support<Int>           // max width
    public var mtp: Support<URL>
    public var expertStreaming: Bool                        // honours --ram-budget / slots
    // … one field per user-facing parameter
}

public enum Support<Value> {
    case supported
    case only([Value], reason: String)    // other values refused by name
    case refused(reason: String)
}
```

The host checks every parameter the user set (CLI flag, server flag,
`EngineConfiguration` field) against this table **before loading**.
- **A supported value** is passed to the plugin.
- **An explicit unsupported value** is refused with the plugin's reason, as
  `--kv-bits 8` on Inkling is today.
- **A default value the plugin cannot honour** is replaced by the plugin's own
  default and logged once, as Inkling's 16-bit notice does. The host performs
  that substitution, so no plugin can forget it.

This table is also what turns "all features available" from a hope into a
checked fact: the conformance kit (below) exercises every `supported` entry.

### 5. Load planning

`LoadRequest` carries what the user asked for:
- window;
- `--ram-budget`, the whole-process target;
- explicit slots;
- K/V precision;
- sequences;
- prefill chunk;
- integrity policy.

`LoadPlanV1` is the plugin's answer:
- the granted window, which may be smaller than asked;
- slots;
- planned bytes, broken down into weights, cache, KV and reserve;
- the per-sequence bytes;
- notices.

The host prints the plan, refuses one that exceeds the budget, sizes the batch
width from the per-sequence bytes, and uses `min(asked, granted)` as the
session window (the rule TT-122 added).

### 6. The loaded engine

`LoadedEngine` bundles what `EngineFactory` closures return today:
- an `EngineModel`;
- an `EngineRunner`, which brings prefill, decode, reset, state capture and
  restore, and statistics;
- the completion loop (`run(RawCompletionRequest) -> RawDecodeResult`);
- optionally an `EngineMTPDecoder` and an `EngineExpertWarmup`.

These are the existing Core protocols, unchanged, so migrating a 7.0 variant
moves code and changes no numerics.

### 7. Feature inheritance

What a plugin gets by implementing TPP/1, and the hook each feature rests on:

| Feature | Provided by the host | Needs from the plugin |
|---|---|---|
| Plain and streamed chat, token by token | session, HTTP layer | runner + completion loop emitting `.token` progress |
| Tool calls (plain/streamed), tool-result follow-up | session + dialect parser | `ChatDialect.render` with tools, parser |
| Stop strings | session | — |
| `response_format` JSON schema | session grammar (`JSONConstraint`) | dialect: `structuredAnswer` render + document-end tokens; loop applies the mask |
| Thinking control, effort levels | session, `--reasoning` mapping | `ReasoningControl`, dialect render |
| Anthropic `/v1/messages`, Responses `/v1/responses` | HTTP mappers | — (they go through the chat path) |
| Prompt cache, interior checkpoints, shared system prefix, disk tier | session stores | runner `capture/restoreInferenceState`; `messageMarkers` |
| `--ram-budget`, `--expert-cache-slots`, `--max-context` | parameter gate, banner | `plan` |
| Concurrent sequences | coordinator, batch width | slot-aware runner, per-sequence bytes |
| MTP | session MTP path | `EngineMTPDecoder` |
| Library (`Engine`, `Session`), CLI | facade | — |
| Catalog, `--models-dir` switching, lazy load, idle unload | router | `probe`, `descriptor` |
| Memory layer, watchdogs, compaction | wrappers over the backend | — |
| Golden baselines, `calibrate`, release gates | tools read the registry | a golden capture per installed width |

## Registration and packaging

**v1 is static.** A plugin is a SwiftPM target under `sources/Plugins/<Name>/`
(today's `sources/Engines/<Name>/`) that exposes one type conforming to
`EnginePlugin`. Registration is one generated list:

```swift
// sources/TinyTitanPlugins/Registry.swift — generated from Package.swift
public let builtInPlugins: [any EnginePlugin.Type] = [
    Ornith35B4BitPlugin.self, DenseQwen35Plugin.self, Inkling2GGUFQ4Plugin.self, …
]
```

`PluginRegistry` (in `TinyTitanPluginAPI`) holds `builtInPlugins` plus anything
an embedder registers at startup (`PluginRegistry.register(MyPlugin.self)`),
which is how an out-of-tree plugin works: a package depends on TinyTitan and on
the plugin, and registers it. The CLI, server and library resolve an install
only through `PluginRegistry.resolve(installURL)`.

Why not `dlopen` in v1:
- **Codesigning and hardened runtime** refuse unsigned dylibs.
- **Swift has no stable ABI** across separately built packages of the same
  module graph.
- **Security:** a dynamically loaded plugin would run unreviewed code inside the
  server.
- **Isolation:** static linking keeps the stdout-clean and variant-isolation
  gates meaningful.

Dynamic loading can come in a later protocol version, if ever, over a C ABI.

## Versioning

- `ProtocolVersion(major: 1, minor: n)`.
- A minor version only **adds** requirements, each with a default
  implementation in a protocol extension. The default for a new parameter is
  `.refused("not implemented by <pluginID>")`, so an older plugin keeps
  building and the gap is reported to the user rather than hidden.
- A major version may remove or change requirements; the host refuses a plugin
  whose major differs, naming both versions.
- `TinyTitanPluginAPI` is in the `library-facade` gate's allowlist
  (`tools/library-facade-baseline.txt`) beside `TinyTitanLib` and `CPUEngine`
  (decided 2026-11-24): a new public plugin requirement is a deliberate,
  re-measured change, never inherited.

## Conformance kit (what "automatically all features" is checked by)

`tinytitan-plugin-check <install>` runs, in order:

1. **Model-free contract tests**, against the plugin's own fixtures:
   - the dialect: render/parse round trips, a tool call, a JSON document end,
     foreign-turn stop;
   - the parameter table: every `.only`/`.refused` produces a refusal naming
     the plugin;
   - the plan arithmetic.
2. **Model-gated tests:**
   - `LibraryContractTests` through the registry: two engines on one device,
     cancellation, `unload()`;
   - snapshot capture/restore bit-identity: logits after a restore equal
     logits after a full prefill;
   - greedy agreement with the plugin's declared oracle (llama.cpp, a reference
     implementation, or the previous route).
3. **The server surface:**
   - `tools/api-feature-check.py`, ten rows, including the streamed-chunks
     check;
   - a prompt-cache round trip (repeat and follow-up hits);
   - one launch per declared parameter value.
4. **A golden capture** for the installed widths. That is the qualification
   record; `Qualification` in Core already has the shape.

A plugin is **supported** only when all four pass; that replaces today's
per-model checklist in `docs/adding-a-model.md`. CI runs step 1 for every
plugin; steps 2–4 run where an install exists, and the release notes name
any that were not checked.

## Gates

- `core-purity` extends to `TinyTitanLib`, `TinyTitanServer`, `TinyTitanCLI` and
  `TinyTitanPluginAPI`: no model name and no `ModelFamily` in host code. An
  empty scan is a failure, as today.
- `variant-isolation` becomes `plugin-isolation`: a plugin imports Core,
  `TinyTitanPluginAPI` and system frameworks only.
- `plugin-registry`: every target under `sources/Plugins/` is in the
  generated registry, and nothing else is.

## Migration

The order is chosen so each step is shippable and bit-identical:

1. **TT-130: `TinyTitanPluginAPI` and `PluginRegistry`.**
   - Build them, plus an adapter that wraps an existing `EngineFactory` as an
     `EnginePlugin`, so all current engines are plugins on day one.
   - Route the server, CLI and library through `PluginRegistry.resolve`.
   - Golden baselines and api-feature-check must not move.
2. **TT-131: the conformance kit and the parameter gate.**
   - Move the per-flag checks into the gate.
   - Inkling's K/V refusal becomes data in its `ParameterSupport`.
3. **TT-132: dialects become `ChatDialect` values.**
   - Delete `TokenizerDialect` and the decoder copies.
   - Migrate engines one per commit: Inkling Q4, then Ornith, then dense Qwen,
     then the 6.x engine (as `LegacyPlugin`), then K2 and the CPU engine.
   - Then delete `gpuSelection`, the `main.swift` model branches and
     `ModelFamily` outside the 6.x runtime.
4. **The Inkling NVFP4 engine (TT-125) is the first engine written directly as
   a plugin.**
   - It is the test of the claim: it must reach every feature without one host
     edit.
5. **TT-133: out-of-tree plugins.**
   - Prove them with `examples/plugin/` beside `examples/embedded/`, built by a
     check script like `tools/embedded-dependency-check.sh`.

Each step is gated by the full lint set, the unit suites, the golden baselines
of every installed model, and api-feature-check 10/10 on the engines it touches.

## Risks and open questions

- **Performance seams.** The protocol is the shape of today's closures, so the
  hot path (runner, completion loop) stays inside the plugin and is not
  wrapped. The risk is an adapter that copies logits or adds an actor hop;
  the golden and paired benchmarks of step 1 catch it.
- **Batched decode across plugins.** `produceBatch` is per runner, so batching
  stays within one model; cross-model batching is out of scope.
- **The 6.x engine as one plugin.** It serves several families; it registers one
  plugin with a probe per family it still owns, until those families have
  their own plugins.
- **Who owns a manifest family.** Two plugins may claim `inkling2`, for the Q4
  and NVFP4 installs; `probe` keys on quant scheme as well, and a tie is an
  error by design.
- **Open: per-plugin defaults that differ from the server's** (the K/V
  precision is the first case). v1 lets the plugin's default win when the user
  did not set the flag. Decide whether `--help` should show per-plugin
  defaults.
