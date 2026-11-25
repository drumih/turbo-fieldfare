# TinyTitan Plugin Protocol v1 (TPP/1): specification

The contract between TinyTitan and one LLM engine. The design and its reasons
are in `docs/plan-engine-plugins.md`. This page is the spec a plugin is written
against, and `sources/Plugins/TemplatePlugin` is its worked example: the golden
base every plugin is copied from.

## 1. What a plugin is

- **A standalone library target.** It is compiled into the build and is a
  SwiftPM library product of its own, built and tested on its own.
- **Narrow dependencies.** It depends only on `TinyTitanPluginAPI`,
  `TinyTitanCore` and system frameworks (Foundation, Metal, Accelerate). It
  imports no other plugin and no host target: not `TinyTitanLib`, not the
  server, not the 6.x `TinyTitan` runtime.
- **All model-specific code is inside it.** That means recognising the install,
  chat format, load arithmetic, kernels, sampling, tuning and defaults.
  Changing how one model runs never touches the host or another plugin. Code a
  plugin needs from 6.x is copied in with a provenance line, as
  `Inkling2GGUFQ4/ModelIO/GGUFTokenizerSidecar.swift` is.
- **It never writes to stdout.** It logs through `HostServices.log`.
- **One type is the entry point.** It conforms to `EnginePlugin` and is listed
  in `EngineFactories.builtInPlugins`, or registered by an embedder with
  `PluginRegistry.shared.register(_:)`.

## 2. `EnginePlugin`

| Requirement | Contract |
|---|---|
| `pluginID` | Stable reverse-DNS id, never reused: `ai.tinytitan.<model>-<scheme>`. |
| `protocolVersion` | `ProtocolVersion.current` at build time. The host refuses another major. |
| `displayName` | Human name used in logs and refusals. |
| `probe(_:)` | From `InstallProbe` (directory, parsed `.ssdai` manifest, declared family) alone: no weights, no Metal, no directory walk. Return `.exact` for installs built for this plugin (family + scheme + model), `.family` for a generic family engine, `.fallback` only for the 6.x adapter, nil otherwise. A tie at the top is a resolution error naming both plugins. |
| `parameters(for:)` | The `ParameterSupport` table (§3). |
| `reasoningControl(for:)` | The model's thinking and effort levels (`ReasoningControl`). Only the 6.x adapter may return nil. |
| `engineFactory(host:)` | The engine as the shared session consumes it: a Core `EngineFactory` (§4). |

## 3. Parameters: declared, never ignored

Each user-facing parameter is `.supported`, `.only([values], reason:)` (the
first value is the engine's own default) or `.refused(reason:)`. The fields in
v1.0:

| Field | Parameter |
|---|---|
| `kvCachePrecision` | `--kv-bits`, `EngineConfiguration.cachePrecision` |
| `ropeScaling` | `--rope-scaling` |
| `hostKVWindow` | `--host-kv-window` |
| `concurrentSequences` | `--max-concurrent-sequences` |
| `multiTokenPrediction` | `--mtp-model` |
| `expertStreaming` | `--ram-budget`, `--expert-cache-slots` streaming routed experts |

The host enforces the table:
- **An explicit value the engine does not accept** is refused before loading,
  as `"<flag> <value> is not supported by <plugin>: <reason>"`.
- **A default the engine does not accept** is replaced by the engine's own and
  announced once in the log.

The reason must be a fact: a measurement, a missing structure, or a tracker row
for the work that would add the feature.

## 4. The engine (`EngineFactory`)

These are the 7.0 Core contracts, unchanged:

| Stage | Contract |
|---|---|
| `prepareLoad` | Plan from `SessionBuildOptions`. `maxContext` is the asked window; `expertCacheBudgetBytes` is `--ram-budget` (whole process); `expertCacheSlots` is explicit slots. Return an `EngineLoadPlan`; put the plugin's own plan in `engineConfiguration`. Log the plan through `host.log`. |
| `metadata`, `resolveArchitecture`, `preview` | Identity (model id, family label, bits, sampling profile, reasoning control) and geometry from the install, without loading weights. |
| `buildModel` / `buildModelWithSlots` | The `EngineModel`. |
| `buildRunner` | The `EngineRunner`, sized from the plan. Its `maxContext` is the granted window; the session uses `min(asked, granted)`. |
| `runRawCompletion` | The generation loop (§5). |
| `perSlotBytes` | Bytes per extra concurrent sequence; a value no budget meets keeps the width at 1. |
| `tokenizerFolder` | A folder with `tokenizer.json` and `chat_template.jinja` when the install does not keep `tokenizer/`; nil otherwise. |
| `mtpDecoder`, `makeWarmup` | Optional. |

**Runner obligations:**
- `prefillChunked` and `produce` write the next-token logits (FP16 rows of the
  model vocabulary) or return a greedy seed.
- `continuationPosition` is the number of tokens whose state is held.
- `captureInferenceState` and `restoreInferenceState` round-trip that state
  exactly. Logits after a restore must equal logits after a full prefill: this
  is what the prompt cache and interior checkpoints rest on.
- `supportsInferenceStateRestore` is false only if they are not implemented,
  which turns the prompt cache off for the model.

## 5. The generation loop

`runRawCompletion(RawCompletionRequest) -> RawDecodeResult`
(`TemplatePlugin/TemplateCompletion.swift` is the reference):

1. `.reset` clears the state; `.resume(n)` continues a restored prefix of `n`
   prompt tokens and prefills only the rest.
2. Prefill, stopping at each `checkpointPositions` entry and calling
   `onPrefillCheckpoint` there; report `.prefill(done:total:)`.
3. Decode, reporting `.token(index:id:delta:)` as each token is chosen. This is
   the streaming contract: no route may buffer a multi-token answer (TT-128).
   Report `.tail` once at the end.
4. Apply `configuration.constraint` (the JSON grammar) to the logits before
   choosing; sample with the plugin's own sampler; greedy when the
   configuration is pure greedy.
5. Stop on `tokenizer.stopTokenIDs` / `configuration.extraStopTokens`
   (`.endOfTurn` for the turn's own end), stop strings (`.stopString`), the
   session's `shouldStop` (`.external`) or the budget (`.maxTokens`).
6. Return `kvBackedTokenIDs` (every token fed), `kvPosition` (their count) and
   the one uncommitted boundary token (chosen, never fed).

## 6. Chat format

A plugin declares its chat format via `EnginePlugin.chatDialect(for:)` (TPP/1.1,
TT-132), returning one of Core's reference `ChatDialect` values
(`ChatMLDialect`, `K2Dialect`, `InklingDialect`) or a plugin-owned conformer.
The 6.x fallback adapter returns `nil` and defers to tokenizer auto-detection.

## 7. Conformance: when a plugin is "supported"

1. **Model-free tests** in `tests/Plugins/<Name>/` (or the engine's test
   target): probe, parameter table, load plan, runner state round trip.
   `TemplatePluginTests` is the pattern.
2. **Model-gated:** `LibraryContractTests` with
   `TINYTITAN_LIBRARY_CONTRACT_MODEL=<install>`, and greedy agreement with the
   plugin's declared oracle.
3. **Server:** `tools/api-feature-check.py` 10/10 against
   `TinyTitanServer --model <install>`, and a prompt-cache round trip: a
   repeat hits a checkpoint, a follow-up hits the live entry.
4. **A golden baseline** per installed width.

The conformance kit that runs these in one command is TT-131.

## 8. Adding a model as a plugin (from the template)

1. Copy `sources/Plugins/TemplatePlugin` to `sources/Plugins/<Name>`, rename the
   types, set `pluginID` and `displayName`.
2. Add the target and its library product to `Package.swift` with the same
   dependencies, and list the type in `EngineFactories.builtInPlugins`.
3. Make `probe` claim exactly this model's installs.
4. Replace `TemplateLoadPlan`, `TemplateModel`, `TemplateRunner` and the loop
   with the model's own, one at a time, keeping the template's tests green and
   adding the model's own.
5. Fill in `ParameterSupport` honestly.
6. Run §7. Remove the model's adapter, if it had one, in the same commit.

## 9. Versioning

- A minor version adds requirements, each with a default. A new parameter
  defaults to refused, so an older plugin reports the gap rather than hiding it.
- A major version may change requirements; the host refuses a plugin of another
  major.
