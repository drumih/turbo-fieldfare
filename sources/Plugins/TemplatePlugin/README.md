# TemplatePlugin: the golden base

Every LLM joins TinyTitan as a plugin copied from this one. It is a standalone
library over `TinyTitanPluginAPI` and Core, compiled into the build, and serves
a miniature counting model that needs no weights, so every requirement of the
protocol is implemented and tested here without a model.

Start with `docs/plugin-api-v1.md` §8 ("Adding a model as a plugin").

| File | Requirement it shows |
|---|---|
| `TemplatePlugin.swift` | `EnginePlugin`: identity, probe, parameter table, reasoning, the factory |
| `TemplateLoadPlan.swift` | load arithmetic from `--max-context` and `--ram-budget` |
| `TemplateModel.swift` | `EngineModel` |
| `TemplateRunner.swift` | `EngineRunner`: prefill, decode, state snapshots, statistics |
| `TemplateCompletion.swift` | the generation loop contract (streaming, stops, KV accounting) |
