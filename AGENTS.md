<!-- graft:start -->
## Graft , repo context graph

This repo is indexed in `graft/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here , understanding how something works, finding where code lives,
or scoping a change , get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `graft map` first , a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `graft ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer , cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete , run `graft grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `graft skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `graft callers <symbol>` gives precomputed, exact edges , who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `graft/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects ,
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `graft ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to , never
re-read whole files.

After big code changes, refresh the graph with `graft build` (deterministic,
no API key, $0).
<!-- graft:end -->

# TurboFieldfare

Swift and Metal inference for Gemma 4 26B-A4B on Apple Silicon.

## Scope

This checkout is for running and reporting existing behavior. Do not edit source, change runtime defaults, or start optimization work unless the user asks.

## Layout and commands

`Sources/TurboFieldfare/` is the runtime; `Sources/TurboFieldfareRepack/`,
`Sources/TurboFieldfareCLI/`, `Sources/TurboFieldfareServer/`, and
`Sources/TurboFieldfareApp/` contain the installer, CLI, loopback server, and
Mac app.
`Tests/` contains focused public tests; `docs/` contains design, benchmark, and experiment notes.

```bash
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo
swift run -c release TurboFieldfareRepack --output scratch/gemma4.gturbo --resume
swift build -c release
.build/release/TurboFieldfareMac
swift run -c release TurboFieldfareCLI \
  --model scratch/gemma4.gturbo \
  --prompt "The capital of France is" \
  --max-new 64
```

The installer streams the pinned model without staging the full source checkpoint. Set `HF_TOKEN` only if requested. The download is about 15 GB. Cancellation preserves verified completed ranges; continue them with `--resume` or remove them with `--discard-partial --output scratch/gemma4.gturbo`.

## Local server

Follow the [server guide](docs/OPENAI_SERVER.md) for launch commands, health
checks, client setup, prompt reuse, tool loops, and supported API behavior.
Apply the model-process checks below first; never start a second model process
or terminate an existing one.

Keep the server on `127.0.0.1`; it has no remote authentication or TLS, so do
not proxy, tunnel, or expose it. A tool call from the local model never bypasses
the client's normal permission policy. Keep the execution session alive while
the server is needed, and stop only a server you launched.

## Test rules

Before a model run, require macOS 26+, Swift 6.2+, enough disk, acceptable `memory_pressure -Q`, a completed `scratch/gemma4.gturbo`, and no process from `pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. If a check fails, inform the user and stop; do not terminate apps or delete or reinstall the model.

Run package tests through `Scripts/test.sh`. Run only one app, CLI, or model-using test at a time.

For performance results, build release once and follow the [community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) exactly. Do not enable experimental controls or profiling.

Do not download a full checkpoint, duplicate the `.gturbo` model, create a worktree, or purge caches just to run tests.

Report the commit, hardware and RAM, macOS, Swift version, exact command, exit code, complete timing footer or error, and every protocol deviation. Treat results as measurements, not performance ceilings.

## App controls

The Mac app sends prompts through the pinned Gemma 4 IT chat format. It
exposes context length, temperature, Top-K, Top-P, expert-cache slots, prefill,
and RDADVISE. The defaults are temperature `0.2`, Top-K `64`, and Top-P `0.95`.
Responses can use the context space left after formatting the prompt, and FP16
is the runtime KV format. The HUD shows generation rate, token count, and
decode-service memory; Last run also shows time to first token and I/O. Build
the app with its sibling `TurboFieldfareDecodeService`; it never loads a second
in-process model. See [README](README.md) and [Runtime controls](docs/RUNTIME_CONTROLS.md).
