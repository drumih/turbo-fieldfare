# dsh-tinytitan

A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) bundle for
running models from a local **TinyTitan** server. It is deliberately thin: the
harness's own `llm-pi-ai` adapter serves these models, so this package adds
**configuration**, not a protocol implementation.

Two jobs, both at boot, both idempotent:

1. **Keeps the route current.** `tools/dsh_route.sh` in this checkout turns the
   installed models under `models/` into the `llm-pi-ai` route block — served
   ids, each template's thinking levels, and the three switches that are easy to
   get wrong by hand (`thinkingFormat: chat-template`, the keyless-route auth
   header, the long stream idle timeout). The plugin applies that block through
   the harness's own `settings` service, so the change lands in the active profile
   patch and hot-reloads; the model picker follows `models/` instead of a copy
   someone typed once.
   A plugin installed from a catalogue is a plain package beside no checkout, so
   there is no script to run: only then the same block is generated in-process
   from the server's own catalog (`generate.js`), and the log says so. Wherever
   `tools/dsh_route.sh` exists it stays the source of truth, so a checkout user
   has one implementation, not two.
2. **Mounts a compaction backend that does not think.** Compaction and session
   titles are marked `purpose: "compaction"` / `"session-title"` and name no
   reasoning level, so the harness fills in the route's default. On a local
   thinking model that spends a summariser's own output cap on thinking and
   costs tens of seconds on the title of every new session. The plugin registers
   an agent preset built from the harness's shipped `standard` composition with
   that row pointed at `dsh-tinytitan/backend`, which forces thinking off for
   those calls only — ordinary turns keep the route's level. While your profile
   has selected no preset of its own, this one becomes the default; an explicit
   choice on the Agent presets page is never overwritten.

## Supported harness version

This bundle supports **exactly one DeepSeek Harness release: `0.2.2-alpha.3`** —
not older, not newer, and not a build from `main`. Both jobs above are written
against that release: the preset is generated from _its_ shipped `standard`
composition (the `preset-standard` row of
`@deepseek-ai/dsh-web-app/presets/standard.patch.yml`, read through
`@deepseek-ai/dsh-agent-preset`'s `agentPresets.register`), so the row ids move
when the harness does, and the compaction backend subclasses that release's
`dsh-compaction-basic`.

The pin lives in three places that cannot import each other — `package.json`'s
`peerDependencies` (exact, no range), `SUPPORTED_DSH_VERSION` in `src/support.js`,
and **the bundle's own `version`**, which is the release it supports rather than a
number of its own — and `test/support.test.js` fails if any of them disagrees.

On any other release — older, newer, or a build from `main` — the plugin **refuses
to run**. It writes one line to **stderr** naming both versions and does nothing
else: no route, no preset, no watcher, and no writes into your harness home. It
never throws, so DSH boots normally, every other plugin loads, and removing this
plugin leaves nothing to undo. A version that cannot be read at all is refused the
same way, because a plugin that writes into the harness home has no business
proceeding on a harness it cannot identify.

stderr is not a style choice. The harness collects a plugin's log records and
prints them **only when the boot itself fails**, and its startup exporter takes
level ≥ 2, so a refusal logged through the host logger is invisible on a healthy
boot — which would be indistinguishable from a plugin that silently stopped
working. Found by booting a throwaway harness, not by reading.

## Install

```sh
dsh plugin --profile web add file:/path/to/TinyTitan/plugins/dsh-tinytitan
```

Restart DSH (or start a new session) and the plugin logs what it did. It writes
through the harness's own services, so the changes are part of your profile
configuration and hot-reload with it:

| What                      | Change                                                                                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| the `llm-pi-ai` entry     | the route block, refreshed from the catalog                                                                           |
| the agent-preset registry | a `tinytitan` preset, built from the harness's shipped `standard` composition with the compaction row on this backend |
| the selected agent preset | set to `tinytitan` **only** while the profile names no selection of its own                                           |

There is no settings file to edit and no preset file to copy: DSH 0.2.0 removed
both, and the plugin uses the replacements. If you would rather select the preset
yourself, set `setDefaultWhenUnset: false` — then the plugin registers `tinytitan`
and leaves your choice alone.

If you remove the plugin, set another preset on the Agent presets page first: the
selection it made is a normal setting, and nothing is left behind to clean it up.

A `file:` install is a copy, not a link: after editing this package, re-install
it (`dsh plugin --profile web remove dsh-tinytitan` then `add` again) or DSH keeps
running the copy it made.

## Upgrading from 0.1.6-alpha.2

The preset keeps its id (`tinytitan`), so **sessions that were already on it keep
working** — the id is what a session records, and only its declaration changed
from a generated file to a registry entry. Three things are worth knowing:

- On 0.2.0 the default preset is a registry field. The plugin points it at
  `tinytitan` **only while nothing else is selected**. If you had picked another
  preset, open the Agent presets page and pick **TinyTitan** to get the quiet
  compaction back; a choice you make there is never overwritten.
- `~/.dsh/.agent-presets/tinytitan/` (the 0.1.6 generated preset) is inert — 0.2.0
  does not read that directory. Deleting it is optional cleanup.
- `~/.dsh/settings.yaml.imported` is your old settings file, moved there by the
  harness when it imported it into the profile patch. Keep it; it is the record of
  what was migrated.

## Configure

Every field is optional; these are the defaults the `cordis.patch.yml` row writes
out, and `TINYTITAN_PORT` / `TINYTITAN_REASONING` / `TINYTITAN_CONTEXT` /
`TINYTITAN_MAX_TOKENS` / `TINYTITAN_REPO` / `TINYTITAN_SERVER` /
`TINYTITAN_MODELS_DIR` / `DSH_HOME` are the environment fallbacks.

The three switches that start work nobody asked for — `autoGoal` (a goal from
every prompt), `autonomy` (the fresh-agent `ralph` loop) and `handoff` (a child
session when the window's budget is spent) — are written out as `false` in that
row. Each one changes what every prompt in a profile means, so a profile opts in
by naming the field with `true`; installing the bundle never does it for you.

| Field                       | Default                                                                                                                                         | Meaning                                                                                                                                                                                                                                                                                                                      |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `port`                      | resolved: `config.port`, else `TINYTITAN_PORT`, else `8080`                                                                                     | the port the TinyTitan server serves on. Nothing in this bundle pins it, so the environment can point the route at a server on another port                                                                                                                                                                                  |
| `provider`                  | `tinytitan`                                                                                                                                     | the `llm-pi-ai` provider route name                                                                                                                                                                                                                                                                                          |
| `reasoning`                 | resolved: `config.reasoning`, else `TINYTITAN_REASONING`, else `medium`                                                                         | the route's declared default reasoning level. It must match how the server was started: a route that says "think" against a server running `--reasoning off` makes a dense Qwen spend its whole output budget inside the reasoning block and never answer                                                                    |
| `presetId`                  | `tinytitan`                                                                                                                                     | the agent preset this plugin registers                                                                                                                                                                                                                                                                                       |
| `context`                   | unset (the writer's `262144`)                                                                                                                   | the `contextWindow` the route declares, for the provider and every model row (`$TINYTITAN_CONTEXT`). Unset keeps the block byte-for-byte what the launcher's pin writes; naming one is how a narrowed route survives the next refresh. A positive whole number of tokens, or the row is refused at boot                      |
| `maxTokens`                 | unset (the writer's `32768`)                                                                                                                    | the output cap the same route declares (`$TINYTITAN_MAX_TOKENS`), with the same unset-means-pin and validation rules as `context` above                                                                                                                                                                                      |
| `registerRoute`             | `true`                                                                                                                                          | refresh the route block from `tools/dsh_route.sh`                                                                                                                                                                                                                                                                            |
| `watchModels`               | `true`                                                                                                                                          | keep watching `models/` and refresh when an install appears or disappears                                                                                                                                                                                                                                                    |
| `watchDebounceMs`           | `2000`                                                                                                                                          | how long the folder has to be quiet before the refresh runs                                                                                                                                                                                                                                                                  |
| `selfContained`             | `false`                                                                                                                                         | use the built-in generator even where `tools/dsh_route.sh` exists                                                                                                                                                                                                                                                            |
| `serverBinary`              | discovered                                                                                                                                      | the `TinyTitanServer` the built-in generator runs (`$TINYTITAN_SERVER`)                                                                                                                                                                                                                                                      |
| `modelsDir`                 | `<repoRoot>/models`                                                                                                                             | the installs it describes (`$TINYTITAN_MODELS_DIR`)                                                                                                                                                                                                                                                                          |
| `writeCompactionPreset`     | `true`                                                                                                                                          | register the `tinytitan` preset                                                                                                                                                                                                                                                                                              |
| `setDefaultWhenUnset`       | `true`                                                                                                                                          | select that preset only while the profile has selected none                                                                                                                                                                                                                                                                  |
| `compactionHeadroomTokens`  | unset (the harness's `65536`)                                                                                                                   | the compaction engine's headroom in the generated preset — see below                                                                                                                                                                                                                                                         |
| `autoGoal`                  | `false`                                                                                                                                         | arm a goal from every direct human prompt, so the harness keeps working until the model completes it — the `/goal <prompt>` behaviour without typing `/goal`                                                                                                                                                                 |
| `autoGoalRounds`            | `99999`                                                                                                                                         | the round cap an auto-created goal gets. Effectively unbounded, so an unattended run is not stopped by a small budget; set a lower number to bound prompts a person did not mark as long-running                                                                                                                             |
| `autonomy`                  | `false`                                                                                                                                         | the preset half of "work it until it is done": append the autonomy policy to the preset's `persona`, so the model chooses the best technical solution instead of asking, and mount the fresh-agent `ralph` loop — enabling the shipped row, or appending the harness's ralph bundle when the standard ships none — see below |
| `autonomyRounds`            | `64`                                                                                                                                            | the `maxRounds` the enabled `ralph` row carries; the harness enforces its own ceiling on a call override                                                                                                                                                                                                                     |
| `autonomySuppressQuestions` | `false`                                                                                                                                         | also remove the `ask_user_question` tool from the preset, so the model cannot ask. Off by default because plan mode's own instructions use that tool                                                                                                                                                                         |
| `handoff`                   | `false`                                                                                                                                         | continue an unfinished objective in a fresh child context before the session's window fills, so a task survives its own context limit — see below                                                                                                                                                                            |
| `handoffHops`               | `3`                                                                                                                                             | how many times one goal may be handed on before the chain stops and is left to a person                                                                                                                                                                                                                                      |
| `handoffAtTokens`           | unset (auto: 60% of the routed model's window, held 128K above its end — `131072` on 256K, `600000` on a 1,000,000-token route, `629145` on 1M) | the prompt-token count at which the next turn hands the objective on. A positive integer pins one number; unset follows the context window the LLM route declares, which is what keeps the hop below compaction on every window size                                                                                         |
| `handoffWindowRatio`        | `0.6`                                                                                                                                           | the fraction of that window the auto budget may use. The compaction trigger is a different fraction per shape (~0.625 of a 256K window, 0.8 of a 1M one), so a ratio has to clear the narrowest shape; above that, compaction continues the session in place and the handoff mostly only fires on the `max-tokens` wall      |
| `handoffMaxChildren`        | `8`                                                                                                                                             | how many handoff children may be alive at once in the process, ancestors included; has to sit above `handoffHops`                                                                                                                                                                                                            |
| `repoRoot`                  | this checkout                                                                                                                                   | where `tools/dsh_route.sh` lives                                                                                                                                                                                                                                                                                             |
| `dshHome`                   | `$DSH_HOME` or `~/.dsh`                                                                                                                         | the harness home, for the built-in generator's file fallback                                                                                                                                                                                                                                                                 |

The built-in generator looks for the server at `serverBinary`, then
`TINYTITAN_SERVER`, then `TinyTitanServer` on `PATH`, then the checkout's
release build; it looks for models at `modelsDir`, then `TINYTITAN_MODELS_DIR`,
then `<repoRoot>/models`. On a harness with no `settings` service it falls back
to editing the home's `settings.yaml` — the 0.1.6 shape — and refuses to write
when that file does not exist; through the service it makes no write at all when
the refresh would not change a byte.

0.1.6's `adoptDefaultPreset` (re-point the current default preset's stock
compaction row) has no counterpart here and the field is gone: presets are
declared rows now, so adopting a shipped one would mean freezing its whole plugin
list in your patch — the drift the generated preset exists to avoid. Select the
`tinytitan` preset instead; it is the default whenever you have chosen nothing.

**The compaction trigger is why `compactionHeadroomTokens` exists.** The engine
compacts at `min(window × thresholdRatio, window − maxTokens − headroomTokens)`.
The harness's default headroom is 65,536 tokens, and the generated preset leaves
that default alone, so on the route this project declares (262,144 window,
32,768 cap) the trigger sits at ~62% of the window rather than the documented
80%. That is a safe direction — less context, less KV on a local server — but on
a _narrow_ declared window the arithmetic flips: below roughly `cap + 65,536`
the pressure budget goes negative, the engine logs one warning and then **never
compacts**, and the session eventually fails on the context wall. So narrow the
window in the row, not by hand: `tools/dsh_route.sh --context <smaller>` writes a
route the plugin's next refresh — boot, or a change under `models/` — puts
straight back, unless the same numbers are in the row as `context` and
`maxTokens`. Then set `compactionHeadroomTokens` too, e.g. `0` to let the ratio
govern again or the cap's quarter (`8192`) to keep a guard: both leave a positive
budget at every window where `window > maxTokens`.

**With no server built, the folder is read directly.** The catalog normally comes
from `TinyTitanServer --catalog`, which is the authority on what an install is.
A profile that has installed models but has not built the server yet has nothing
to ask, so `src/catalog-scan.js` walks `models/` itself — the same rules, mirroring
`ModelCatalog.swift`, and `test/catalog-scan.test.js` compares its rows against
the binary's own output on the same folder so the two cannot drift. A binary that
exists but fails is treated the same way, and the reason goes to the log.

**The route follows the folder while the harness runs.** An install is a long
download someone starts and then wants to use, so `models/` is watched and the
route is rebuilt once the folder goes quiet (`watchDebounceMs`). Deletions count
too: a removed model stops being offered. The watcher is non-persistent and
unref'd — it never keeps the process alive — and a platform where watching fails
falls back to the boot-time refresh with a log line.

### Keep going without `/goal`

A goal is what makes the harness continue by itself: `@deepseek-ai/dsh-goal`
keeps one objective per session, `@deepseek-ai/dsh-goal-round-driver` queues the
next round while the agent is idle, and `@deepseek-ai/dsh-tool-goal` lets the
model complete or block it. Out of the box a person starts one by typing
`/goal …`; the model may also decide a request is goal-shaped.

With `autoGoal: true`, an ordinary manual prompt starts one too. The message
becomes the objective and the goal is created and armed exactly as `/goal` would
create it; everything after that is the harness's own machinery — no round is
scheduled here and no completion is judged here, so the round cap, the blocking
policy, the authority rules, and the arming rules after a session resume or fork
all still apply. A prompt that arrives while an unfinished goal exists is left
alone, so a follow-up steers the work already under way, and after a goal
completes the next prompt starts a fresh one. Subagent sessions are ignored: only
the session a person is driving turns a prompt into an objective.

```yaml
- id: dsh-tinytitan
  config:
    autoGoal: true
    autoGoalRounds: 24 # lower than the effectively unbounded 99999 default
```

Two things to know before switching it on:

- **Every** manual prompt becomes a goal, including a question you meant as a
  chat turn. The model is told to continue until it judges the goal complete, so
  it answers and then completes it; a model that instead keeps working spends
  rounds from the cap above.
- Completion needs the goal tool mounted. The shipped `standard` composition
  mounts `command-goal` and `tool-goal`, and the `tinytitan` preset inherits that
  list, so this works as-is here. A harness composed without goals leaves the
  switch a no-op with one log line, and boot is unchanged either way.

**How it is checked.** `test/keep-going.test.js` pins the four things the switch
must not do — feed on the driver's own round, arm a goal for a subagent, replace
an unfinished goal, or take a boot down when the harness has no goal service —
twelve cases, with no harness and no model.

The behaviour itself was verified on DSH 0.2.0-rc.2 against a local Qwen 3.5 4B,
in a scratch home carrying this plugin with the option on, driven by
`dsh headless --json` (which waits for quiescence, so goal rounds run inside the
one invocation):

- **`autoGoal: false`** — a prompt completed its turn with **no** `goal/change`
  in the session log and no rounds. The control: the switch is what creates them.
- **`autoGoal: true`, cap 3** — the same prompt wrote a `goal/change` **create**
  carrying that prompt as its objective and cap 3; its own turn completed; then,
  with no further input, the driver queued **goal round 1** and the model called
  `get_goal` and `update_goal` and completed it. Exit 0.
- **the same session, a second prompt** — a second create and complete, each
  with its own objective, and no cross-contamination.

Not exercised live: the round-limit block, because the model completed in round 1
both times. The configured cap does reach the goal — the create event carries it
— and the blocking policy belongs to `dsh-goal-round-driver`, not to this plugin.

### Autonomy: work it until it is done

`autoGoal` keeps a session working; it cannot stop a model from politely stopping
to ask. `autonomy: true` is the preset half of that policy, and it changes three
things in the generated `tinytitan` preset:

- the `persona` row gains the autonomy policy — never ask what you can decide,
  choose the best technical solution and state it as an assumption, keep going,
  and treat "done" as the tests and gates passing with the evidence shown, with
  the workspace rather than the conversation as the authority;
- the fresh-agent `ralph` loop is mounted with `maxRounds: autonomyRounds`: a
  composition that still ships the row (0.2.0 profiles) has its **disabled**
  row enabled in place, and a composition that ships none (0.2.1's standard
  dropped it) gets the harness's own experimental ralph bundle appended — the
  isolated `optional-ralph` group carrying its workflow engine and the
  `tool-ralph` row, exactly as `@deepseek-ai/dsh-experimental-ralph-bundle`
  declares it. Ralph is the fresh-agent loop: each round is a new child agent
  that sees only the immutable objective, its round and cap, the
  shared-workspace-as-authority instruction, and the previous round's bounded
  structured report. That is what carries a task past a session's context
  limit — the work moves to a fresh context instead of dying with the old one;
- with `autonomySuppressQuestions: true` the `ask_user_question` tool is removed
  from the preset, so asking becomes impossible rather than discouraged. It is
  off by default because the shipped plan-mode instructions use that tool for
  user-owned choices, and a session that needs plan mode wants it kept.

Two honest limits. **Ralph's rounds are child sessions, not successor root
sessions**: the session you are looking at stays the one you are in, and the
recap across rounds is the bounded report plus the workspace, not your
conversation. And **none of this is unbounded**: ralph stops at `maxRounds`, an
auto-goal stops at `autoGoalRounds` (99999 unless you lower it), a `max-tokens`
turn still disarms a goal, and provider errors, quota and disk remain real
ceilings.

```yaml
- id: dsh-tinytitan
  config:
    autoGoal: true
    autonomy: true
    autonomyRounds: 64
```

### Handoff: continue past the context wall

`autoGoal` and `autonomy` both keep working _in the same session_, and a session
has a window. `handoff: true` is the option that moves the objective somewhere
else before that window fills, which is the only part of "until it is done" the
harness does not do by itself.

At the start of a turn — the one moment a plugin may start a subagent, because
the turn's agent loop is active — the driver looks at the previous step's prompt
tokens. If they reached the budget and the session's goal is active and armed,
one **hop** happens:

1. a `fork` subagent is started: a real child session, seeded from this one's
   completed turns, handed a prompt that leads with the objective verbatim and
   tells it the workspace is the authority;
2. the handing-off session is disarmed, so only one context works the objective
   at a time;
3. the child's inherited goal is **armed** (a fork seed copies the goal's events,
   but its activation is process-local, so the child would otherwise sit there
   with an objective and no rounds);
4. the child is held while its goal is actively continuing, and released — leaf
   first — when it is idle and no longer continuing. Disposing a parent disposes
   its subagent children, so a chain unwinds from its end.

**Where the budget comes from.** Unset, it is the smaller of `handoffWindowRatio`
(0.6) of the context window the routed model declares and the window minus a
128K reserve — `131072` on a 256K route (the reserve binds), `600000` on a
1,000,000-token route and `629145` on a 1,048,576-token one (the ratio binds).
The ratio is what a live 1M route was pinned to by hand after measuring its
sessions at ~400,000 prompt tokens with compaction at ~800,000; the reserve is
the second bound, and it keeps the hop a full output turn below the compaction
trigger, because the trigger is a different fraction per shape (~0.625 of 256K,
0.8 of 1M) and a step that crosses the budget must not be able to outrun it
before the next turn opens. A route whose window the adapter cannot report — and
a window too small to hold the reserve — falls back to `120000`. Set
`handoffAtTokens` to pin one number, or move `handoffWindowRatio` (e.g. `0.75`
for fewer, later hops where the reserve has room).

The budget is the normal path, and the session's own token limit is the fallback.
One turn can jump straight past the window and end on `max-tokens`; the round
driver disarms the goal for that event, which would leave no next turn to hand
off from. The driver therefore watches `turn/end`: when the wall was a _context_
wall (the reported prompt tokens reached the budget, not just the output cap) and
a hop is still available, it re-arms the goal so the next turn hands the objective
on. Each goal gets at most two such re-arms, so a chain that cannot start cannot
loop on the wall.

Two honest limits. **The successor is a child session, not a successor root
session**: it has its own log and context and keeps working autonomously through
the same goal driver, but it is not a chat you steer. And **nothing here is
unbounded**: `handoffHops` caps one goal's chain, an explicit `handoffAtTokens`
must be positive and `handoffWindowRatio` must be above 0 and at most 1 (a zero
budget would hand off at the very first turn), `handoffMaxChildren` caps the
process, every hop is released when it stops, and a goal the model blocked is
never handed on.

```yaml
- id: dsh-tinytitan
  config:
    autoGoal: true
    handoff: true
    handoffHops: 3
    # unset follows the route: 131072 on a 256K window, 600000 on a 1M one
    handoffWindowRatio: 0.6
```

**How it is checked.** `test/handoff.test.js` pins the driver without a harness
or a model — 29 cases covering the budget (including cached input and the
window-aware default), the trigger conditions, the hop counter's fork-bomb
regression, the in-flight guard, the refund of a refused start, the process cap,
the release rules, and the wall recovery's three bounds. The live behaviour was verified on DSH 0.2.0-rc.2
against a local Qwen 3.5 4B in a scratch home, driven from inside the booted
profile — `dsh headless` cannot show this, because it exits as soon as its one
agent is idle and that kills a handoff child mid-turn:

- a prompt whose second turn spent the budget produced **hop 1/2** with a real
  child session, and the log shows the handing-off session disarmed and the
  child's inherited goal armed;
- that child's own next turn produced **hop 2/2**, from the child rather than the
  original session — the chain, not a fan of siblings;
- the last child's next trigger logged **"reached 2 hops"** and no third child
  was created, so the cap holds live;
- each child was **released** once its goal stopped continuing, leaf first, and
  the driver then saw a quiet process: three sessions, every agent disposed, no
  hang. An earlier version released the one-shot run when it settled, which
  killed the child before its round driver could queue the continuation; the
  grandchild's turn ended `aborted/disposed` 3 ms after its parent was released,
  which is why the release rule is what it is.

The wall path was verified separately and deterministically, by running the same
scratch profile with the route's `maxTokens` at **8** so that every turn ends on
`max-tokens` — the harness's own full-session disarm:

- the first turn ended on the wall, the driver logged _"the session hit its token
  wall at 792 tokens; re-armed the goal so the next turn hands the objective
  on"_, and the next turn produced **hop 1/2**;
- the child's turn hit the same wall and produced **hop 2/2**;
- the next wall logged _"hit its wall at 1126 tokens, but no hop is available
  (hops); leaving it to a person"_, no third child appeared, and every child was
  released leaf first — three sessions, clean exit. The session logs carry the
  `turn/end max-tokens` → `goal/change resume` → `turn/start` sequence for each
  hop, which is the loop requirement 6 asks for.

## What it does not do

- **No adapter.** It registers no LLM provider: the harness's `llm-pi-ai` route
  does the work, so a harness upgrade cannot leave a copied protocol
  implementation behind.
- **No budgets, no vision, no dialects.** TinyTitan accepts `reasoning_budget_tokens`
  and does not enforce it; the models are text-only; llama.cpp/TabbyAPI are other
  servers with their own routes. None of that is here.
- **No session-title override.** Titles are issued by a host-plane plugin with
  its own context, which a bundle cannot wrap; that half is upstream ask 1 in
  `docs/dsh-upstream-asks.md`. Until it lands, the route's `reasoning: off` is
  the only way to keep titles unthinking — at the cost of chat starting
  unthinking too.

## Uninstall

```sh
dsh plugin --profile web remove dsh-tinytitan
```

Nothing is left on disk to clean up: the route and the preset are profile
configuration, and removing the row stops both being refreshed. One thing to
undo by hand — if this plugin selected the `tinytitan` preset for you, pick
another preset on the Agent presets page before you remove it, because the
selection outlives the plugin that made it.

## Test

```sh
cd plugins/dsh-tinytitan && node --test test/
```

Seventy-nine tests, no harness packages required: the compaction seam is
exercised against a stub base, the preset builder, the default-preset rule and
the route preferences against stubs and temporary homes, and the route call
against a stubbed runner. The built-in generator's block is compared
byte-for-byte with `tools/dsh_route.sh --print` — on the real catalog when the
checkout's server binary is built, and on a synthetic catalog (mixed backends,
re-sorted thinking levels) whenever the shell tool is present; both comparisons
skip with a clear message when the pieces are absent, so the suite stays
runnable elsewhere. The `dsh-tinytitan/backend` import itself is resolved from
the profile's own `node_modules` once installed.

## Licence

**MIT, deliberately.** The repository it lives in is Apache-2.0, and this package
says MIT on purpose: it is an independent work that talks to the server over its
public HTTP API and copies no code from the project's lineage, so it stays under
the licence its own author chose. Do not "align" it with the repository's
`LICENSE` — that would be a claim about provenance this package does not make.

## Publishing it

`docs/dsh-plugin-publication.md` in this repository is the research: how a DeepSeek
Harness plugin is distributed, the community catalogue that lists it, and exactly
what this package still needs before it can be submitted.
