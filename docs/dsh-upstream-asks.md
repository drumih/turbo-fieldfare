# Three asks for DeepSeek Harness, from running it against a local TinyTitan server

Date: 2026-09-14; **posted 2026-09-18**. Re-verified before posting against the
installed `@deepseek-ai/dsh` **0.1.6-alpha.2** and `@earendil-works/pi-ai` 0.85.1,
and the line numbers below are those versions' (the first draft was written
against 0.1.5-rc.2). Upstream (`deepseek-ai/deepseek-harness`) has **issues
disabled and discussions enabled**, so all three are Discussions in *Ideas*:

| # | Ask | Discussion |
| --- | --- | --- |
| 1 | Honor `purpose` when an auxiliary call names no reasoning level | [#7109](https://github.com/deepseek-ai/deepseek-harness/discussions/7109) |
| 2 | Map pi-ai's reasoning usage into `reasoningTokens` | [#7110](https://github.com/deepseek-ai/deepseek-harness/discussions/7110) |
| 3 | Allow `dsh web --host` to bind a specific LAN interface | [#7111](https://github.com/deepseek-ai/deepseek-harness/discussions/7111) |

The patches are small enough to apply locally meanwhile.

> **Status note, 2026-10-24: not re-verified since.** The harness this project's
> plugins pin has moved to `0.2.2-alpha.3`; this project does not install a harness
> of its own, so the user's DSH is the one to read. Everything below — the line numbers, the quoted behaviour and the
> two replies — was read on `0.1.6-alpha.2` and is left as the record of that
> version. Ask 1 is the one 0.2.0 was most likely to change, because that release
> reworked the settings and preset planes; re-read the three asks against the
> installed harness before quoting them as current.

**Two replies arrived from `argszero` (2026-09-18/19), and both were re-verified
against the installed `0.1.6-alpha.2` before this doc was corrected in place.**
The asks hold. The first suggested patch was wrong in a way that would have
turned working configurations into silent no-ops, and the second does not
compile as written; both corrections are below, and a working third-party
plugin for the first now exists.

Both are about the *seam*, not about any one provider: they change what every
route gets, and both already hold on the first-party DeepSeek adapter.

---

## 1. Honor `purpose` when a call names no reasoning level

**What happens now.** The harness marks its auxiliary model calls with a
`purpose`, and they name no `reasoningEffort`:

- `@deepseek-ai/dsh-compaction-basic` — `summarizeWithLlm`
  (`lib/index.js:299`) streams with `purpose: "compaction"` and no
  `reasoningEffort`;
- `@deepseek-ai/dsh-session-title-llm` (`lib/index.js:216`) streams with
  `purpose: "session-title"` and no `reasoningEffort`.

`@deepseek-ai/dsh-llm` then fills the gap from the *route*:
`resolveCallWithInfo` (`lib/index.js:2136-2147`) takes
`const effective = requested ?? reasoning.defaultEffort`. So on any route whose
profile default level is a thinking level, compaction and session titles think.

**Why that is wrong rather than merely expensive.**

- Those calls exist to produce a *bounded* amount of visible text — a checkpoint
  or a title — inside an output cap the caller chose (`maxTokens`). Thinking
  spends that cap first, which is the failure the old Qwen plugin's author
  described: a summariser whose thinking eats the cap truncates the checkpoint.
- On a local model the cost is wall-clock, not quota. Measured here on a base M3
  with 24 GB and Qwen3.8-Flash-Next 4-bit: **~5.8 tok/s** decode, so a few
  hundred thinking tokens is minutes per compaction and tens of seconds per new
  session's title.
- The harness already decided this for its own route:
  `@deepseek-ai/dsh-llm-deepseek`'s `resolveThinking`
  (`lib/index.js:32`) returns `{ thinking: "disabled" }` for
  `purpose === "session-title"`, and the compaction call carries
  `x-deepseek-harness-compact: 1` (`lib/index.js:1276`) for the server to act on.
  A pi-ai-backed route has no equivalent, so the same harness behaves two ways
  depending on which adapter serves the model.

**Suggested patch, corrected by the reply.** The shape first proposed here —
force `off` whenever the purpose is auxiliary — is wrong, because a named
`reasoningEffort` is not a preference: it is a claim that the route supports
that level, and `resolveCallWithInfo` enforces the claim in two places
(`packages/llm/llm/src/index.ts:886-901`):

- on a **non-reasoning route** (`reasoning === undefined`) it raises
  `UNSUPPORTED_REASONING_EFFORT`;
- on a **reasoning route whose `efforts` omit `off`** it raises the same. And
  `off` is easy to omit: `resolveModelReasoning`
  (`llm-pi-ai/src/catalog.ts:710-745`) writes `map[level] = null` for every
  level a `reasoningEfforts` dict did *not* name, and
  `getSupportedThinkingLevels` (`llm-pi-ai/src/models.ts:59-66`) drops exactly
  those — so `reasoningEfforts: { low, high }` advertises a reasoning route
  with no `off` in it. (Spelling `off:` with an empty value keeps it; leaving
  the key out is what removes it.)

Measured on three declared pi-ai routes, dispatching the auxiliary-call shape
(`purpose: "compaction"`) with `off` named:

| route profile | `resolveModelInfo().reasoning` | auxiliary call naming `off` |
| --- | --- | --- |
| `reasoningEfforts: { low: …, high: … }` | `{ efforts: [low, high] }` | `finish` `{"kind":"error","code":"UNSUPPORTED_REASONING_EFFORT"}`, **zero content chunks** |
| `reasoningEfforts: { off:, low: …, high: … }` | `{ efforts: [off, low, high] }` | dispatched normally |
| `reasoningEfforts: false` | `undefined` | `finish` error `UNSUPPORTED_REASONING_EFFORT`, zero content chunks |

The failure is silent where it matters: `stream()` does not throw at the call
site — the refusal becomes a terminal `finish` chunk carrying the error and no
content. A compaction produces no checkpoint and a session-title call produces
a title-less session. The fix therefore has to **ask** the capability rather
than assume it: `reasoning` present **and** the target level ∈ `efforts` ⇒ name
it; anything else ⇒ pass the request through unchanged. That is exactly what
the public `ctx.llm.resolveModelInfo(provider, model, signal)` answers
(`packages/llm/llm/src/index.ts:733`).

**Also measured on the reply, and not in the original ask:** `purpose ===
"session-title"` is the **only** purpose `resolveThinking` special-cases in
`llm-deepseek` (`protocols/chat-completions/serialize.ts:77`), so **compaction
thinks even on the harness's own adapter** — `x-deepseek-harness-compact: 1`
asks the *server* to act, it does not stop client-side thinking. The split is
title vs compaction inside deepseek as well as pi-ai vs deepseek.

**Mountable meanwhile.** The reply author shipped
[`@argszero/cordis-plugin-aux-reasoning@0.1.0`](https://www.npmjs.com/package/@argszero/cordis-plugin-aux-reasoning)
([repo](https://github.com/argszero/cordis-plugin-aux-reasoning)), which joins
the public `llm/stream` waterfall — it sees the raw options, `purpose`
included, before `resolveCallWithInfo` — looks the route up with
`resolveModelInfo`, and re-dispatches as a new object only when the target
level is genuinely offered and the route's declared default differs. Every
uncertain case is a pass-through plus a report, never a refusal; the target
defaults to `off`. 25 tests pass against four published `dsh-llm` lines
(0.1.2-rc.1, 0.1.3-alpha.2, 0.1.5-rc.2, 0.1.6-alpha.2).

**Local mitigation meanwhile.** TinyTitan ships `plugins/dsh-tinytitan`, a thin
bundle that mounts a compaction backend forcing `off` for compaction calls —
not titles, because a host-plane plugin's context cannot be wrapped, which is
ask 1's title half. It names `off` unconditionally, which is correct for the
routes this checkout emits (`tools/dsh_route.sh` and `generate.js` always list
`off`), but a route that omits it would silently produce no checkpoint; the
capability query above is the hardening if this override is ever pointed at a
foreign route.

**Where the guard lives — answered by the reply, 2026-09-19.** Asked whether the
capability check belongs inside `resolveCallWithInfo` or in the layer that selects
the level, the reply recommends the **selection layer**, for four checkable
reasons:

1. `purpose` is not on the resolver's input. `resolveCallWithInfo(config, info)`
   takes `LlmCallConfig`, a six-field scalar bag (`provider`, `model`,
   `reasoningEffort`, `temperature`, `maxTokens`, `stop`); `purpose` lives on
   `GenerateOptions`. Verified on our installed `0.1.6-alpha.2`: `purpose` appears
   in `dsh-llm` only in `lib/types/types.d.ts`, never in `lib/index.js`.
2. The resolver has three call sites, so a guard there fires for every call and
   must itself answer "is this auxiliary" — the input it does not have.
3. The layer already normalizes: `resolveCallWithInfo` writes the effective level
   back into the returned config (`if (requested !== effective) resolvedConfig =
   { ...defaulted, reasoningEffort: effective }`), so choosing the level is
   already its habit.
4. Its only failure vocabulary is `UNSUPPORTED_REASONING_EFFORT`, whose
   consumption is fatal — a terminal `finish` with zero content. A policy
   decision expressed there becomes a new silent no-op in the configurations the
   guard exists to protect.

If a capability check does live in the resolver, the refinement is that it must
**normalize rather than reject**: a level the route cannot express is dropped to
the route's default (or to no level), never routed into the terminal error. Then
the two homes compose — the selection layer names only levels the route lists,
and the resolver never becomes the enforcement point for a policy decision.

The deciding test the reply offers: the rule must be expressible **without
widening `LlmCallConfig`**, and the selection home wins that outright. That leaves
one acknowledged gap — a caller that names a level explicitly is not covered by
the selection home — which is the resolver's job as a *capability* rule, not the
policy's.

**The plugin-side half, from a third reply (2026-09-19).** `wangzhanchao883`
adds the case this ask does not cover: a third-party plugin's *own* model call
is not one of the harness's auxiliary calls, so it cannot borrow the `purpose`
classification at all — the vocabulary is `compaction`/`session-title`, and a
plugin-invented value is in no allowlist. The plugin can only query the
capability at its own call site and then name a level.

Its failure shape is worse than the one above. An error `finish` at least
carries a code; the plugin-side failure returns normally, throws nothing and
carries no code — the caller simply never gets the JSON, and a caller that
retries on a thrown error never retries. They hit it twice, and lost a second
round to the "no error" part. This is exactly why `plugins/dsh-tinytitan`'s
compaction override names `off` unconditionally *only* while the routes this
checkout emits list `off`: on a foreign route it would fail in that silent way,
and the capability query is the hardening.

The reply also corrects its own earlier advice (#3468, #6797) — "just pass
`reasoningEffort: off`" — which holds only where the route lists `off`, and
points at field notes in #6857.

---

## 2. Map pi-ai's reasoning usage into `reasoningTokens`

**What happens now.** pi-ai parses the provider's split
(`@earendil-works/pi-ai/dist/api/openai-completions.js:1201`):

```js
reasoning: rawUsage.completion_tokens_details?.reasoning_tokens || 0,
```

`@deepseek-ai/dsh-llm-pi-ai`'s `mapUsage` (`lib/index.js:1403-1410`) then maps
`input`, `output`, `totalTokens`, `cacheRead` and `cacheWrite` — and drops
`reasoning`:

```js
function mapUsage(usage) {
  return {
    inputTokens: usage.input,
    outputTokens: usage.output,
    totalTokens: usage.totalTokens,
    ...usage.cacheRead > 0 ? { cacheReadTokens: usage.cacheRead } : {},
    ...usage.cacheWrite > 0 ? { cacheWriteTokens: usage.cacheWrite } : {}
  };
}
```

**Consequence.** The harness's `TokenUsage.reasoningTokens` is populated on the
DeepSeek route (visible in a session transcript as
`{"inputTokens": 9707, "outputTokens": 153, "totalTokens": 11140, "cacheReadTokens": 1280, "reasoningTokens": 132}`)
and never on a pi-ai route, so the context meter and any per-turn accounting
cannot separate thinking from answer there — exactly where local models spend
most of their output tokens. TinyTitan does report the field; the adapter discards
it. The doc comment ("reasoning folded into output by pi-ai") describes pi-ai's
*output* bucket, not the separate count pi-ai also returns.

**Suggested patch, corrected by the reply.** `reasoning` is optional in pi-ai's
`Usage` (`@earendil-works/pi-ai/dist/types.d.ts:277`), so the one-line form
above does not compile under strict null checks; it has to guard the type as
well as zero:

```js
...usage.reasoning !== undefined && usage.reasoning > 0
  ? { reasoningTokens: usage.reasoning } : {},
```

`> 0` is still the right predicate, but not as a copy of the cache fields'
"only when non-zero" rule: `dsh-llm-deepseek`'s `mapUsage` keeps a reported
`0` (`reasoning !== undefined`,
`protocols/chat-completions/translate.ts:70`), and transposing *that* predicate
here would emit `reasoningTokens: 0` on **every** pi-ai turn. pi-ai's own
mappers collapse absence into `0` before the harness sees it —
`reasoning: rawUsage.completion_tokens_details?.reasoning_tokens || 0`
(`openai-completions.js:1201`, and the same shape in
`openai-responses-shared.js:449`, `google-generative-ai.js:171` and
`google-vertex.js:180`) — which is why `!== undefined && > 0` is what makes a
non-thinking turn omit the field while a reported `0` also omits it.

**Boundary worth putting in the record.** On the completions / responses /
google paths, "the provider reports no breakdown" and "the provider reports
zero" both surface as `0`, so the mapped field cannot separate them. It reports
thinking tokens where thinking was counted and stays absent otherwise, which is
the most the harness vocabulary can express without a second tri-state field.

**Acceptance.** On a pi-ai route, a thinking turn's `assistant/message` record
carries `reasoningTokens` matching the provider's
`completion_tokens_details.reasoning_tokens`; a non-thinking turn omits it, as
the cache fields do. The reply adds two cases to `llm-pi-ai`'s tests: a
non-thinking turn, and a provider that reports a count.

**Not mountable.** The reply checked before assuming: the value is discarded
inside the adapter, before any chunk exists. The only surfaces a plugin can
see are the mapped `TokenUsage` on the usage chunk and the terminal `finish`
chunk's `ReplayEnvelope`, and `toPiReplayState` (`llm-pi-ai/src/replay.ts:77-111`)
carries no usage and no cost. A plugin could only invent a count, which
per-turn accounting should not accept as provider-reported. This belongs in
the adapter.

---

## 3. Allow `dsh web --host` to bind a specific LAN interface

**What happens now.** The webserver schema takes two literals and no others:
`host: z.union([z.const("127.0.0.1"), z.const("0.0.0.0")]).required()`
(`@deepseek-ai/dsh-host-webserver` `lib/index.js:141`). The Web app then refuses
the all-interfaces one at startup (`@deepseek-ai/dsh-web-app` `lib/startup.js:40`):
*"error: --host 0.0.0.0 is intentionally not supported yet for safety: it would
expose remote code execution to the network; use 127.0.0.1 instead"*.

**Consequence.** Nothing built on the Web UI — a LAN manager for a small fleet of
harness instances, say — can be reached from another machine, and a specific
interface cannot be named at all.

**The fence it would need is already there.** `resolveLanTrust`
(`lib/index.js:83`) computes the machine's non-internal IPv4 addresses, folds
them into `trustedHosts` for the `/api` browser-trust fence, and the ready banner
announces a LAN candidate (`lib/index.js:199`) — every bit of it gated on
`bindHost === "0.0.0.0"`, which startup rejects, so on the CLI path it is
unreachable. A non-wildcard bind would also need the bound address added to the
fence: that function returns an empty `lanAddresses` for anything but the
wildcard.

**Suggested shape.** Accept an explicit interface literal — `--host 192.168.1.5`
— and keep rejecting the wildcard, or gate the wildcard behind an explicit
opt-in. One address exposes the listener only on the subnet that address is on.
If remote code execution is the concern, an opt-in that also requires a token or
an allowlist is the safer form than a silent wildcard bind.

**Acceptance.** The bound literal serves and is printed in the ready banner;
another machine on the same subnet can load it; and the `/api` fence still
rejects a Host header that is neither the bound address nor an explicit
`--trusted-host`.

**The reply (2026-09-19, `PerryLink`), and one correction to our reading.** Our
description of the tree holds except for *which* check is the gate:

- The startup guard is not it. `packages/bundle/web-app/src/startup.ts:74-76`
  refuses the literal `0.0.0.0` and nothing else — `--host` is an unrestricted
  `.option('--host <host>')` (`:51`), so `--host 192.168.1.5` passes the action,
  is published to the bundle patch, and reaches the webserver row.
- It dies one layer down, in the webserver's own schema:
  `host: z.union([z.const('127.0.0.1'), z.const('0.0.0.0')]).required()`
  (`packages/host/webserver/src/index.ts:126`), above the comment naming "the two
  supported values" (`:59-61`). An interface literal dies at schema validation;
  the wildcard dies earlier, at the CLI guard.
- The LAN-trust half is as we wrote, and its second half is the sharper point:
  `resolveLanTrust` branches on the wildcard literal
  (`packages/bundle/web-app/src/index.ts:126-130`), so any other bind returns
  `lanAddresses: []` and a `trustedHosts` of only the explicit `--trusted-host`
  values (`:131`); the banner reuses that snapshot (`:263-271`), and the fence
  admits a non-loopback Host only from `trustedHosts`
  (`packages/client/connection/src/api-request-trust.ts:103`, 403/401 at
  `rpc-host.ts:97-100`). A specific-interface bind would need its address folded
  into that list.
- On the security consideration: "requires a token" is **already true** —
  `/api` wants the launch token's signed cookie on top of the fence, minted into
  the index URL by `authenticatedUrl` (`browser-auth.ts:223-247`) — so an opt-in
  flag would not be adding authentication. A LAN bind changes the *reach*, not
  the credential, while that credential is printed in the ready banner. Whether
  the trade is acceptable is a deployment call, not the maintainer's.
- **What a user can do today:** remote access has to arrive *at* loopback. The
  documented path is an SSH launch (the URL still prints, the browser handoff is
  suppressed); a local proxy or tunnel to `127.0.0.1` also works. The reply
  grepped for an explicit `ssh -L`/`LocalForward` recipe and found none in the
  tree — the docs describe the SSH case without spelling out the forwarding
  command — so that is unverified **as documented guidance**.

---

## Applying these locally

The first two are small enough to carry as a patch against the installed package
while upstream decides — but an upgrade replaces `node_modules`, so re-apply
after one. The first is worked around by `plugins/dsh-tinytitan` for compaction,
and now also by the third-party `@argszero/cordis-plugin-aux-reasoning` for both
purposes; the seam fix is still the right home for the default, and any local
patch has to take the corrected "ask, then name" shape rather than forcing
`off`. The second cannot be worked around from outside the adapter, which is
why it is the more valuable of the two, and its corrected form is one line. The
third is a schema change in two packages, so it is upstream or a fork rather
than a patch.
