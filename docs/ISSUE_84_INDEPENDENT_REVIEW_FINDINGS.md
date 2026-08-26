# Independent Review Findings: Issue #84

Read-only investigation performed 2026-08-14 against `origin/main` (`3e87d92`)
per `docs/ISSUE_84_INDEPENDENT_REVIEW_BRIEF.md`. No source files were changed.

GitHub state at review time:

- Issue #84: OPEN. Comments: two maintainer replies, then the independent
  reproduction from `idallasj` (2026-08-13, comment 5284001871).
- PR #90 (diagnostics): merged; the reproduction used its output.
- PR #107 (grammar-constrained decoding): OPEN, CONFLICTING, no CI checks.

## 1. Verdict: immediate failure mechanism — understood, with one correction

The brief's step-by-step mechanism is confirmed against `main`:

- `StructuredAssistantDecoder.consume` collects payload token IDs between
  `<|tool_call>` and `<tool_call|>`, re-decodes them with
  `decode(skipSpecialTokens: false)`, and hands the text to
  `GemmaToolCallParser.parse`
  (`Sources/TurboFieldfare/Tokenization/StructuredAssistantDecoder.swift:71-87`).
- On parse failure, `ServerInference` records the error and sets
  `shouldStop = true`
  (`Sources/TurboFieldfareServer/Core/ServerInference.swift:590-593`).
  `RawCompletion.swift:221-226` folds `stopMatcher.isStopped || shouldStop()`
  into one label and records `reason = .stopString`. **`raw_stop=stop_string`
  in the diagnostics is the server's own post-error abort, not a client stop
  string.** The brief's nuance is verified correct.
- `completion_tokens=16` with `last_tool_end_offset=15` matches the loop
  arithmetic exactly: token 0 opens the call, tokens 1–14 are payload, token
  15 closes it, the decoder throws, the loop aborts on the next check.

Correction: **the two reproductions are not the same failure shape.**

- Original OpenCode report (`fcd8f78`, 231s, ~3k tokens, thrown at old
  `ServerInference.swift:365`): the *orphan `<|tool_response>`* guard. The
  model sampled the stop sentinel with zero complete calls — it either never
  opened a call or wrote something else for thousands of tokens. Post-#90
  this would be `kind=orphan_tool_response cause=none`.
- New Hermes repro (`acefaf1`): `kind=decoder_consume cause=malformed` — a
  **closed 14-payload-token block that failed parsing** after only 16
  generated tokens.

Both are "model output the parser refuses," but evidence from one does not
automatically transfer to the other.

## 2. Verdict: underlying root cause — NOT established

The payload bytes between the markers are unknown and nothing checked in can
recover them (the #90 diagnostics deliberately log no content). Assessment of
the brief's seven candidates:

1. **Canonical JSON instead of `call:name{...}`** — most probable, unproven.
   14 tokens fits `{"name": "x", "arguments": {}}` comfortably. Contributing
   prior: `encodeToolChat` hardcodes `enable_thinking: false`
   (`Tokenizer.swift:368-372`), so the template force-closes an empty thought
   channel and the model must emit the call format with zero deliberation
   tokens at a 17k-token prompt.
2. **Another malformed native call** — equally possible; the parser also
   rejects `call :`, leading-zero/`+`-prefixed numbers, trailing text.
3. **Parser rejects what it should accept** — partially confirmed as latent
   bugs, but they cannot be this specific failure:
   - `OpenAIToolName.isValid` (`OpenAIModels.swift:258-269`) accepts hyphens;
     `identifier()` (`GemmaToolCallParser.swift:88`) stops at them. A
     correctly emitted `call:web-search{...}` is undecodable. But that path
     throws `unknownTool`, and the diagnostic says `cause=malformed`, so the
     Hermes failure is not the hyphen bug.
   - `jsonString()` (`GemmaToolCallParser.swift:170-183`) loops through
     `take()`, which calls `skipWhitespace()` first — spaces/tabs/newlines
     inside JSON string values are silently eaten (`"/tmp/a b"` → `"/tmp/ab"`).
     Silent corruption, not a throw.
4. **Detokenization changed the payload** — weak on current `main`: #118
   (`acefaf1`) made decode lossless by construction (`GemmaDecoding`, no
   `clean_up_tokenization_spaces`). PR #107's "cleanup swallows bytes"
   rationale describes the pre-#118 decoder.
5. **Long-context numeric error** — cannot be ruled out; the KV diagnostics
   compare token IDs/counts/positions only, never tensor values.
6. **Nondeterministic runtime** — ruled out for the GPU code (see §4).
7. **The two "identical" attempts were not identical** — favored explanation
   for the hash discrepancy (see §4), and decidable from existing diagnostics.

Signal worth noting: 100% reproducibility *with differing token hashes* means
the failure is robust across trajectory variation — this fits systematic
format drift better than a knife's-edge numeric fault.

The commenter's "14 tokens is nowhere near enough for a real call" is wrong: a
valid minimal call (`call:` + short name + `{}`) fits in fewer tokens. The
count proves only that the closed block was malformed, not why.

### Payload simulation (2026-08-14)

Since the payload bytes are unknown, candidate payloads were simulated with
the real tokenizer (`scratch/gemma4.gturbo/tokenizer/tokenizer.json`) and a
bug-faithful Python port of `GemmaToolCallParser` (both known bugs
reproduced). Script: `docs/issue84_simulate_payloads.py` (needs
`pip install tokenizers`; the port was validated against all predicted
parser behaviors — space-eating corruption, hyphen/dot → `unknown_tool`,
backslash-space → `malformed`, gemma strings parse).

Observed signature to match: exactly 14 payload tokens, `cause=malformed`.
Token-count bands across 9 representative tool names:

| payload shape | verdict | tokens (minimal args) |
| --- | --- | --- |
| canonical JSON `{"name": "N", "arguments": "{}"}` (string-typed args, i.e. the OpenAI wire serialization) | malformed | **exactly 14** for every 2-token name tried |
| canonical JSON `{"name": "N", "arguments": {}}` | malformed | 13 |
| canonical JSON, one short arg | malformed | 15–17 |
| `{"tool": "N", "args": {}}` / fenced ```` ```json ```` block | malformed | 13 |
| native near-misses (`Call:`, missing colon, paren args, single quotes, unquoted value, JSON-quoted keys, trailing text) | malformed | 3–11 |
| valid native `call:N{}` / gemma-string args | ok | 4–12 |

Read: the observed 14-token `malformed` block sits squarely in the
**canonical-JSON band** and is an exact-width match for the OpenAI wire
format with string-encoded empty `arguments` — the precise shape a model
imitating the request-side JSON it was shown would produce. Native-dialect
near-misses are systematically too short unless they carry real arguments.

Caveats: the model's own token split need not equal canonical BPE encoding
(±1–2 tokens), and the real Hermes tool names are unknown; this narrows the
hypothesis space, it does not replace capturing the payload. But it upgrades
hypothesis 1 (canonical-JSON drift) from "most probable" to "strongly
favored", and it demonstrates that essentially *any* JSON-shaped drift
classifies as `malformed` while dialect-prefix drift (`call:` + bad name)
classifies as `unknown_tool` — so the observed `cause=malformed` itself is
evidence the model abandoned the `call:` prefix entirely.

## 3. Confirmed facts, probable explanations, unsupported claims

**Confirmed by code reading:**

- The full immediate mechanism in §1, including the `stop_string` mislabeling.
- Both parser bugs claimed by PR #107 exist on `main` (hyphen identifier,
  whitespace-eating `jsonString`), and its diff for them is correct.
- The server accepts tool names (`-` allowed) that the parser can never
  decode — a validator/parser alphabet mismatch.
- Payload decode on `main` is lossless (`Tokenizer.swift:233-249`,
  `Detokenizer.swift`).
- A failed request invalidates the prompt cache and resets the runner via the
  `defer` in `generate` (`ServerInference.swift:486-491`).

**Probable:** the model drifted out of the native dialect (canonical JSON or
similar) under a 17k prompt with no thinking budget.

**Unsupported:** PR #107's claim that its private repro's drift is this
issue's cause; any inference from KV bookkeeping consistency to numeric
correctness; treating the OpenCode and Hermes failures as one mechanism.

## 4. Temperature-0 hash discrepancy

A full kernel audit found **no run-to-run nondeterminism in the GPU code**:

- Greedy argmax is deterministic: fixed lane→element mappings, `simd_max` +
  lowest-index tie-break at every level (`Metal/Sampling/logit.metal:687-787`,
  same construction in the `sample` kernel's greedy branch).
- MoE expert streaming cannot reorder accumulation: each pread writes a
  pre-assigned slot; the combine (`moe_phase2_down_reduce_k8`,
  `Metal/MoE/moe.metal:446-482`) sums in router-rank order on a single
  thread. Expert-cache hit/miss subset kernels are bit-identical to the full
  kernel.
- No float atomics, no concurrent dispatch, fixed-order split-KV attention
  combine.

But two identical-looking runs can still diverge legitimately:

- **Prefill and decode use different kernel families with different
  precision** (tensor-core MPP path dequantizes weights to `half`,
  `Metal/TensorCore/tensorops.metal:75`; decode GEMV uses FP32 fma). A chunk
  under 32 tokens switches kernels again
  (`RealForwardRunner.swift:646`, dispatch policy `:118-131`). Chunk
  boundaries shift with `cachedPromptTokens`
  (`PrefillRuntimeConfig.swift:95-116`).
- So a **cache-resume legitimately produces different logits than a fresh
  prefill** of the same tokens. The MoE router's discrete top-8 selection
  (`moe.metal:135-185`) amplifies 1-ULP differences into O(1) activation
  changes; at 17k context a near-tie top-2 flips the argmax with no bug.
- At temperature 0.2 the sampler seeds from `CLOCK_MONOTONIC` per token
  unless the client sends `seed` (`Sampler.swift:198-208`) — nondeterministic
  by design.

Likely explanations for the differing temp-0 hashes, in order: different
cache/KV state between the attempts (attempt ordering matters because
failures invalidate the cache), or prompts that were not byte-identical
(agent frameworks often embed timestamps). **Decidable today with zero new
code:** the two failure lines already carry `rendered_prompt_i32le_sha256`
and `cached_prompt_tokens` — ask the commenter to diff those fields across
the two attempts. Identical rendered hashes and cached counts with different
generated hashes would be a genuinely new finding deserving its own issue.

## 5. PR #107 assessment

- **Root-cause claim: unverified.** The drift-to-JSON evidence is a private
  fixture; the public repro has no payload. Plausible, but the PR should not
  merge under the banner "root-cause fix" on current evidence.
- **Fix vs mask:** grammar-constrained decoding is standard, legitimate
  engineering for tool calls, but it is a behavioral guarantee, not a
  root-cause fix — and it would conceal hypothesis 5 (bad logits would
  silently produce well-formed but wrong calls) and mask the drift signal.
  Acceptable as a knowing product decision.
- **Scope: too broad for #84.** `--force-json` CLI mode, `TokenByteTable`,
  and the sampler rework are unrelated to the issue. The two parser fixes are
  correct and necessary regardless — worth extracting into a small standalone
  PR together with a decision on accepting canonical-JSON payloads as a
  fallback.
- **Staleness:** it predates #118; its decoder rationale (library `cleanUp`
  mangling bytes) no longer describes `main`, and its
  `StructuredAssistantDecoder` diff conflicts with the rewritten one.

## 6. Smallest decisive next experiment

The reporter explicitly offered the full request payload — take it. Then on
current `main`:

1. Run the captured request at `temperature 0` with `--prompt-cache-mode off`,
   fresh process per attempt.
2. Capture the bounded region between `<|tool_call>` and `<tool_call|>`. This
   requires the one thing #90 withheld: an opt-in, failure-only, flag-gated
   dump of just the tool-region token IDs (~10 lines).
3. Cheaper first step, zero code: have the reporter rerun twice and diff the
   two diagnostic lines' `rendered_prompt_i32le_sha256` and
   `cached_prompt_tokens` fields.

The payload bytes immediately separate hypotheses 1/2/3; the cache-off
determinism run separates 5/6/7.

## 7. Test and fix plan (deferred — all details for later execution)

State of coverage on `main`: `GemmaToolCallParser` and
`StructuredAssistantDecoder` have **no dedicated unit tests**. Only indirect
references exist, in `Tests/TurboFieldfareServer/OpenAIValidationTests.swift`
and `Tests/TurboFieldfareServer/StructuredOutputDiagnosticsTests.swift`.
There is no `Tests/.../GemmaToolCallParserTests.swift` — PR #107 adds one,
but on its own conflicted branch.

### Phase A — confirmed bugs, writable today, no model, no decisions needed

Create `Tests/TurboFieldfare/Core/Tokenization/GemmaToolCallParserTests.swift`.

**A1. Hyphenated tool names are undecodable.**

- Bug: `OpenAIToolName.isValid`
  (`Sources/TurboFieldfareServer/Core/OpenAIModels.swift:258-269`) accepts
  byte 45 (`-`); `identifier()`
  (`Sources/TurboFieldfare/Tokenization/GemmaToolCallParser.swift:88`)
  accepts only letters/digits/`_` and stops at `-`.
- Failing test: `parse("call:web-search{}", allowedTools: ["web-search"],
  id: "t")` — currently throws `unknownTool("web")`; must return a call
  named `web-search` with empty arguments.
- Fix: extend the `identifier()` alphabet with `-` (PR #107's one-line diff
  for this is correct and can be lifted verbatim).
- Hardening test: property test over the `OpenAIToolName` alphabet
  (`[A-Za-z0-9_-]{1,64}`): every valid name must round-trip through
  `parse("call:<name>{}", allowedTools: [name])`. Pins the validator and
  parser alphabets together permanently.

**A2. `jsonString()` silently eats whitespace inside string values.**

- Bug: the loop at `GemmaToolCallParser.swift:170-183` matches quote and
  backslash via `take()`, which calls `skipWhitespace()` first
  (`:278-283`) — every space/tab/newline inside a `"…"` value is dropped.
- Failing tests:
  - `parse("call:read{path:\"/tmp/a b\tc\"}", ...)` → argument must be
    `/tmp/a b\tc`; currently decodes as `/tmp/abc` (silent corruption, no
    throw).
  - Backslash-space inside a string (`"a\ b"` in payload bytes): currently
    throws `malformed` via `escapedFragment()` because the space after `\`
    is treated as an escape char; after the fix `\` + invalid escape should
    still throw, but a literal space must never be consumed by
    `skipWhitespace`.
- Fix: read characters positionally inside the string loop instead of via
  `take()` (PR #107's diff for `jsonString()` is correct and liftable).
- Note: `gemmaString()` (`:151-168`) reads positionally and does NOT have
  this bug — add a test locking that in (`<|"|>a b<|"|>` keeps its space).

**A3. Jinja `| upper` crash is reachable through validated requests.**

- Mechanism: `GemmaToolSchema.adapt`
  (`Sources/TurboFieldfareServer/Core/GemmaToolSchema.swift`) validates
  `type` only where it recurses (`properties`, `items`); annotation values
  (`default`, `examples`, `title`, `$comment`, …) pass through untouched.
  `validateSchemaKeys` (`OpenAIModels.swift:382-410`) whitelists nothing —
  it only charset-checks property names. The template's
  `filter_keys=true` branch (in `format_parameters`,
  `scratch/gemma4.gturbo/tokenizer/chat_template.jinja`) is taken for an
  object-typed schema with **no** `properties` key and iterates all
  non-standard keys (standard = description/type/properties/required/
  nullable) as if they were property schemas, evaluating
  `value['type'] | upper` on arbitrary annotation values.
- Reproducer schema (passes validation, expected to crash at render):

  ```json
  {"type": "object", "properties": {"cfg": {"type": "object", "default": {}}}}
  ```

- Test location: `Tests/TurboFieldfare/Core/Tokenization/ChatTemplateTests.swift`
  (or the server validation suite if rendering there is easier). First
  assert current behavior (Jinja runtime error, e.g. "upper filter requires
  string") to confirm reachability, then flip the assertion after fixing.
- Fix options: (a) guard the 6 unguarded `value['type'] | upper` template
  sites the commenter identified (they offered their diff); (b) make
  `GemmaToolSchema` strip or whitelist annotation keys so the
  `filter_keys=true` path can never see non-schema values. (b) is safer:
  it also fixes `default: 5` (subscripting an int) and keeps the template
  in sync with upstream.
- This is a separate crash class from #84 (render-time, before generation).

### Phase B — writable today but each encodes a product decision

**B1. Accept canonical JSON tool-call payloads.**

- Test: `parse("{\"name\": \"x\", \"arguments\": {\"a\": 1}}",
  allowedTools: ["x"])` decodes as a call to `x`.
- This is the direct fix for hypothesis 1 (most probable cause of the
  Hermes repro) and the lightweight alternative to PR #107's grammar. Risk:
  if the captured payload turns out to be something else, this fixes a
  different bug than #84. Decide after the payload capture (Phase C), or
  accept both dialects proactively — accepting both is strictly more
  permissive and cannot break existing native-dialect parses.

**B2. Fail-open server behavior (original reporter's actual ask).**

- Behavior change: when `StructuredAssistantDecoder` fails, return the raw
  generated text as assistant `content` with `finish_reason: "stop"`
  instead of HTTP 500, so agent clients can retry/recover.
- Test at `ServerInference.generate` level with a scripted
  `LogitProducer`/backend emitting a malformed tool block; assert 200-path
  completion, cache invalidated (the current `defer` semantics must stay),
  and no partial `tool_calls` array.
- Tension: current design is deliberately fail-closed; #90's plan doc lists
  "returning raw generation as assistant content" as an explicit non-goal.
  This needs a maintainer decision, not just a patch.

**B3. Diagnostics honesty for the abort label.**

- `raw_stop=stop_string` currently conflates a client stop match with the
  server's own decoder-abort (`RawCompletion.swift:221-226` +
  `ServerInference.swift:590-593`). Either add a distinct `StopReason`
  label for the abort, or add a test asserting `stop_string_matched=false`
  always accompanies decoder-abort lines so log readers can distinguish.
  (#90's plan forbade changing `StopReason` — a doc/test-only clarification
  is the minimal version.)

### Phase C — blocked on evidence (the actual #84 regression test)

The regression test for the Hermes `cause=malformed` failure cannot be
written yet: the payload bytes between `<|tool_call>` and `<tool_call|>` are
unknown, so there is no string to assert on. Required sequence:

1. Get the reporter's full request (offered in the issue comment).
2. Zero-code first step: have the reporter run the failing request twice and
   diff `rendered_prompt_i32le_sha256` and `cached_prompt_tokens` across the
   two diagnostic lines (resolves the temp-0 hash puzzle: prompt
   non-identity vs cache-state difference).
3. Add an opt-in, failure-only, flag-gated dump of the tool-region token IDs
   (~10 lines in `ServerInference`; #90 deliberately withheld this, so it
   must be explicit opt-in).
4. Run on current `main`, `temperature 0`, `--prompt-cache-mode off`, fresh
   process per attempt; capture the payload.
5. Turn the captured payload into a fixed unit-test string in
   `GemmaToolCallParserTests` — the true #84 regression test — and only
   then decide between B1 (accept canonical JSON), PR #107's grammar
   constraint, or a numeric investigation (if the payload is garbage bytes
   rather than a recognizable format).

### Recommended PR ordering

1. Phase A1+A2 (pure parser bug fixes + new test file; zero risk, needed
   regardless of #84's root cause).
2. Phase A3 (template/schema guard, separate crash class, cite the
   commenter's report).
3. Phase C instrumentation (opt-in dump flag) + ask the reporter to rerun.
4. Phase B decisions once the payload is in hand; evaluate PR #107 against
   the evidence then (its parser fixes will already be merged via step 1;
   its remaining value is the grammar constraint, which should be judged as
   a product feature, not a bug fix).

Validation commands used by this repo:

```bash
swift build -c release
Scripts/test.sh
ruby Scripts/check_markdown_links.rb
```

## 8. Mistakes in the brief's interpretation

- It treats both reproductions as one failure class; they differ in kind
  (orphan sentinel vs failed closed block) and scale (~3k vs 16 tokens).
- The template observation is understated: the unguarded Jinja `| upper`
  paths are **not** all unreachable. `GemmaToolSchema.adapt` never validates
  annotation values (`default`, `examples`, …), and the template's
  `filter_keys=true` branch — taken for an object-typed parameter with
  annotations but no `properties` key — iterates those annotation values as
  property schemas and hits unguarded `value['type'] | upper`. A validated
  request like `{"type":"object","default":{}}` as a parameter plausibly
  reaches the crash the commenter couldn't isolate. One unit test would
  confirm.
- "Deterministic at temp 0" must not be assumed even absent bugs:
  cache-resume vs fresh prefill is a numeric fork by design, and temp 0.2 is
  clock-seeded by design.
