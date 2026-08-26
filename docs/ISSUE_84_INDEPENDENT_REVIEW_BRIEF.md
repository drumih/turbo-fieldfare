# Independent Review Request: TurboFieldfare Issue #84

Perform a read-only, evidence-based investigation. Do not implement or edit
anything. Separate the immediate failure mechanism from the underlying cause,
and independently verify every conclusion below.

## Primary Sources

- Issue: <https://github.com/drumih/turbo-fieldfare/issues/84>
- New reproduction:
  <https://github.com/drumih/turbo-fieldfare/issues/84#issuecomment-5284001871>
- Diagnostic PR already merged:
  <https://github.com/drumih/turbo-fieldfare/pull/90>
- Proposed grammar-constrained fix:
  <https://github.com/drumih/turbo-fieldfare/pull/107>

Repository:

```text
/Users/andreymikhaylov/development/turbo-fieldfare
```

Refresh the current GitHub and `origin/main` state before drawing conclusions.

## Original Report

An OpenCode client successfully completed two tool-calling turns, then failed
on the third:

```text
prepared prompt=23803 -> completed in 46.575s cached=16742 completion=57 finish=tool_calls
prepared prompt=25658 -> completed in 14.396s cached=23875 completion=57 finish=tool_calls
prepared prompt=27513 -> failed after 231s phase=generating status=500
error=TurboFieldfare.GemmaToolCallParserError.malformed
```

Setup:

- Commit `fcd8f78`
- `--max-context 65536`
- Apple Silicon, 48 GB
- Tool definitions sent on each turn

The original theory was drift or truncation after a very long generation, but
raw generated output was unavailable.

## New Independent Reproduction

The linked comment reports:

- Commit `acefaf1`
- Hermes Agent using `/v1/chat/completions`
- 26 tools
- Two-message conversation
- 17,355 rendered prompt tokens
- Failure is reportedly 100% reproducible at this prompt size
- Same result at temperature `0.2` and greedy temperature `0`

Diagnostics:

```text
completion_tokens=16 max_completion_tokens=48181 raw_stop=stop_string
tool_start_count=1 tool_end_count=1 decoded_calls=0
last_tool_start_offset=0 last_tool_end_offset=15
effective_count_matches_result=true effective_prefix_matches_kv=true
kv_position_matches_history=true completion_count_matches_history=true
prefill_accounting_matches=true
```

The commenter also reports different generated-token hashes across two
ostensibly identical temperature-0 attempts.

## Current Interpretation to Verify

The immediate failure mechanism appears clear:

1. Generated token 0 is `<|tool_call>`.
2. Tokens 1-14 form the tool-call payload.
3. Token 15 is `<tool_call|>`.
4. When the closing marker arrives, `StructuredAssistantDecoder` decodes the
   collected payload.
5. `GemmaToolCallParser` expects:

   ```text
   call:<allowed-tool-name>{...}
   ```

6. The payload fails that parser, producing
   `decoder_consume cause=malformed`.
7. The server stops generation and returns HTTP 500.

Relevant files:

- `Sources/TurboFieldfareServer/Core/ServerInference.swift`
- `Sources/TurboFieldfare/Tokenization/StructuredAssistantDecoder.swift`
- `Sources/TurboFieldfare/Tokenization/GemmaToolCallParser.swift`
- `Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift`
- `Sources/TurboFieldfareServer/Core/GemmaToolSchema.swift`

Important nuance: `raw_stop=stop_string` probably does not mean a
client-supplied stop string matched. `ServerInference` sets `shouldStop = true`
after the decoder throws, while `RawCompletion` combines
`stopMatcher.isStopped || shouldStop` and records either condition as
`.stopString`. Verify this carefully.

## What the Diagnostics Seem to Establish

- This reproduction is not an unfinished tool block or maximum-token
  truncation.
- The model emitted both opening and closing tool markers.
- Parsing failed while consuming the closing marker.
- The failure is not merely an orphan `<|tool_response>` marker.
- It reproduces under greedy generation, so ordinary sampling luck is
  unlikely.
- The server's token-history, prefix-cache, position, and prefill-count
  bookkeeping is internally consistent.

Do not overinterpret the KV checks. They compare token IDs, counts, and
positions. They do not validate the numerical K/V tensors or the correctness
of long-context attention or prefill computation.

Also challenge the commenter's claim that 14 payload tokens are "nowhere near
enough" for a valid call. A short tool name with an empty or small argument
object may fit. The token count proves that the closed block was malformed,
not why it was malformed.

## Main Unresolved Question

What were the actual payload bytes or token IDs between the two tool markers?

Without that evidence, distinguish among:

1. The model emitted canonical JSON instead of Gemma's native
   `call:name{...}` dialect.
2. The model emitted another malformed native-dialect call.
3. The parser rejects a representation that should reasonably be accepted.
4. Detokenization changed the payload before parsing.
5. Long-context prefill or attention produced incorrect logits despite
   consistent bookkeeping.
6. A nondeterministic runtime issue explains the differing temperature-0
   hashes.
7. The two "identical" greedy attempts were not actually identical in prompt
   IDs, cache state, runtime configuration, or process state.

## PR #107

PR #107 claims its private reproduction showed the model drifting to canonical
JSON inside `<|tool_call>`. It adds grammar-constrained decoding to prevent
invalid tool payloads.

Treat that claim as a hypothesis requiring independent verification because:

- The public issue comment does not include the raw payload.
- The end-to-end fixture used by the PR author is private and not checked in.
- The PR is broad, adding general forced-JSON support as well as the
  issue-specific tool grammar.
- At the previous inspection it was open, conflicting, and had no reported
  GitHub checks; refresh this state.

Assess whether PR #107:

- fixes the demonstrated cause or merely masks malformed generation;
- is necessary compared with accepting both native and canonical
  representations;
- could conceal a long-context numerical correctness problem;
- preserves prompt-cache and fail-closed semantics;
- handles incomplete calls, UTF-8, byte limits, unknown tools, and completion
  budgets correctly; and
- is appropriately scoped for issue #84.

## Template Observation

The commenter noticed several unguarded `value['type'] | upper` expressions in
`chat_template.jinja`.

The current server's `GemmaToolSchema` adapter appears to reject unsupported
types and normalize nullable type arrays to one concrete string type before
invoking the template. Verify whether every unguarded Jinja path is therefore
unreachable through validated OpenAI requests.

Treat this as separate from issue #84 unless a concrete request reaches the
Jinja error.

## Requested Output

Return:

1. A direct verdict: is the immediate failure understood?
2. A separate verdict: is the underlying root cause established?
3. Confirmed facts, probable explanations, and unsupported claims.
4. An explanation of the temperature-0 hash discrepancy.
5. An assessment of PR #107's root-cause claim and scope.
6. The smallest decisive next experiment.
7. Any mistakes in the interpretation above.

The likely decisive experiment is to obtain the reporter's full request or a
minimized reproducer, run current `main` at temperature 0 with prompt caching
disabled, and capture the exact bounded payload between `<|tool_call>` and
`<tool_call|>`. Do not expose private prompt text in public logs.

No source files should be changed during this review.
