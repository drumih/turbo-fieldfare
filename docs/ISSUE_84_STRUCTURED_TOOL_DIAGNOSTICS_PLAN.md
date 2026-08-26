# Issue 84: Structured Tool-Generation Diagnostics

## Goal

Make failures like issue 84 diagnosable from one server log line, without
logging prompts, generated text, tool arguments, or reversible token
sequences.

This change diagnoses the failure. It does not attempt to fix model behavior,
cache behavior, retry policy, or response fallback.

## Confirmed behavior

The implementation must be based on these established facts:

1. `runRawCompletion` sets `.toolCalls` only when the model samples
   `tokenizer.toolResponseID`.
2. That stop token is intercepted before
   `StructuredAssistantDecoder.consume` receives it.
3. The guard in `ServerInference.swift` runs only after no decoder error was
   captured, `decoder.finish()` succeeded, and zero complete calls were
   decoded.
4. Therefore that guard means specifically that the model emitted
   `<|tool_response>` without a preceding complete valid tool call.
5. `RawDecodeResult` already retains everything required for diagnostics:
   the effective prompt and committed generated tokens in
   `kvBackedTokenIDs`, the final sampled stop or failing token in
   `uncommittedBoundaryTokenIDs`, and prompt, cache, completion, stop-reason,
   and KV-position fields.

No decode-loop collector is needed.

## Scope

Modify only:

- `Sources/TurboFieldfareServer/Core/ServerInference.swift`
- `Tests/TurboFieldfareServer/OpenAIValidationTests.swift`, or one small
  dedicated diagnostics test file if that is materially clearer

Do not modify:

- `RawCompletion.swift`
- `StructuredAssistantDecoder.swift`
- `GemmaToolCallParser.swift`
- `HTTPServer.swift`
- `ServerLog.swift`
- server arguments or environment handling
- prompt-cache logic
- documentation beyond this implementation plan

The existing `ServerLog.failed` path already provides request-ID correlation
and logs `String(reflecting: error)`. Carry the diagnostic payload in the
thrown error and reuse that path.

## 1. Add a dedicated structured-generation error

Add internal types near the server inference types in `ServerInference.swift`.

Suggested shape:

```swift
enum StructuredOutputFailureKind: String, Sendable {
    case decoderConsume = "decoder_consume"
    case decoderFinish = "decoder_finish"
    case orphanToolResponse = "orphan_tool_response"
}

struct StructuredOutputFailureDiagnostics: Equatable, Sendable {
    // Scalar fields and hashes.
}

struct StructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let kind: StructuredOutputFailureKind
    let cause: String
    let diagnostics: StructuredOutputFailureDiagnostics

    var debugDescription: String {
        // Stable, bounded, single-line key=value representation.
    }
}
```

Keep these types internal. They are server implementation details, not API
types.

`CustomDebugStringConvertible` is important because `ServerLog.failed` uses
`String(reflecting:)`. Add a test proving that reflection produces the
intended bounded log text.

### Cause classification

Do not store or reflect an arbitrary underlying error. Classify known parser
errors:

| Existing error | Diagnostic cause |
| --- | --- |
| `.malformed` | `malformed` |
| `.unknownTool` | `unknown_tool` |
| `.oversized` | `oversized` |
| Anything unexpected | `unexpected` |
| Orphan-sentinel guard | `none` |

Do not include the generated unknown tool name. Although request tool names
are validated, a model-generated identifier could otherwise make the log
unnecessarily large or expose internal names.

## 2. Wrap all post-generation structured-output failures

There are three distinct failure points after `runRawCompletion` returns.

### A. Decoder failure during token consumption

Replace:

```swift
if let decodingError { throw decodingError }
```

with a `StructuredOutputFailure` using:

- `kind = .decoderConsume`
- the classified parser cause
- diagnostics built from the completed `RawDecodeResult`

This covers malformed arguments, invalid framing, duplicate starts, unmatched
ends, unknown tools, and oversized calls detected during streaming.

Do not throw directly from the progress callback. The existing callback
behavior captures the error, requests generation stop, and throws after
`runRawCompletion` returns. That is what makes the final token evidence
available.

### B. Decoder failure at `finish()`

Wrap:

```swift
try decoder?.finish()
```

in `do/catch` and throw:

- `kind = .decoderFinish`
- the classified parser cause
- the same diagnostics

This primarily identifies an unfinished `<|tool_call>` block at EOS,
end-of-turn, stop-string, or max-token termination.

### C. Orphan tool-response sentinel

Replace the current `GemmaToolCallParserError.malformed` throw with:

- `kind = .orphanToolResponse`
- `cause = "none"`
- the same diagnostics

This corrects the misleading classification without weakening fail-closed
behavior.

### Preserve failure semantics

The wrapper must remain an ordinary non-`ServerRequestError`, so HTTP behavior
remains unchanged:

- non-streaming requests return HTTP 500 with the generic internal-error
  envelope;
- streaming requests emit the existing SSE error and `[DONE]`;
- the prompt cache is invalidated;
- the runner is reset;
- no partial content is converted into a successful response.

## 3. Reconstruct the exact generated token sequence

Add one private or internal helper in `ServerInference.swift`.

The generated sequence is:

```swift
let committedGenerated = result.kvBackedTokenIDs.dropFirst(result.prefillTokens)
let generatedIDs =
    Array(committedGenerated) + result.uncommittedBoundaryTokenIDs
```

Why this is correct:

- `kvBackedTokenIDs` begins with the full effective prompt.
- Nonterminal generated tokens are appended after being accepted into KV.
- The terminal or failing token is retained in
  `uncommittedBoundaryTokenIDs`.
- Therefore the concatenation includes the orphan `<|tool_response>` token
  and parser-failing boundary tokens.

Make reconstruction defensive: diagnostics must never crash while reporting
another failure. If `prefillTokens > kvBackedTokenIDs.count`, use a safe
bounded drop and report failed lineage invariants.

Do not add a generated-token property to `RawDecodeResult`; reconstruction is
only needed on this server failure path.

## 4. Diagnostic payload

Emit a stable, single-line, failure-only summary.

### Request and cache fields

- `rendered_prompt_tokens`: `promptIDs.count`
- `effective_prompt_tokens`: `effectivePromptIDs.count`
- `result_prompt_tokens`: `result.prefillTokens`
- `cached_prompt_tokens`: `result.cachedPromptTokens`
- `computed_prefill_tokens`: `result.computedPrefillTokens`
- `completion_tokens`: `result.newTokens`
- `max_completion_tokens`: final `config.maxNewTokens`
- `raw_stop`: explicit mapping to `eos`, `end_of_turn`, `max_tokens`,
  `stop_string`, or `tool_calls`
- `kv_position`: `result.kvPosition`
- `kv_backed_tokens`: `result.kvBackedTokenIDs.count`
- `boundary_tokens`: `result.uncommittedBoundaryTokenIDs.count`

### Structured decoder fields

- `decoded_calls`: `calls.count`
- `visible_bytes`: `content.utf8.count`
- `stop_string_matched`: `stopMatcher.isStopped`

Use UTF-8 bytes rather than Swift character count.

### Tool-marker fields

Scan only the reconstructed generated sequence:

- `tool_start_count`
- `tool_end_count`
- `tool_response_count`
- `tool_response_end_count`
- `last_tool_start_offset`
- `last_tool_end_offset`
- `last_tool_response_offset`
- `last_tool_response_end_offset`

Offsets are zero-based within the generated sequence. Use `-1` when absent.

Do not log arbitrary special-token IDs or a complete special-token trace.
Counts and final offsets are sufficient to distinguish:

- no call framing followed by a tool response;
- an opened but unfinished call;
- an unmatched close;
- one valid call followed by malformed additional framing;
- repeated response markers.

### Lineage invariants

Report separate booleans so a false result is actionable:

- `effective_count_matches_result`:
  `effectivePromptIDs.count == result.prefillTokens`
- `effective_prefix_matches_kv`: enough KV-backed tokens exist and their
  prompt prefix exactly equals `effectivePromptIDs`
- `kv_position_matches_history`:
  `result.kvPosition == result.kvBackedTokenIDs.count`
- `completion_count_matches_history`: reconstructed generated count equals
  `result.newTokens`
- `prefill_accounting_matches`: cached plus computed prefill equals total
  prefill

These checks are failure-only and bounded by the configured context size.

## 5. Token hashes

Compute three deterministic SHA-256 fingerprints:

- `rendered_prompt_i32le_sha256`
- `effective_prompt_i32le_sha256`
- `generated_i32le_sha256`

Serialization must be defined precisely:

1. Treat each token as its `UInt32(bitPattern:)`.
2. Serialize four bytes in little-endian order.
3. Concatenate without delimiters.
4. SHA-256 the resulting bytes.
5. Emit lowercase hexadecimal.

`CryptoKit` is already imported by `ServerInference.swift`; do not add a
dependency or a general hashing utility.

These hashes allow later cache-on/cache-off comparison without sharing
content. They are fingerprints, not anonymization; do not describe them as
anonymous or secret-safe against dictionary attacks.

Hashing happens only after a structured generation fails, so it does not
affect successful generation performance.

## 6. Log format

Keep one line and follow existing server logging conventions.

Expected shape:

```text
error=structured_output_failure kind=orphan_tool_response cause=none rendered_prompt_tokens=27513 effective_prompt_tokens=... cached_prompt_tokens=... computed_prefill_tokens=... completion_tokens=... max_completion_tokens=... raw_stop=tool_calls kv_position=... kv_backed_tokens=... boundary_tokens=1 decoded_calls=0 visible_bytes=0 stop_string_matched=false tool_start_count=0 tool_end_count=0 tool_response_count=1 tool_response_end_count=0 last_tool_start_offset=-1 last_tool_end_offset=-1 last_tool_response_offset=... last_tool_response_end_offset=-1 effective_count_matches_result=true effective_prefix_matches_kv=true kv_position_matches_history=true completion_count_matches_history=true prefill_accounting_matches=true rendered_prompt_i32le_sha256=... effective_prompt_i32le_sha256=... generated_i32le_sha256=...
```

Requirements:

- no newlines;
- stable field order;
- no arrays;
- no request text;
- no decoded generation;
- no tool schemas, names, arguments, or results;
- no arbitrary error descriptions;
- bounded length independent of prompt or completion size.

Do not change `ServerLog.failed`; its existing prefix supplies the timestamp,
request ID, HTTP status, and request phase.

## 7. Focused test

Add one synthetic diagnostics test using `RawDecodeResult`; no model is
required.

Suggested scenario:

1. Build a small effective prompt.
2. Put two ordinary generated tokens in `kvBackedTokenIDs`.
3. Put `tokenizer.toolResponseID` in
   `uncommittedBoundaryTokenIDs`.
4. Set `reason = .toolCalls`, `newTokens = 3`, nonzero cached prompt tokens,
   and consistent KV position and accounting.
5. Construct `.orphanToolResponse` diagnostics.
6. Assert that:
   - reconstructed completion includes the boundary token;
   - completion count is correct;
   - tool-response count is one;
   - tool start and end counts are zero;
   - response offset is the final generated offset;
   - all lineage invariants are true;
   - hash output is lowercase 64-character hex;
   - `String(reflecting: error)` contains the required fields;
   - the reflected error contains no decoded prompt or content text and no
     complete token arrays.

Include a fixed hash vector for a tiny known `[Int32]` sequence so the `i32le`
serialization convention cannot silently change.

Do not create model-backed server tests or mock `ServerModelSession`; that
would require unnecessary dependency injection for a pure diagnostic
transformation.

## 8. Validation

Mandatory checks:

```bash
Scripts/test.sh --filter 'TurboFieldfareServerTests\.(StructuredOutputDiagnosticsTests|GemmaToolCallTests|ServerPromptCacheTests)'
swift build -c release --product TurboFieldfareServer
git diff --check
```

If the new test is added to an existing suite, adjust the filter accordingly.

Also review the final diff for these properties:

- only failure paths changed;
- successful completion logging is unaffected;
- no backend protocol signatures changed;
- no new CLI or environment controls exist;
- no prompt-cache behavior changed;
- HTTP and SSE envelopes remain unchanged;
- the existing `defer` still invalidates the cache and resets the runner on
  every wrapped failure.

No model run is required to validate this implementation.

## 9. Reporter follow-up

After the diagnostic patch is available, ask the issue 84 reporter to rerun
the same three-turn OpenCode flow and provide:

- the complete single failure log line;
- server commit;
- whether the request was streaming;
- the unchanged launch command.

Do not initially ask for the raw prompt or generated text.

Interpret the result as follows:

| Evidence | Likely meaning |
| --- | --- |
| `tool_start_count=0`, `tool_end_count=0`, `tool_response_count=1` | Confirmed orphan response sentinel |
| Start count greater than end count and `kind=decoder_finish` | Unfinished tool call |
| `kind=decoder_consume` with matching start and end counts | Malformed call body or invalid tool |
| Any false lineage invariant | Runtime or result-accounting defect; investigate before blaming model or cache |
| Cached tokens greater than zero | Cache path participated; this does not prove it caused divergence |
| Cache-off and cache-on produce identical generated hashes | Behavior is not caused by cache reuse |
| Only cache-on reproduces under deterministic generation | Investigate long-context continuation and KV parity |

If cache causation remains plausible, perform a separate controlled
reproduction:

1. Capture the exact sanitized request sequence.
2. Force deterministic generation with temperature `0`.
3. Run sequentially with `--prompt-cache-mode off`.
4. Run sequentially with `--prompt-cache-mode single-prefix`.
5. Follow all repository model-process and memory checks.
6. Compare counts, prompt fingerprints, generated fingerprints, and marker
   positions.

Do not run two model processes simultaneously.

## Explicit non-goals

Do not expand this work into:

- returning raw generation as assistant content;
- retrying automatically;
- exposing hidden thought output;
- adding `--debug`, `--verbose`, or raw-dump flags;
- logging complete token IDs;
- changing `StopReason`;
- changing the decoder grammar;
- changing prompt-cache matching;
- adding a general logging framework;
- claiming the root cause is model drift or KV divergence before comparative
  evidence exists.

A raw-generation dump can be considered later only if the scalar diagnostics
reproduce but cannot distinguish the cause. It must be a separate, explicitly
sensitive opt-in change.

## Acceptance criteria

The work is complete when:

1. The former orphan-sentinel guard throws `orphan_tool_response`, not
   `GemmaToolCallParserError.malformed`.
2. All three structured-output failure phases carry the same bounded
   diagnostic schema.
3. The existing HTTP logger produces one request-correlated line with actual
   completion and cache counts.
4. The terminal boundary token is included in marker counts and the generated
   hash.
5. No raw user, model, or tool content is logged.
6. Existing failure, cache invalidation, runner reset, HTTP 500, and SSE
   behavior remain unchanged.
7. Focused tests and the release server build pass.

Suggested single commit:

```text
Diagnose malformed server tool generations
```
