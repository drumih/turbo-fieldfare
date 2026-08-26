# Issue 84: Root-Cause Test Plan

## Purpose

Determine why issue 84 produces a closed `<|tool_call>...<tool_call|>` block
that `GemmaToolCallParser` rejects. The immediate failure path is already
understood; this plan isolates the source of the malformed payload.

Treat the two known reports as separate cases until evidence shows they share a
cause:

- **Case O — original OpenCode flow:** two successful tool turns followed by a
  long third generation and HTTP 500.
- **Case H — Hermes reproduction:** a two-message, 26-tool request that closes a
  malformed tool block after 16 generated tokens.

This plan complements:

- [Independent review brief](ISSUE_84_INDEPENDENT_REVIEW_BRIEF.md)
- [Structured tool-generation diagnostics](ISSUE_84_STRUCTURED_TOOL_DIAGNOSTICS_PLAN.md)
- [Local server guide](OPENAI_SERVER.md)
- [Runtime controls](RUNTIME_CONTROLS.md)

## Root-cause standard

Do not call a hypothesis the root cause merely because one run succeeds after
changing a setting. A root cause is established only when all of the following
are true:

1. The exact request, rendered prompt token IDs, generation settings, runtime
   settings, model revision, and process state are known.
2. The exact generated token IDs and bytes inside the rejected tool block are
   captured privately.
3. One controlled change predicts the first output or logit divergence.
4. An A-B-A check removes the failure and then restores it, or an independent
   oracle reproduces the same first divergence.
5. The explanation accounts for the entire failing path without contradicting
   successful tool turns.

Case O and Case H may finish with different root causes.

## Safety and privacy rules

- Follow the repository model-process, OS, Swift, disk, memory-pressure, and
  completed-model checks before every model session.
- Run only one TurboFieldfare, MLX-LM, app, CLI, server, or model-backed test at
  a time.
- Keep the server on `127.0.0.1`; do not proxy, tunnel, or expose it.
- Do not download another checkpoint, duplicate the `.gturbo` model, purge
  caches, or create a worktree for this investigation.
- Keep full requests, raw generations, tool schemas, tool results, and token
  sequences outside the repository. They may contain private or reversible
  information.
- Public logs and issue updates may contain bounded counts and hashes, but not
  prompt text, generated text, tool names, arguments, results, or token arrays.
- Do not alter the installed `.gturbo` sidecars. Render template variants
  out-of-process and record the template hash.
- A temporary local probe or oracle harness requires separate authorization.
  Keep it uncommitted until its output proves that a durable test is needed.

## Required inputs

Obtain these separately for Case O and Case H before running a comparison:

- complete request JSON for every turn, including tool order and full schemas;
- original server command and commit;
- whether the request was streaming;
- client name and version;
- generation options, especially temperature, maximum completion tokens, stop
  strings, seed, Top-K, and Top-P;
- server runtime settings and prompt-cache mode;
- complete request-correlated diagnostic line;
- for Case O, all messages returned by the server and tool results appended by
  the client during the two successful turns.

Store the private request unchanged. Generate experimental variants from it so
the original remains a byte-for-byte reference.

## Evidence record

Create one row per run in a private results table:

| Field | Required value |
| --- | --- |
| Run ID | Stable case and variant label |
| Code | `git rev-parse HEAD` and dirty-state summary |
| Host | Hardware, RAM, macOS, Swift version |
| Model | `.gturbo` manifest/revision and tokenizer/template hashes |
| Request | Private request SHA-256 and turn number |
| Prompt | Rendered/effective token counts and token hashes |
| Runtime | Full server command and process freshness |
| Generation | Temperature, limits, stop options, Top-K, Top-P, seed |
| Result | Exit/HTTP status, timing footer, finish reason |
| Structure | Marker counts/offsets, decoder phase, parser cause |
| Output | Private generated token IDs and exact tool-payload bytes |

If rendered or effective prompt hashes differ between supposedly identical
runs, stop. Fix the harness or explain the changed input before comparing
generated output.

## Phase 1: Capture the decisive artifact

### 1.1 Reproduce on current `main`

Build release once. Start with:

- temperature `0`;
- prompt cache off;
- prefill on with 128-token chunks;
- 16 expert-cache slots, LFU;
- RDADVISE off;
- the reporter's context and completion limits;
- a fresh server process.

Submit the exact request three times. Do not run unrelated warmups. For Case O,
replay the complete three-turn flow; do not submit only its final request unless
that final request is independently shown to reproduce with cache off.

The baseline is usable when either:

- all three runs fail with the same generated token hash and parser cause; or
- the inputs match but output hashes differ, proving a deterministic-runtime
  investigation is required.

### 1.2 Capture raw token evidence privately

Capture, before structured parsing:

- all generated token IDs;
- exact token IDs between the final tool-start and tool-end markers;
- the exact bytes produced from those IDs with special-token skipping disabled;
- incremental decoder deltas around both markers;
- final stop reason and boundary token IDs;
- first parser error and allowed-tool set hash.

Prefer a debugger breakpoint at the `StructuredAssistantDecoder` call to
`GemmaToolCallParser.parse`. If release optimization prevents inspection, use
the smallest failure-only local probe that writes to a permission-restricted
private file. Do not add raw output to normal server logging.

### 1.3 Classify the payload

| Captured payload | Next branch |
| --- | --- |
| Valid native `call:name{...}` rejected | Parser or allowed-tool validation |
| Canonical JSON object | Template/model dialect branch |
| Native prefix with malformed/truncated body | Template, model, or numerical branch |
| Token IDs reconstruct differently by decoding path | Tokenizer/detokenizer branch |
| Payload differs with identical prompt IDs and greedy settings | Runtime nondeterminism branch |

Do not proceed to a broad runtime matrix until this classification exists.

## Phase 2: Minimize each reproducible case

Minimize one dimension at a time while requiring the same parser cause and
payload shape:

1. Binary-search prior conversation turns.
2. Binary-search the ordered tool list.
3. Remove unused schema properties and descriptions.
4. Reduce tool results and ordinary message content.
5. Check whether tool ordering, one tool name, one schema construct, or one
   historical assistant/tool turn is necessary.

After every reduction, run A-B-A: failing original, candidate reduction,
failing original. Stop minimizing when another reduction changes the failure
class or removes the first divergent token.

Useful deliverables are:

- the smallest private reproducer preserving the original failure;
- a sanitized public reproducer if its tokenization and behavior are identical;
- an explicit list of request features that are necessary and unnecessary.

## Phase 3: Template and protocol comparison

This phase is especially important for Case O because it fails after successful
tool turns.

### 3.1 Freeze both templates

Compare:

- the pinned template embedded in the installed model; and
- a hash-pinned copy of Google's current canonical Gemma 4 template.

Record both SHA-256 hashes and a focused diff covering assistant tool calls,
tool responses, turn closures, reasoning/history reinjection, and null schema
handling. Relevant upstream references:

- <https://huggingface.co/google/gemma-4-12B-it/blob/main/chat_template.jinja>
- <https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/40>
- <https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/47>

Discussion reports are hypotheses, not proof for issue 84.

### 3.2 Run the template A-B-A

Render the same messages and tools out-of-process with each template. Preserve
the resulting prompt token IDs as private artifacts. Run:

1. pinned template;
2. current canonical template;
3. pinned template again.

Keep the model, runtime, and generation settings fixed. Do not modify the
installed `.gturbo` directory. If TurboFieldfare cannot accept explicit prompt
IDs through an existing validation surface, build the smallest non-production
harness only after authorization.

Interpretation:

- Only the canonical template succeeds: prompt framing is causal.
- Both templates generate the same malformed dialect: template revision is not
  sufficient; compare against the model oracle.
- Only a historical tool turn is required: inspect its exact rendered closure
  and the next model-turn opening.
- The template changes the output but neither result is valid: continue with
  first-divergence oracle comparison rather than choosing the nicer output.

### 3.3 Protocol parser comparison

Feed the captured payload bytes, without model execution, to:

- `GemmaToolCallParser`;
- a minimal parser implementing the pinned template's native syntax; and
- MLX-LM's Gemma tool parser where applicable.

MLX-LM documents native Gemma tool-call support here:
<https://github.com/ml-explore/mlx-lm/issues/1096>.

Agreement between parsers establishes payload validity, not model correctness.

## Phase 4: Small runtime differential matrix

Run this phase only if the payload is malformed under the pinned template and
the cause remains numerical, cache-related, or nondeterministic. Keep exact
request bytes and generation settings fixed.

Start from baseline `B0`:

```text
max-context=65536
prompt-cache-mode=off
prefill=on
prefill-chunk-tokens=128
expert-cache-slots=16
expert-cache-policy=lfu
rdadvise=off
temperature=0
```

Run only the first-level variants initially:

| ID | Single change from B0 | Question answered |
| --- | --- | --- |
| B1 | `--prefill off` | Does chunked prefill change the first output token? |
| B2 | `--max-context 32768` | Is the failure dependent on the 64K allocation/path? |
| B3 | `--expert-cache-slots 32` | Does expert residency alter greedy output? |
| O1 | `--prompt-cache-mode single-prefix` | Does verified KV reuse change Case O? |

Use B2 only when the full prompt plus fixed completion limit fits 32K. O1 must
replay the real multi-turn sequence and is not useful as a substitute for raw
payload capture in Case H.

Escalate only the axis that changes the first divergent token:

- If B1 differs, test chunk sizes 32, 64, and 128.
- If B2 differs, compare the first logits at identical prompt positions and
  inspect position/RoPE and attention-length handling.
- If B3 differs, compare LFU and LRU, then expert loads and routed expert IDs.
- If O1 differs, compare each turn's rendered/effective prompt hashes, cached
  token count, KV position, and first post-prefix logits.
- If identical inputs produce different greedy hashes, run three fresh-process
  repetitions and three same-process repetitions to distinguish initialization
  from mutable process state.

The bounded diagnostics prove token lineage and accounting only. They do not
prove numerical K/V tensor equality.

## Phase 5: Independent MLX-LM oracle

Use the exact pinned checkpoint named in
[Implementation references](IMPLEMENTATION_REFERENCES.md). The checkpoint is
already present on the current investigation host; do not download or copy it.
At planning time, the active shell did not expose an `mlx_lm` runtime, so locate
an existing compatible environment or obtain authorization before installing
anything.

Run TurboFieldfare and MLX-LM sequentially with:

- identical explicit prompt token IDs;
- the same pinned weights, tokenizer, and template sidecars;
- greedy decoding;
- the same maximum generation count;
- cache reuse disabled;
- generated token IDs retained privately.

Compare at every generated position:

- chosen token ID;
- top-k token IDs and logits;
- top-1/top-2 margin;
- first position at which ranking or token choice differs.

Interpretation:

- Same malformed payload from both engines: model/template behavior, not a
  TurboFieldfare-only numerical defect.
- MLX emits a valid call and TurboFieldfare diverges with a large logit margin:
  investigate TurboFieldfare model math at the first divergent position.
- Different token choice with a near-zero margin: repeat and compare logits;
  token equality alone is too strict for floating-point implementations.
- Logits agree through the payload but parsing differs: parser or decoding
  defect.

Do not compare free-form text alone. The oracle is useful only with exact prompt
IDs and token/logit evidence.

## Phase 6: Focused hard tests

Add durable tests only after the captured payload or first divergence identifies
the relevant boundary.

### Parser and tokenizer branch

- Round-trip native calls through template rendering, tokenization,
  token-by-token structured decoding, and parsing.
- Preserve a sanitized minimal failing payload as a regression fixture.
- Test exact token-byte reconstruction separately from ordinary decoded text.
- Cover only observed edge classes: marker boundaries, whitespace, Unicode,
  quoted or native object keys, nested values, and allowed tool identifiers.

### Runtime branch

- Add the smallest scalar/CPU or bounded-logit comparison that fails at the
  first divergent layer or position.
- Avoid a full model-backed package test when an existing local reference test
  can express the failing invariant.
- Require the test to fail before the fix and pass after it.

Do not begin broad parser fuzzing, long soak runs, or performance profiling
without evidence that the corresponding subsystem is involved.

## Phase 7: Evaluate fixes and PR #107

Evaluate a proposed change only after the root-cause branch is known:

- If the parser rejects a valid supported representation, fix and test the
  parser at that representation.
- If the pinned template frames multi-turn tool history incorrectly, update the
  pinned template with exact before/after render fixtures.
- If TurboFieldfare logits diverge from the oracle, fix the earliest numerical
  defect before adding grammar constraints.
- If both engines naturally emit malformed syntax, grammar-constrained decoding
  is a mitigation. Report it as such unless the experiment also proves why the
  unconstrained model changed dialect.

PR #107 must still be checked for incomplete calls, UTF-8, size limits, unknown
tools, completion budgets, prompt-cache semantics, and fail-closed behavior. A
successful grammar-constrained run does not by itself establish root cause.

## Execution order and stop conditions

Run the investigation in this order:

1. Acquire exact Case O and Case H requests.
2. Reproduce and capture private payload token IDs and bytes.
3. Classify and minimize each case.
4. Run the template A-B-A.
5. Run only the relevant first-level runtime variants.
6. Use MLX-LM at the first unresolved model-output boundary.
7. Add one focused regression test and evaluate the smallest fix.

Stop early when an A-B-A result plus raw evidence establishes the root cause.
Do not complete the remaining matrix merely for coverage.

Stop and report a blocker when:

- the full request cannot be obtained and the public reproduction cannot be
  recreated;
- preflight checks fail;
- supposedly identical runs have different prompt hashes;
- the private payload cannot be captured safely;
- the oracle would require another checkpoint download;
- a test requires simultaneous model processes.

## Final report

Create `docs/ISSUE_84_ROOT_CAUSE_RESULTS.md` only after experiments begin. It
should contain:

1. commit, host, model revision, and exact commands;
2. protocol deviations and unavailable artifacts;
3. one compact run table;
4. the captured payload classification without private content;
5. first-divergence evidence;
6. separate conclusions for Case O and Case H;
7. confirmed root cause, probable contributors, and rejected hypotheses;
8. fix recommendation and assessment of PR #107;
9. remaining uncertainty.

Do not publish private prompts, payloads, tool schemas, results, or reversible
token sequences in that report.
