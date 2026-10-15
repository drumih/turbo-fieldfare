# Local OpenAI-compatible server

`TurboFieldfareServer` exposes a local Chat Completions API for one Gemma
model. It binds to `127.0.0.1` without authentication or TLS. Do not expose it
through a proxy or tunnel.

## Start the server

First, install the model with the Mac app or `TurboFieldfareRepack`. Then check
that no other TurboFieldfare model process is running:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If the command prints a match, do not start the server.

```bash
swift build -c release --product TurboFieldfareServer
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --port 8080 \
  --max-context 16384
```

The server loads the model before opening the port. Wait for
`TurboFieldfareServer ready`, then keep the process running while clients use
it.

Check the server from another terminal:

```bash
curl --silent --show-error http://127.0.0.1:8080/health
curl --silent --show-error http://127.0.0.1:8080/v1/models
curl --silent --show-error http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_completion_tokens": 16
  }'
```

By default, the server runs one generation and admits up to four additional
requests for preparation or queueing. The limit is enforced before prompt
rendering and tokenization. Use `--queue-limit` to change it. Press Control-C
to stop the server.

## Runtime settings

The server accepts the same runtime flags as the CLI, with the same values and
defaults. See [Runtime controls](RUNTIME_CONTROLS.md) for what each one does.

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --expert-cache-slots 32 \
  --expert-cache-policy lru \
  --prefill on \
  --prefill-chunk-tokens 64 \
  --rdadvise bounded
```

Without these flags the server runs the production defaults: 16 expert-cache
slots, LFU eviction, chunked prefill on with 128-token chunks, and read advice
off. `--prefill-chunk-tokens auto` runs at the cap,
256, on the server: a per-request size is the smallest allowed size covering the
span being prefilled, so the cap prefills every prompt in exactly those spans,
and it costs about 32.5 MB of prefill scratch against about 16.4 MB at the 128
default. Values are validated before the model loads, so an unsupported one
exits with the usage text rather than failing partway through startup. Chunked
prefill needs at least 16 expert-cache slots, so `--expert-cache-slots 8`
requires `--prefill off`.

The settings are fixed for the life of the process. Restart the server to
change them.

## Connect a client

The base URL is `http://127.0.0.1:8080/v1`. Some client libraries require an
API key, but the server ignores it.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

OpenCode:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "turbofieldfare": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TurboFieldfare",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "gemma-4-26b-a4b-it": {
          "name": "Gemma 4 26B-A4B IT",
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "limit": {
            "context": 16384,
            "output": 4096
          }
        }
      }
    }
  }
}
```

Select `turbofieldfare/gemma-4-26b-a4b-it` in OpenCode.

`attachment` and `modalities` are what make images work. OpenCode decides whether
a model accepts images from this configuration, not from the `capabilities` field
`/v1/models` returns, so without them it refuses an image with "this model does
not support image input" before sending anything. Images then reach the server
through OpenCode's `read` tool, which returns image files as attachments; ask it
to read the file explicitly.

Pi uses its `openai-completions` adapter:

```json
{
  "providers": {
    "turbofieldfare": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsReasoningEffort": false,
        "supportsStrictMode": false,
        "supportsUsageInStreaming": true
      },
      "models": [{
        "id": "gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B-A4B IT",
        "reasoning": false,
        "input": ["text", "image"],
        "contextWindow": 16384,
        "maxTokens": 4096
      }]
    }
  }
}
```

Keep the client context setting at or below the server's `--max-context`.

Pi's `input` field declares image support the same way OpenCode's `modalities`
does. With it, `pi -p @photo.png "What is in this image?"` sends the image as a
base64 data URL in an `image_url` part, which is the shape this server accepts;
its RPC interface takes an explicit `images` array and produces the same
request.

## Prompt reuse

KV reuse is on by default. Send the complete message history with every
request. When a request continues a retained conversation exactly, the server
reuses the verified KV prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

Reuse serves conversation **continuations**. Two independent requests that
share a long prefix do not reuse it however much text they have in common; a
request that continues a conversation the server still holds does.

### How many conversations are retained

By default the server retains **one**, and any other caller's request replaces
it. Two conversations interleaving on one server therefore evict each other on
every call, and neither ever continues — including a single-shot request, such
as a summary, that will never continue anything itself but still takes the slot.
Nothing in the response distinguishes this from a client sending a history that
does not match; `cached_tokens` is 0 either way.

`--prompt-cache-slots` raises the number retained:

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --prompt-cache-slots 3
```

A slot is a whole KV lineage, not bookkeeping: it allocates about 575 MiB at the
default 16,384-token context. The startup log reports the figure for the context
configured.

```text
prompt cache mode=single-prefix slots=3 kv_per_slot_max=574.6MiB kv_total_max=1723.9MiB
  (ceiling; a slot goes resident as far as its conversation reaches)
```

**That is a ceiling, not a prediction.** The buffers fault in on write, so a
slot holds only what its conversation has reached, and dropping a lineage
returns the pages. The two halves behave differently:

| layers | storage | residency |
|---|---|---|
| 25 sliding-window | ~255 MiB | reached within the first ~1,300 tokens |
| 5 full-attention | ~320 MiB at 16K | grows with position |

So a 4,000-token conversation holds roughly 255 + 320 × 4,000/16,384 ≈ **333
MiB**, not 575. Three slots carrying prompts that size sit near 1.0 GiB rather
than the 1.7 GiB ceiling.

The gap is narrower than it looks, though: the ring is the larger half and is
reached early, so a quarter of the context costs 57% of the ceiling rather than
a quarter of it. A conversation has to be short to be cheap, and one long enough
to be worth caching is most of the way up already. Size for the ceiling if the
machine must never swap, and expect the smaller number in Activity Monitor.

Only the five full-attention layers scale with `--max-context`; the ring caps
the other twenty-five, so a four-times-larger context is not a four-times-larger
slot. Slots past the first are allocated when a conversation first uses one, so
a server configured for more than its clients open pays for the ones they do.

A request reuses a slot only when that slot's history, tools, images and
runtime identity all match, so raising the count can cost a prefill and can
never return a prefix a caller did not send.

### Reserving a slot

Without a key, requests share the slots and the least recently used one is
evicted when a new conversation needs room. Send `prompt_cache_key` to name a
conversation and reserve a slot for it:

```json
{
  "model": "gemma-4-26b-a4b-it",
  "messages": [{"role": "user", "content": "..."}],
  "prompt_cache_key": "extraction"
}
```

A key names one lineage. A keyed request looks at its own slot and nowhere
else, and unkeyed traffic uses only the slots no key has claimed — so a caller
that names its conversation cannot be evicted by one that does not. The key is
opaque to the server, is never logged, and must be 1 to 128 UTF-8 bytes.

**One key, one conversation at a time.** Because the key selects a slot rather
than a history, reusing a name *over time* is the intended lifecycle: when a
conversation ends and a new one takes its place under the same name, the new one
misses, claims the slot that name already holds, and disturbs nothing else. A
periodic job that starts a fresh conversation each run should keep one stable
key for exactly this reason.

Two conversations alive *at the same time* under one name is the case to avoid.
They resolve to the same slot and overwrite each other on every call, so they
never continue — and unlike unkeyed traffic they cannot spread across the
remaining slots, which sit unused:

```text
three live conversations, three slots, one shared key
  A → claims slot 0        B → overwrites slot 0      C → overwrites slot 0
  A → overwrites slot 0    ...                        slots 1 and 2 idle
```

That is the one way sending a key leaves a client worse off than sending none.
Derive the key from whatever identifies the conversation — a thread or session
id, not the calling surface — whenever more than one can be open at once.

Earlier versions accepted `prompt_cache_key` and ignored it, so a client already
sending one needs no change — but both versions answer 200, so the field alone
cannot tell them apart. `GET /health` reports the count:

```json
{"status": "ok", "vision": "ready", "prompt_cache_slots": 3}
```

A server without the field ignores the key; one reporting `1` honours it but has
a single slot to give.

Size the count for the conversations that run at once. Single-shot traffic —
a summary or a classification that will never be continued — should send
`prompt_cache_mode: off` instead of being budgeted a slot; see below.

More names than slots is allowed: a new key recycles the least recently used
one, which then starts cold. Use `--prompt-cache-mode off` to disable reuse
entirely; it retains nothing, so it cannot be combined with more than one slot.

### Opting out for one request

A caller that will never continue a conversation still has to prefill
somewhere, and prefilling into a slot destroys the conversation in it. So a
single-shot request costs someone else their prefix just by running, even
though it can never benefit from the cache itself.

`prompt_cache_mode` overrides `--prompt-cache-mode` for one request, in the
flag's own vocabulary:

```json
{
  "model": "gemma-4-26b-a4b-it",
  "messages": [{"role": "user", "content": "..."}],
  "prompt_cache_mode": "off"
}
```

Such a request reuses nothing, publishes nothing, and leaves every slot as it
found it — it prefills into a lineage kept aside for exactly this, allocated
the first time one is used. A server no client opts out of never pays for it.

The override only narrows. `single-prefix` on a server started with
`--prompt-cache-mode off` is accepted and changes nothing, because no lineage
is retained for the request to join. Any other value is a 400 rather than a
fall back to the server's mode: `"none"` is a caller who believes caching is
off, and answering 200 with it on is the failure `unknown_parameter` exists to
prevent. Sending `prompt_cache_key` together with `prompt_cache_mode: off` is
also a 400 — one asks to reserve a lineage and the other asks not to have one,
and either silent winner leaves the caller believing the opposite of what
happened.

Each request reports which it took:

```text
prompt cache slot=1 of 3 occupied=2 outcome=hit
prompt cache opted-out lineage=3 slots=3 occupied=2 outcome=not-cached
```

The second line is worth having because a request that opted out and one that
simply missed both report `cached_tokens` 0, and only the log separates them.

## Tool calls

The server can return OpenAI-style function calls, but it cannot authorize or
execute them. The client runs the tool loop:

1. Send function schemas in `tools`.
2. When `finish_reason` is `"tool_calls"`, inspect each function name and JSON
   argument object. Apply the client's normal permission checks before running
   the function.
3. Append the assistant message, including its unchanged `tool_calls`.
4. Append each result as a `role: "tool"` message. Its `tool_call_id` must
   match the call it resolves.
5. Send the complete history and tool schemas again.

The server accepts only function tools. Omit `tool_choice` or set it to `auto`
to allow calls. Set it to `none` to disable them. The server does not support
`required`, named tool selection, or `parallel_tool_calls: false`.

Tool schemas need a non-null object at the top level and explicit JSON Schema
types. Nested properties and items may use nullable forms with one concrete
type plus `null`, including equivalent two-branch `anyOf` and disjoint `oneOf`
forms. Unions of string constants are also supported. Overlapping `oneOf`,
mixed-type unions such as `string | object`, and `allOf` return HTTP 400 with
`invalid_tool_schema`; the server does not guess which branch the model should
use.

## Supported API

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Chat Completions supports JSON and Server-Sent Events responses. Set
`"stream": true` for streaming. Set
`"stream_options": {"include_usage": true}` to receive a final usage chunk.

Requests may contain system, developer, user, assistant, and tool messages.
Supported options include `temperature`, `top_p`, `top_k`,
`repetition_penalty`, `seed`, `stop`, `max_tokens`,
`max_completion_tokens`, `prompt_cache_key`, `prompt_cache_mode`, and
function-tool fields.

Unknown top-level request fields return HTTP 400 with `code`
`unknown_parameter` and the field name in `param`, so a misspelled option is
refused rather than silently ignored. `response_format` is accepted only as
`{"type": "text"}`; `json_object` and `json_schema` return
`unsupported_value`, as do `logit_bias`, `top_logprobs`, `reasoning_effort`,
`verbosity`, `modalities`, `audio`, `prediction`, `web_search_options`, and
the legacy `functions` and `function_call`. `response_format` must be an object; any
other JSON value returns `invalid_value`. `user`, `store`, `metadata`,
`service_tier`, and `safety_identifier` are accepted and ignored.
`prompt_cache_key` reserves a prompt-cache slot (see
[Prompt reuse](#prompt-reuse)) and must be 1 to 128 UTF-8 bytes; an empty or
longer one returns `invalid_value`. Earlier versions accepted and ignored it,
so honouring it cannot break a client already sending one.
`prompt_cache_mode` takes `off` or `single-prefix`; any other value, or `off`
alongside a `prompt_cache_key`, returns `invalid_value`. A top-level field set to `null` is treated as absent. Fields inside
`messages`, `tools`, and `stream_options` are not checked for extras. Inside
`stream_options` only `include_usage` is read, so a misspelled key there is
ignored rather than refused.

The server supports one model and one choice. It does not support the Responses
API, legacy Completions, embeddings, structured output,
batching, log probabilities, or remote model switching.

Context length can be 4K, 8K, 16K, 32K, 64K, 96K, 128K, 192K, or 256K.
The default is 16K. Before loading the model, the server checks whether the
selected context and expert cache fit its memory estimate for your Mac.
Larger contexts need more memory for the FP16 KV cache. See
[context length and memory](RUNTIME_CONTROLS.md#context-length-and-memory)
for the calculation and diagnostic override, and the
[long-context report](experiments/summaries/10-long-context.md) for measured
results and validation limits. On an 8 GB Mac, run one model process at a
time and watch memory pressure.

For long requests, stderr reports the request lifecycle as prepared, queued,
generating, completed, or failed. It includes token counts and timing, but not
prompt text, tool arguments, headers, or request bodies. Each request also
reports which prompt-cache slot it resolved to and whether it continued one:

```text
prompt cache slot=1 of 3 occupied=2 outcome=hit
prompt cache slot=2 of 3 occupied=3 outcome=history-diverged
```

The slot index says which lineage without repeating the `prompt_cache_key` the
caller chose, which is never logged.

## Images

`messages[].content` accepts `image_url` parts in user messages. An
`image_url` on any other role returns HTTP 400. The URL must be a data URL,
and `detail` must be absent or set to `auto`.

Image bytes go to disk as the request body arrives, so a large upload does not
sit in memory. Before any pixel is decoded, the server works out how many
tokens the images will occupy and rejects the request if they do not fit the
context. Bodies and images over their limits return HTTP 413.

`GET /v1/models` reports `capabilities` as `["text", "image"]` when a valid
companion pack is loaded and `["text"]` otherwise. `GET /health` reports the
same state under `vision`, as `ready`, `missing`, `invalid`, or `unsupported`.
`unsupported` means the image tower cannot run on this Mac; it requires M2 or
newer. Text requests remain available.

Choose the pack and the residency policy with `--vision-pack <dir>` and
`--vision-residency on-demand|keep-ready`.
