# Local OpenAI-compatible server

`TurboFieldfareServer` exposes a local Chat Completions API for one loaded
model — either Gemma 4 26B-A4B (`.gturbo` v1) or Kimi K3 (`.gturbo` v2,
see [Kimi K3 bundles](#kimi-k3-bundles) below). It binds to `127.0.0.1`
without authentication or TLS. Do not expose it through a proxy or tunnel.

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
        "contextWindow": 16384,
        "maxTokens": 4096
      }]
    }
  }
}
```

Keep the client context setting at or below the server's `--max-context`.

For a K3 batch/research server on a 128 GB machine, the disk-oriented controls
are also available on the server executable:

```bash
swift run -c release TurboFieldfareServer \
  --model scratch/kimi-k3.gturbo \
  --max-context 262144 \
  --prompt-cache-mode single-prefix \
  --prefill-chunk 32 \
  --expert-predict off \
  --expert-cache-gib 16 \
  --expert-io-workers auto \
  --expert-io-splits 1 \
  --expert-io-cache uncached \
  --model-verification trusted-install
```

The direct-resident RAM banks are optional and compete with the int8 trunk,
the MLA context cache, applications, and the OS. Only the currently executing
layer receives a Metal view, avoiding a full GPU working-set reservation equal
to the cache size. 24 GiB covers 91 of 92 MoE layers without cache-to-compute
copies, but at the 262144-token context ceiling the MLA cache and int8 trunk
leave less headroom — an 8/16/24 GiB A/B on the real checkpoint at this
context confirmed 16 GiB as the correct default: it decodes as fast as
8 GiB while reading noticeably less from SSD, whereas 24 GiB's larger cache
reads the least but runs ~51% slower once its memory pressure outweighs the
I/O it saves. See
[docs/K3_DISK_RUNTIME.md#cache-size-sweep-8--16--24-gib-real-checkpoint-262144-context](docs/K3_DISK_RUNTIME.md#cache-size-sweep-8--16--24-gib-real-checkpoint-262144-context)
for the full numbers. Check `memory_pressure -Q`; reduce the budget to 16 or
8 GiB before reducing context if pressure is poor. Repeating
`--expert-shard-root /Volumes/...` enables a prepared multi-SSD stripe set.

## Prompt reuse

Single-prefix state reuse is on by default. Send the complete message history
with every request. When a request continues the retained conversation exactly,
the server reuses the verified prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

The server retains one prefix. A different or incompatible history replaces
it. Gemma retains its KV state; K3 retains the full recurrent KDA/conv state,
active MLA rows, last logits, and routing-predictor history. Both paths require
exact token-prefix identity. Use `--prompt-cache-mode off` to disable reuse.

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
`max_completion_tokens`, and function-tool fields.

The server supports one model and one choice. It does not support the Responses
API, legacy Completions, embeddings, multimodal input, structured output,
batching, log probabilities, or remote model switching.

Context length can be 4K, 8K, 16K, 32K, or 64K. The default is 16K. Larger FP16
KV contexts use more memory. On an 8 GB Mac, run one model process at a time and
watch memory pressure.

For long requests, stderr reports the request lifecycle as prepared, queued,
generating, completed, or failed. It includes token counts and timing, but not
prompt text, tool arguments, headers, or request bodies.

## Kimi K3 bundles

When `--model` points at a `.gturbo` **v2** bundle (Kimi K3), the server probes
the manifest and serves the K3 engine instead of Gemma. Differences from the
Gemma behavior above:

- The model id defaults to the bundle's manifest id (for example `Kimi-K3`);
  `--model-id` still overrides.
- Requests accept an optional `reasoning_effort` field (`"low"`, `"high"`, or
  `"max"`; anything else is a 400). Thinking is always on; assistant responses
  carry `reasoning_content` alongside `content`, and streamed reasoning arrives
  as `reasoning_content` deltas. Multi-turn clients must send prior assistant
  turns back with both `reasoning_content` and `tool_calls` intact — dropping
  them degrades the model (upstream serving requirement).
- `tool_choice` accepts `"auto"`, `"none"`, and `"required"`.
- The `stop` field is rejected with a 400 for K3 (token-level stop only);
  `max_tokens`/`max_completion_tokens` work as usual. Generation stops at the
  model's `<|end_of_msg|>` token.
- Single-prefix state reuse is enabled by default. K3 snapshots the recurrent
  KDA/conv state, active MLA cache rows, last logits, and routing-predictor
  history; a token mismatch falls back to a full prefill. Hits report their
  exact reused count in `cached_tokens`. Use `--prompt-cache-mode off` to
  disable it.
- Decode speed is SSD-bound: plan for roughly 0.3–1 tok/s and size client
  timeouts accordingly. Prefill uses the chunked NAX path by default.
