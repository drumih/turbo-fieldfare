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

By default, the server runs one generation and queues up to four requests. Use
`--queue-limit` to change the queue size. Press Control-C to stop the server.

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
          "name": "Gemma 4 26B-A4B IT"
        }
      }
    }
  }
}
```

Select `turbofieldfare/gemma-4-26b-a4b-it` in OpenCode.

## Prompt reuse

Single-prefix KV reuse is on by default. Send the complete message history with
every request. When a request continues the retained conversation exactly, the
server reuses the verified KV prefix and reports the number of reused tokens in:

```text
usage.prompt_tokens_details.cached_tokens
```

The server retains one prefix. A different or incompatible history replaces
it. Use `--prompt-cache-mode off` to disable reuse.

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

## Errors

The prompt is rendered and checked against the context window before any
response is written, so an overlong prompt, an unknown model, an unsupported
parameter, an oversized body, or a full queue comes back as a JSON error
envelope with a real status code — `400`, `404`, `413`, `415`, or `429` — and
no stream is started. Asking for `"stream": true` does not change this.

A failure raised after that point cannot change the status, because a
streaming request already has `200` and the SSE head on the wire. It is
reported in-band instead: one frame carrying an `error` object, then
`data: [DONE]`, then a normal end of the chunked body.

```text
data: {"error":{"message":"generation failed","code":"internal_error","type":"server_error"}}

data: [DONE]

```

The frame carries the same envelope the blocking path would have returned, so
a failure that can only surface once generation is under way still names its
cause. Reusing a KV prefix is the case that reaches it: whether the retained
prefix plus the new turn fits the context window is known only after the
prompt cache has been matched, which happens after the head is committed.

Treat any frame with an `error` key as fatal for that request; no
`finish_reason` chunk precedes it. The stream is never terminated by dropping
the connection, so a client that sees an aborted transport (`TypeError:
terminated` under undici, for example) should look for a dead server process
or its own timeout rather than a generation error.

## Server log

The server writes one line per request to stderr:

```text
[2026-07-31T17:03:10Z] request chatcmpl-f6a02587… started streaming=true
[2026-07-31T17:03:12Z] request chatcmpl-f6a02587… completed in 2.4s prompt=812 cached=768 completion=96 finish=stop
```

The start line is written before the model runs, so a long prefill — which
emits nothing for minutes — is distinguishable from a wedged server. The
completion line reports how much of the prompt the KV prefix supplied in
`cached`, matching `usage.prompt_tokens_details.cached_tokens`.

A failed request logs the status it would have carried, whether or not a
stream had already committed `200`, along with the underlying error — which is
more detail than the response carries, since responses deliberately do not
leak runtime internals:

```text
[2026-07-31T17:09:32Z] request chatcmpl-b36dd1ed… failed status=400 streaming=true error=context_length_exceeded: prompt exceeds the configured context
```

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
