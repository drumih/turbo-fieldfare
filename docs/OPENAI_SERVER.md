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

Unknown top-level request fields return HTTP 400 with `code`
`unknown_parameter` and the field name in `param`, so a misspelled option is
refused rather than silently ignored. `response_format` is accepted only as
`{"type": "text"}`; `json_object` and `json_schema` return
`unsupported_value`, as do `logit_bias`, `top_logprobs`,
`verbosity`, `modalities`, `audio`, `prediction`, `web_search_options`, and
the legacy `functions` and `function_call`. `response_format` must be an object; any
other JSON value returns `invalid_value`. `user`, `store`, `metadata`,
`service_tier`, `prompt_cache_key`, and `safety_identifier` are accepted and
ignored. A top-level field set to `null` is treated as absent. Fields inside
`messages`, `tools`, and `stream_options` are not checked for extras. Inside
`stream_options` only `include_usage` is read, so a misspelled key there is
ignored rather than refused.

The server supports thinking and reasoning controls via `reasoning_effort`,
`chat_template_kwargs.enable_thinking`, and `thinking.type`. See
[Thinking and reasoning](#thinking-and-reasoning) below.

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
prompt text, tool arguments, headers, or request bodies.

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

## Thinking and reasoning

Gemma 4 natively supports reasoning via internal channel tokens. When thinking
is enabled, prompt templates inject `<|think|>` into the system guidance and
leave the model turn open, prompting the model to emit reasoning tokens inside
`<|channel>thought\n...<channel|>` before outputting visible responses or calling
tools. When thinking is disabled, the prompt closes the thought channel
immediately (`<|channel>thought\n<channel|>`), suppressing internal reasoning.

### Server thinking policy

Configure the server's thinking policy with `--thinking <auto|on|off>`:

```bash
.build/release/TurboFieldfareServer \
  --model scratch/gemma4.gturbo \
  --thinking auto
```

- `auto` (default): Thinking is enabled only when explicitly requested by the
  client (e.g., via `reasoning_effort`, `chat_template_kwargs`, or `thinking`).
- `on`: Thinking is enabled by default for all requests, unless explicitly
  disabled by a request parameter (`reasoning_effort: "none"`, etc.).
- `off`: Thinking is disabled for all requests, overriding any client request.

### Client parameters

The server supports standard OpenAI and open-weight reasoning controls:

1. **`reasoning_effort`**:
   Accepted values: `"low"`, `"medium"`, `"high"`, `"default"`, and `"none"`.
   Values other than `"none"` enable thinking.

   ```json
   {
     "model": "gemma-4-26b-a4b-it",
     "messages": [{"role": "user", "content": "How many r's are in strawberry?"}],
     "reasoning_effort": "high"
   }
   ```

2. **`chat_template_kwargs.enable_thinking`**:
   Accepts a boolean (`true` or `false`), matching Hugging Face / vLLM client conventions:

   ```json
   {
     "model": "gemma-4-26b-a4b-it",
     "messages": [{"role": "user", "content": "Solve this puzzle..."}],
     "chat_template_kwargs": {"enable_thinking": true}
   }
   ```

3. **`thinking.type`**:
   Accepts `{"type": "enabled"}` or `{"type": "disabled"}`.

### Response format

- **Streaming (`stream: true`)**:
  Reasoning deltas stream under `delta.reasoning_content`:

  ```json
  data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning_content":"Let's count the letters..."}}]}
  data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"There are 3 'r's."}}]}
  ```

- **Non-streaming**:
  The full thought process is returned in `choices[0].message.reasoning_content`:

  ```json
  {
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "Let's count the letters in strawberry:\ns-t-r-a-w-b-e-r-r-y...",
        "content": "There are 3 'r's in strawberry."
      },
      "finish_reason": "stop"
    }],
    "usage": {
      "prompt_tokens": 15,
      "completion_tokens": 85,
      "total_tokens": 100,
      "prompt_tokens_details": {"cached_tokens": 0},
      "completion_tokens_details": {"reasoning_tokens": 62}
    }
  }
  ```

- **Usage reporting**:
  When reasoning tokens are generated, `usage.completion_tokens_details.reasoning_tokens`
  reports the exact count of tokens spent in the reasoning channel.

### Tool calls and prompt cache

Thinking works seamlessly alongside tool calls: the model can deliberate inside
the thought channel before invoking tools or producing the visible answer. The
decoder parses thought tokens and routes them cleanly without leaking control
markers into visible content or tool call arguments. Single-prefix prompt
caching matches continuation turns with identical thinking states.

