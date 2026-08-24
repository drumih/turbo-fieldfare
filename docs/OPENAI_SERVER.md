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

Without these flags the server runs the production defaults: 24 expert-cache
slots, LFU eviction, chunked prefill on with 128-token chunks, and read advice
off. Values are validated before the model loads, so an unsupported one exits
with the usage text rather than failing partway through startup. Chunked
prefill needs at least 16 expert-cache slots, so `--expert-cache-slots 8`
requires `--prefill off`.

The settings are fixed for the life of the process. Restart the server to
change them.

Pass `--diagnostics` to include Qwen decode and prefill diagnostics in responses.
For non-streaming requests, `turbo_fieldfare_diagnostics` is a top-level field.
For streaming requests, it is included only in the final usage chunk;
diagnostics also cause that usage chunk to be sent when `include_usage` was not
requested. Responses omit the field when diagnostics are disabled.

Decode fields remain in the versioned top-level diagnostics object. Chunked
Qwen prefill adds an optional nested `prefill` object:

Qwen decode diagnostics schema version 5 includes non-overlapping GPU timestamps
for both decode command buffers. The first group covers the combined
mixer/shared-expert/router command buffer:

| Field | Meaning |
| --- | --- |
| `gpu_stage_timing_sample_count` | Number of decoded layer executions with a complete valid timestamp sample |
| `gpu_mixer_nanos` | Exclusive GPU time from the mixer boundary to the shared-expert boundary |
| `gpu_shared_expert_nanos` | Exclusive GPU time from the shared-expert boundary to the router boundary |
| `gpu_router_nanos` | Exclusive GPU time from the router boundary to the end boundary |

Each `layers` entry exposes the corresponding `gpuStageTimingSampleCount`,
`gpuMixerNanos`, `gpuSharedExpertNanos`, and `gpuRouterNanos` totals. A fully
sampled response has one sample per decoded layer execution. Unsupported or
invalid Metal counter samples leave the count and duration totals at zero.

Schema version 5 adds the routed tail:

| Field | Meaning |
| --- | --- |
| `routed_setup_nanos` | CPU wall time for routed argument-buffer and weight-view setup after expert fetch |
| `routed_command_buffer_encoding_nanos` | CPU wall time to create and encode the routed command buffer |
| `routed_command_buffer_commit_nanos` | CPU wall time spent submitting the routed command buffer with `commit()` |
| `routed_command_buffer_wait_nanos` | Host wall time blocked for routed command-buffer completion |
| `routed_gpu_stage_timing_sample_count` | Number of decoded layer executions with a complete valid routed timestamp sample |
| `gpu_routed_phase1_nanos` | Exclusive GPU time for routed gate/up projection and activation |
| `gpu_routed_phase2_nanos` | Exclusive GPU time for routed down projection and weighted reduction |
| `gpu_routed_combine_nanos` | Exclusive GPU time for shared gating, routed/shared combination, and residual add |

Each `layers` entry exposes the corresponding camel-case routed fields. A fully
sampled response has one front-stage and one routed-stage sample per decoded
layer execution. CPU wall fields overlap their enclosing stage totals, and GPU
durations execute inside command-buffer wait time; do not add overlapping
fields to estimate total decode time. Zero durations without a positive sample
count are not measurements. Timestamp markers and host substage clocks are
enabled only by diagnostics mode and do not change the production path when
diagnostics are disabled.

The optimized production decode schedule chains routed layer N with the front
of layer N+1. Diagnostics mode retains separate front and routed command
buffers so schema-5 timestamps remain independently attributable. Diagnostic
throughput and command-buffer counts therefore describe the measurement
schedule, not the optimized production schedule.

Chunked Qwen prefill uses these nested fields:

| Field | Meaning |
| --- | --- |
| `execution_path` | Prefill implementation used: `scalarFallback`, `chunked`, or `mixed` |
| `scalar_forward_count` | Scalar forward passes performed during prefill |
| `chunk_pass_count` | Chunked prefill passes performed |
| `command_buffer_count` | Metal command buffers submitted during prefill |
| `embedding_nanos` | Token embedding wall time |
| `mixer_nanos` | RMSNorm, attention or DeltaNet, residual, and post-attention norm wall time |
| `deltanet_mixer_nanos` | Portion of `mixer_nanos` spent in DeltaNet layers |
| `full_attention_mixer_nanos` | Portion of `mixer_nanos` spent in full-attention layers |
| `moe_prepare_nanos` | Shared expert, router, route grouping, and metadata preparation wall time |
| `expert_fetch_nanos` | Routed expert binding and fetch wall time |
| `routed_moe_nanos` | Streamed routed-expert execution wall time |
| `moe_reduce_nanos` | Routed reduction, shared gate combination, and residual wall time |
| `final_head_nanos` | Final normalization and language-model head wall time |
| `attributed_wall_nanos` | Sum of the seven stage timing fields |

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

For Qwen, an identical request also restores the retained recurrent/KV prompt
state and next-token logits without recomputing prefill. The server retains one
prefix. A different or incompatible history replaces it. Use
`--prompt-cache-mode off` to disable reuse.

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
API, legacy Completions, embeddings, structured output,
batching, log probabilities, or remote model switching.

Context length can be 4K, 8K, 16K, 32K, or 64K. The default is 16K. Larger FP16
KV contexts use more memory. On an 8 GB Mac, run one model process at a time and
watch memory pressure.

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
