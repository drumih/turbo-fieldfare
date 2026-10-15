---
name: turbofieldfare-api
description: Manage and call the TurboFieldfare local OpenAI-compatible Chat Completions API server (Gemma 4 26B-A4B IT) with token budgeting rules, max context boundaries, and task-specific preset profiles.
user-invocable: true
argument-hint: "[start|stop|status|chat] [code|structured|creative|balanced|long] [prompt]"
---

# TurboFieldfare API Skill

Quick reference for AI agents to launch and query the local **TurboFieldfare** OpenAI-compatible API server with strict token budgeting and task-optimized parameter presets.

> Paths use `$HOME` / `~` so the skill is portable. The model is a ~14.3 GB
> `.gturbo` directory; only one model-owning process may run at a time
> (server, Mac app, CLI, or decode service are mutually exclusive).

---

## 📍 Environment & Endpoint

- **Server Binary**: `.build/release/TurboFieldfareServer` (build with `swift build -c release --product TurboFieldfareServer` from the repo root)
- **Model Path**: `"$HOME/Library/Application Support/TurboFieldfare/gemma4.gturbo"` (packaged app location), or `scratch/gemma4.gturbo` from a checkout
- **Base URL**: `http://127.0.0.1:52642/v1`
- **Port `52642`**: the server's *code* default is `8080`, but this guide uses
  **`52642`** instead to avoid collisions. `8080`, `3000`, `5000`, and `8000` are
  heavily used by dev servers; `52642` sits in the macOS ephemeral range
  (`49152–65535`), which has no IANA-registered service, so it rarely clashes
  with another app's fixed port. (Caveat: the OS can hand ephemeral ports to
  transient sockets, so if a bind fails, retry or pick another value in the
  range.) Override with `--port`.
- **Model ID**: `gemma-4-26b-a4b-it` (server default; override with `--model-id`)
- **Loopback only**: `127.0.0.1`. No auth, no TLS — never proxy, tunnel, or expose.

---

## 📏 1. Context Window & Token Budgeting Rules

Total Context Budget: **`Total Tokens = Prompt Tokens (In) + Completion Tokens (Out)`**

- **Server Context Limit (`--max-context`)**: Default `16384` (Supports `4096`, `8192`, `16384`, `32768`, `65536`).
- **Safety Boundary Formula**: `max_completion_tokens <= max_context - prompt_tokens`
- **Context Overflow**: If the prompt exceeds `--max-context`, the server returns `400 Invalid Request` (`prompt exceeds the configured context`).
- **KV Cache Optimization**: Single-prefix KV reuse is **on by default** (`--prompt-cache-mode single-prefix`). System prompts & conversation prefixes are reused; reused token count is reported in `usage.prompt_tokens_details.cached_tokens`. Disable with `--prompt-cache-mode off`.

---

## 🚀 2. Server Lifecycle & Context Setup

> **Before starting**, confirm no other TurboFieldfare model process is running:
> ```bash
> pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
> ```
> If it prints a match, **do not start** — only one model process at a time.

### Standard Startup (16K Context Window, guide port 52642)
```bash
.build/release/TurboFieldfareServer \
  --model "$HOME/Library/Application Support/TurboFieldfare/gemma4.gturbo" \
  --port 52642 \
  --max-context 16384
```

### Extended Startup (32K / 64K Context Window for Large Context)
```bash
.build/release/TurboFieldfareServer \
  --model "$HOME/Library/Application Support/TurboFieldfare/gemma4.gturbo" \
  --port 52642 \
  --max-context 32768
```

Optional server flags:
- `--queue-limit <count>` — maximum queued requests (default `4`).
- `--prompt-cache-mode <off|single-prefix>` — KV reuse mode (default `single-prefix`).

### Stop the Server

Stop the server from the **terminal where you launched it** with `Control-C`.
The server listens for `SIGINT`/`SIGTERM` and shuts down cleanly.

> Only stop a server **you** launched. Do not terminate a server (or Mac app)
> started by the user or another session. If you must clean up a process you own,
> target it explicitly rather than blanket-killing by name.

### Health Check
```bash
curl --silent --show-error http://127.0.0.1:52642/health
curl --silent --show-error http://127.0.0.1:52642/v1/models
```
Wait for `TurboFieldfareServer ready` before sending requests; the model loads
before the port opens.

---

## 🎯 3. Preset Parameter Profiles for Task-Specific Requests

> `top_p` must be `> 0` and `<= 1`; `top_k` must be `1–256`. At `temperature 0`
> the server uses greedy decoding, so `top_p` / `top_k` have no sampling effect.

| Preset | Target Use-Case | `temperature` | `top_k` | `top_p` | `repetition_penalty` | Recommended `max_completion_tokens` |
| --- | --- | --- | --- | --- | --- | --- |
| **`code`** | Code Generation & Debugging | `0.1` | `32` | `0.90` | `1.05` | `2048` |
| **`structured`** | JSON & Tool Calling (deterministic) | `0.0` (greedy) | N/A (greedy) | N/A (greedy) | `1.00` | `1024` |
| **`creative`** | Brainstorming & Writing | `0.7` | `100` | `0.95` | `1.10` | `2048` |
| **`balanced`** | General Chat & Summaries | `0.2` | `64` | `0.95` | `1.00` | `1024` |
| **`long`** | Long-context synthesis (start server with `--max-context 32768`) | `0.3` | `64` | `0.95` | `1.05` | `4096` |

---

## 💬 4. Profile Request Examples (cURL)

### Code Generation (`code` profile)
```bash
curl -s http://127.0.0.1:52642/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "messages": [
      {"role": "system", "content": "You are an expert Swift software engineer."},
      {"role": "user", "content": "Write an async/await HTTP fetch utility in Swift."}
    ],
    "temperature": 0.1,
    "top_k": 32,
    "top_p": 0.90,
    "repetition_penalty": 1.05,
    "max_completion_tokens": 2048
  }'
```

### Structured / JSON (`structured` profile, greedy)
```bash
curl -s http://127.0.0.1:52642/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-26b-a4b-it",
    "messages": [
      {"role": "user", "content": "Return a JSON object with keys \"name\" and \"age\" for a fictional persona."}
    ],
    "temperature": 0,
    "max_completion_tokens": 1024
  }'
```

---

## 🔌 5. Connect OpenCode (maintainer's reference client)

TurboFieldfare's server docs use **OpenCode** as the reference client setup. To
run an OpenCode coding agent backed by the local Gemma model, register a
`turbofieldfare` provider in `opencode.json`. The server ignores the API key, so
any non-empty value works.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "turbofieldfare/gemma-4-26b-a4b-it",
  "provider": {
    "turbofieldfare": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "TurboFieldfare",
      "options": {
        "baseURL": "http://127.0.0.1:52642/v1",
        "apiKey": "local"
      },
      "models": {
        "gemma-4-26b-a4b-it": {
          "name": "Gemma 4 26B-A4B IT",
          "limit": { "context": 16384, "output": 8192 },
          "options": { "temperature": 0.2, "topP": 0.95, "topK": 64 }
        }
      }
    }
  }
}
```

- Set the top-level `model` key so OpenCode picks TurboFieldfare by default.
- The `limit` block is **required** — without `context`/`output`, OpenCode does
  not know the model's token budget and may fail to generate.
- The `options` block sets sampling defaults via the AI SDK names
  (`temperature`, `topP`, `topK`, `maxOutputTokens`). Note: `repetition_penalty`
  is a TurboFieldfare extension, **not** an AI SDK standard option, so OpenCode
  cannot pass it — leave it at the server default (1.0).
- For a Python client, use the `openai` SDK with the same base URL and pass
  `repetition_penalty` directly in the request body.

### ⚠️ Known OpenCode + TurboFieldfare limits (empirically verified)

The server runs **one generation at a time** and queues up to four (`--queue-limit`).
Two practical limits surface when driving it from OpenCode:

1. **Auto-title generation causes `generation queue is full`.** OpenCode fires a
   `small=true agent=title` request (to name the session) at almost the same time
   as the real `agent=build` request. The second one hits the queue and fails with
   `AI_APICallError: generation queue is full`, then AI SDK retries 3× and stalls.
   **Fix:** pass `--title <name>` to `opencode run` so it skips the title request:
   ```bash
   opencode run -m turbofieldfare/gemma-4-26b-a4b-it --title "session" "your prompt"
   ```

2. **The agent context is large and slow on Gemma.** OpenCode's `build` agent sends
   the system prompt + every skill definition + tool schemas (tens of thousands of
   tokens) on every turn, so a single agent turn can take minutes. For plain chat,
   prefer the raw cURL/Python path in §4 — it is much faster because it sends only
   the prompt, not the full agent context.


---

## 🧪 6. CLI Fallback (when the server is unsuitable)

Prefer the server (§2) for repeated calls, conversations, and tool loops — it
loads the model once and reuses the KV prefix. Use the **CLI** only for:

- A **single one-shot** generation (diagnostic, benchmark, smoke test).
- A **fully reproducible** sample via `--seed`.
- When you cannot keep a long-lived process alive.

> **Cost:** the CLI reloads the entire model on every invocation (tokenizer →
> Metal context → `Model.load` with SHA-256 verification → runner setup). It has
> no KV reuse and **cannot make tool calls**. Do not use it in a multi-turn loop.

### Binary & invocation

```bash
.build/release/TurboFieldfareCLI \
  --model "$HOME/Library/Application Support/TurboFieldfare/gemma4.gturbo" \
  --max-context 4096 \
  --max-new 1024 \
  --temperature 0.2 \
  --quiet
```

Input modes (**mutually exclusive**):

- `--prompt "<string>"` — raw completion, no chat formatting. Use for reproducible
  comparisons.
- `--messages-file <path>` — JSON array of `{"role","content"}` chat messages,
  formatted the same way as the Mac app. Supported roles: `user`, `model`, and
  optional `system`.

```bash
# messages.json
[
  {"role": "user", "content": "Explain chunked prefill in one sentence."}
]

.build/release/TurboFieldfareCLI \
  --model "$HOME/Library/Application Support/TurboFieldfare/gemma4.gturbo" \
  --messages-file messages.json \
  --quiet
```

### CLI generation flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-new <int>` | `1024` | Generated-token limit. The CLI caps it to `max-context - prompt_tokens`. |
| `--max-context <int>` | `4096` | CLI/app default is 4K (the server defaults to 16K). |
| `--temperature <f>` | `0.2` | `0` = greedy. |
| `--top-k <int>` | `64` | `1–256`; **`0` disables** truncation. |
| `--top-p <f>` | `0.95` | `> 0` and `<= 1`. |
| `--repetition-penalty <f>` | `1.0` | Float `> 0`. |
| `--seed <uint64>` | off | Deterministic sampling seed (server has no per-request equivalent advantage). |
| `--stop <string>` | none | Stop substring, **repeatable**. |
| `--quiet` | off | Suppresses the `[stop=… tok/s=…]` timing footer on stderr. |

### CLI sampling constraint (stricter than the server)

The CLI **enforces** what the server only accepts silently: with
`temperature > 0`, if `top_p < 1` then `top_k` must be between `1` and `256`.
To disable both truncation controls, pass `--top-k 0 --top-p 1`.

### Output

Generated text goes to **stdout**; the timing footer goes to **stderr**. In
scripts, capture text with `$(… 2>/dev/null)` or pass `--quiet`.

### When to switch back to the server

For any second request in the same task — especially tool-call loops or
multi-turn chat — start the server (§2) instead. The CLI's per-call model load
makes repeated invocation inefficient, and it cannot emit or resolve tool calls.

