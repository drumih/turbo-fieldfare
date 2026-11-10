# TurboFieldfare: AI Installation, Agent Skill & API Usage Guide

This guide gives an autonomous coding agent everything it needs to install **TurboFieldfare** (Gemma 4 26B-A4B IT) in a single run, then documents the local API server, token budgeting, parameter mapping, and the bundled agent skill.

---

## 📋 Table of Contents

1. [Automated AI Agent Prompt](#1-automated-ai-agent-prompt)
2. [Local OpenAI-Compatible API Server](#2-local-openai-compatible-api-server)
3. [Token Budgeting & Context Window Boundaries](#3-token-budgeting-context-window-boundaries)
4. [GUI Sidebar Settings vs. API Parameter Mapping](#4-gui-sidebar-settings-vs-api-parameter-mapping)
5. [One Model Process at a Time](#5-one-model-process-at-a-time)
6. [Agent Skill Specification (`turbofieldfare-api`)](#6-agent-skill-specification-turbofieldfare-api)
- [Appendix A. Build Artifacts](#appendix-a-build-artifacts)
- [Appendix B. Model Path Notes](#appendix-b-model-path-notes)

---

## 1. Automated AI Agent Prompt

Copy and pass the prompt below to an autonomous coding agent (Antigravity, Claude, Codex, etc.) to perform complete installation, app bundling, and skill registration in a single run. It is **self-contained** — it does not require the rest of this guide.

```text
Install TurboFieldfare in ~/fork/turbo-fieldfare, package it as a native macOS application in ~/Applications/TurboFieldfare.app, configure local API server persistence, and register agent skills.

[Background]
- TurboFieldfare runs Gemma 4 26B-A4B IT on Apple Silicon (macOS 26+, Swift 6.2+, Metal 4).
- A release build produces five binaries in .build/release/:
    TurboFieldfareMac          — native macOS GUI app (loads its own decode service)
    TurboFieldfareDecodeService — backing Metal decode engine (sibling of the Mac app)
    TurboFieldfareServer        — OpenAI-compatible local API server
    TurboFieldfareCLI           — command-line interface
    TurboFieldfareRepack        — streaming model installer and install verifier
- Only ONE model-owning process may run at a time (app, CLI, or server are mutually exclusive).

[Steps to Execute]
1. Clone & Build:
   - Clone https://github.com/drumih/turbo-fieldfare.git into ~/fork/turbo-fieldfare
   - Execute: swift build -c release

2. Model Download & Repack:
   - Launch .build/release/TurboFieldfareMac and monitor the ~15 GB Gemma 4 26B-A4B IT model download, streaming repack, and hash verification to complete. The completed model lands at scratch/gemma4.gturbo (about 14.3 GB).

3. macOS App Packaging & Path Fix (only if running the packaged .app from outside the checkout):
   - Create the macOS app bundle at ~/Applications/TurboFieldfare.app with Info.plist, AppIcon.icns, and the five compiled binaries.
   - Copy the verified model directory to ~/Library/Application Support/TurboFieldfare/gemma4.gturbo and update modelDirectoryPath in verified-install.json so launching ~/Applications/TurboFieldfare.app initializes into the Ready state instantly without re-downloading.
   - Note: when launched from the checkout, the app/CLI/server read scratch/gemma4.gturbo directly — no copy is needed.

4. API Server & Agent Skill Registration:
   - Start TurboFieldfareServer on loopback 127.0.0.1, port 52642 (http://127.0.0.1:52642/v1/chat/completions). The server's code default port is 8080, but use 52642 to avoid collisions: 8080/3000/5000 are heavily used by dev servers, while 52642 is in the macOS ephemeral range (49152–65535) with no IANA-registered service. The server binds to loopback with no auth or TLS, so never proxy, tunnel, or expose it.
   - Create the turbofieldfare-api agent skill in ~/.agents/skills, ~/.claude/skills, and ~/.codex/skills.
   - Register TurboFieldfare as a provider in OpenCode (opencode.json): use the @ai-sdk/openai-compatible npm package with baseURL http://127.0.0.1:52642/v1 and apiKey "local" (the server ignores the key). Add a model entry named gemma-4-26b-a4b-it with limit.context 16384 and limit.output 8192, and set temperature/topP/topK defaults under options (0.2/0.95/64). Set the top-level model key to turbofieldfare/gemma-4-26b-a4b-it so OpenCode picks it by default. The limit values are required — without them OpenCode does not know the model's token budget and may fail to generate.

5. Final Verification:
   - Perform text generation tests via both GUI and API, confirm successful output, and output summary. Run only ONE of the GUI app or the server at a time.
```

---

## 2. Local OpenAI-Compatible API Server

`TurboFieldfareServer` runs a lightweight loopback HTTP server. It binds to `127.0.0.1` with **no authentication and no TLS** — keep it on loopback and never proxy, tunnel, or expose it. The server's *code* default port is `8080`, but this guide uses **`52642`** to avoid collisions with heavily-used dev ports (`8080`, `3000`, `5000`, `8000`); `52642` sits in the macOS ephemeral range (`49152–65535`), which has no IANA-registered service. (Caveat: the OS can hand ephemeral ports to transient sockets — if a bind fails, retry or pick another value in the range.) Override the port with any `1–65535` value via `--port`. Before starting, confirm no other TurboFieldfare model process is running:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If it prints a match, do not start the server — only one model-owning process may run at a time.

### Starting the API Server

```bash
.build/release/TurboFieldfareServer \
  --model "$HOME/Library/Application Support/TurboFieldfare/gemma4.gturbo" \
  --port 52642 \
  --max-context 16384
```

Single-prefix KV reuse is **on by default** (`--prompt-cache-mode single-prefix`),
so the flag is shown for clarity but can be omitted. Other server options include
`--queue-limit <count>` (default `4`), `--model-id <id>` (default
`gemma-4-26b-a4b-it`), and `--prompt-cache-mode off` to disable prefix reuse.
Wait for `TurboFieldfareServer ready` before sending requests; the model loads
before the port opens.

### Stopping the API Server

Stop the server from the terminal that launched it with `Control-C`. Only stop a
server you started.

### Connect a client

The base URL is `http://127.0.0.1:52642/v1`. Some client libraries require an
API key, but the server ignores it — pass any non-empty value.

**OpenCode** (`opencode.json`) — the maintainer's reference client setup:

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

Set the top-level `model` to `turbofieldfare/gemma-4-26b-a4b-it` so OpenCode selects
it by default. The `limit` block is required — without it OpenCode does not know the
model's token budget and may fail to generate. The `options` block sets sane sampling
defaults (match the GUI defaults of temperature 0.2, Top-K 64, Top-P 0.95).

> **Two practical OpenCode limits.** (1) Pass `--title <name>` to `opencode run` —
> otherwise OpenCode's auto session-title request collides with the main request and
> the server returns `generation queue is full` (it serves one generation at a time).
> (2) OpenCode's agent context (system prompt + skills + tool schemas) is large, so a
> turn takes minutes on Gemma 4. For plain chat, use the Python/cURL path above. See
> the skill's OpenCode section for full details.

**Python** (openai SDK):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52642/v1", api_key="local")
response = client.chat.completions.create(
    model="gemma-4-26b-a4b-it",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

---

## 3. Token Budgeting & Context Window Boundaries

In LLM architectures, total context memory equals the sum of prompt tokens (In) and generated tokens (Out):

$$\text{Total Tokens} = \text{Prompt Tokens (In)} + \text{Completion Tokens (Out)}$$

### Context Boundary Rules
1. **Server Context Limit (`--max-context`)**:
   - Supported values: `4096`, `8192`, `16384` (Server Default), `32768` (32K), `65536` (64K). (Note: the Mac app / CLI default to 4K; the server defaults to 16K.)
   - If a prompt exceeds the server's `--max-context`, `TurboFieldfareServer` returns a `400 Invalid Request` error (`prompt exceeds the configured context`).
2. **Output Token Cap (`max_completion_tokens`)**:
   - Agents MUST ensure: `max_completion_tokens <= max_context - prompt_tokens`.
3. **KV Cache Efficiency (`prompt_tokens_details.cached_tokens`)**:
   - `TurboFieldfareServer` enables single-prefix KV caching by default (`--prompt-cache-mode single-prefix`). When conversation prefixes match, prefill is skipped for cached tokens, significantly reducing TTFT (Time-To-First-Token). Disable with `--prompt-cache-mode off`.

---

## 4. GUI Sidebar Settings vs. API Parameter Mapping

| GUI Sidebar Setting | Scope | API / CLI Equivalent | Supported Values |
| --- | --- | --- | --- |
| **Context** | Memory / Server | Server CLI: `--max-context` | `4096`, `8192`, `16384`, `32768`, `65536` (Server default `16384`) |
| **Expert-cache Slots** | Memory / App only | Mac app only — **no server/CLI flag** | `8`, `16`, `24`, `32` (Default `16`) |
| **Temperature** | Generation / Request | Request JSON: `"temperature"` | `0.0` to `2.0` (Default: `0.2`) |
| **Top-K** | Generation / Request | Request JSON: `"top_k"` | `1` to `256` (Default: `64`) |
| **Top-P** | Generation / Request | Request JSON: `"top_p"` | `> 0` to `1.0` (Default: `0.95`) |
| **Repetition Penalty** | Generation / Request | Request JSON: `"repetition_penalty"` | Float > 0 (Default: `1.0`) |
| **Prompt prefill / KV Cache** | Runtime / Server | Server CLI: `--prompt-cache-mode` | `off`, `single-prefix` (Default: `single-prefix`) |

> **Sampling constraints (server):** `top_p` must be `> 0` and `<= 1`; `top_k`
> must be `1–256`. At `temperature 0` the server decodes greedily, so `top_p`
> and `top_k` have no sampling effect. `expert-cache slots` is a Mac-app runtime
> control only — the server and CLI do not expose a `--slots` flag.

---

## 5. One Model Process at a Time

> ⚠️ **Do not run the GUI app and the API server at the same time.** Each one
> loads its own copy of the model, so running both creates two model-owning
> processes.

TurboFieldfare is designed to keep memory low (~1.35 GB of core weights in RAM,
experts streamed from SSD), but the Mac app (`TurboFieldfare.app` with its
`TurboFieldfareDecodeService`) and the API server (`TurboFieldfareServer`) are
**separate model-owning processes**. Run only one of them at a time. Before
starting either, confirm no other TurboFieldfare process is running:

```bash
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If it prints a match, do not start a second process. Close the other product
before switching.

---

## 6. Agent Skill Specification (`turbofieldfare-api`)

This repository includes a pre-configured Agent Skill specification at [`skills/turbofieldfare-api/SKILL.md`](../skills/turbofieldfare-api/SKILL.md). It covers server lifecycle, token budgeting, task-specific parameter presets, and a CLI fallback.

---

## Appendix A. Build Artifacts

A release build (`swift build -c release`) from the repo root generates five binaries in `.build/release/`: `TurboFieldfareMac` (GUI app), `TurboFieldfareDecodeService` (Metal decode engine), `TurboFieldfareServer` (OpenAI-compatible API server), `TurboFieldfareCLI` (CLI), and `TurboFieldfareRepack` (streaming installer/verifier).

---

## Appendix B. Model Path Notes

When launched from the repository checkout, the app, CLI, and server read the relative `scratch/gemma4.gturbo` path directly — no copy is needed. Only the packaged `TurboFieldfare.app` run from outside the checkout (e.g. `~/Applications`) resolves its model from `~/Library/Application Support/TurboFieldfare/gemma4.gturbo`; to avoid a re-download prompt, copy the verified model there and set `modelDirectoryPath` in `verified-install.json`.
