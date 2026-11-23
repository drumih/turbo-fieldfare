# OpenCode provider for TinyTitan

`opencode.json` adds TinyTitanServer to [OpenCode](https://opencode.ai) as the
provider `tinytitan` ("TinyTitan"), with the shipped Ornith 1.5 35B-A3B 4-bit as
the default model. It lists **every model the installer supports** — all
eighteen `install_models.sh` rows, every family and width — so OpenCode shows
the full list even where only one install is on disk; a model whose install is
missing serves a 404 until it is installed. It also lists **Inkling 2 Mini**
(`inkling-2-mini`), which the server runs straight from its GGUF file; see
[Inkling 2 Mini](#inkling-2-mini) below. It is a config file, not a package:
copy it, or merge its `provider` block, into `~/.config/opencode/opencode.json`.
It is not at the repository root on purpose — a root `opencode.json` would
change the default model of everyone who runs OpenCode inside this checkout.

```bash
cp plugins/opencode-tinytitan/opencode.json ~/.config/opencode/opencode.json   # no config yet
.build/release/TinyTitanServer --models-dir models \
  --model models/ornith-1.5_35B_A3B_4Bit --max-context 262144
opencode -m tinytitan/ornith-1.5-35b-a3b_4-Bit
```

`tools/server_launcher.sh --client opencode` does this for you: it merges this
file's `tinytitan` provider into `~/.config/opencode/opencode.jsonc` (or into the
`opencode.json` already there; a new config is a copy of this file), with only the
`baseURL` port and the launched model's `limit.context` set from the server it
starts, then runs `opencode -m tinytitan/<model>`. Your other settings and any
models you added under `tinytitan` are kept, `tinytitan` is taken out of
`disabled_providers` (which hides a provider whatever its block says) in every
global config file and in `$OPENCODE_CONFIG`, and each file it changes is backed
up beside itself first (`*.tinytitan-backup`, then `*.tinytitan-backup-<time>`).
Comments do not survive the rewrite; they stay in the backup.

## What is wired

| Setting | Value | Why |
| --- | --- | --- |
| `baseURL` | `http://127.0.0.1:8080/v1` | The server's default `--port 8080`; change both together |
| Model keys | the ids `GET /v1/models` reports | The server refuses any other name with 404 `model_not_found`; ids always end in `_4-Bit` / `_8-Bit` |
| `limit.context` | 262144 | 256K tokens, the largest native rung and the server's default `--max-context` |
| `limit.output` | 32768 | OpenCode caps `max_tokens` at 32000 itself |
| `tool_call` | true | Function tools with `tool_choice: auto` |
| `temperature` | false | OpenCode sends no `temperature` / `top_p`, so the server applies each model's own sampling defaults |
| `timeout` | false | A long agent prompt on the 125B can take minutes to prefill |

OpenCode cannot list models from a custom provider's `/v1/models`, so the file
lists them. Only installs that `/v1/models` actually reports can be used; to add
another, copy an entry and key it by the id from
`curl -s http://127.0.0.1:8080/v1/models`. Use `--models-dir` so OpenCode can
switch between installs; a `--model`-only server serves exactly one id.

## Thinking

Each variant sends `reasoning_effort`, which the server reads per request
(`docs/server-api.md`, "Reasoning levels"). Pick one in the TUI's variant
selector or with `opencode run --variant <name>`; with no variant the server's
own `--reasoning` (default off) applies.

| Model | Variants | Notes |
| --- | --- | --- |
| Qwen3.8-Flash-Next | `off`, `low`, `medium`, `xhigh` | Exactly the levels its chat template renders; `on` would equal `xhigh` |
| Ornith 1.5, Qwen 3.6, Qwen 3.5 dense | `off`, `on` | Binary `enable_thinking` templates; an effort level maps to `on` |
| Inkling 2 Mini | `off`, `low`, `medium`, `high`, `xhigh` | The template's effort levels; `high` is its default |

## Context beyond 256K

262144 needs no flag: it is the default. Larger windows need YaRN on the server
(`--rope-scaling yarn --max-context 524288|1048576|2097152`) and a matching
`limit.context` here.

## Inkling 2 Mini

The GGUF is not an installer row and is not picked up by `--models-dir`: put the
file in its own folder under `models/` and serve that folder on its own; OpenCode
reaches it at the same address.

```bash
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --local-dir models/inkling-2-mini-gguf
```


```bash
.build/release/TinyTitanServer --model models/inkling-2-mini-gguf
opencode -m tinytitan/inkling-2-mini
```

Its entry declares the same 256K window as the others. The server asks for it and
allocates what the GPU can hold beside the weights, logging the result: every
position costs 32 KB of KV cache, so the 9.1 GB Q2 file gets the full 256K on a
24 GB Mac, while the 17.6 GB Q4 file gets about 26K there — less than OpenCode's
first request (about 27K tokens with its tools), so on a 24 GB Mac use Q2.
