# OpenCode provider for TinyTitan

`opencode.json` adds TinyTitanServer to [OpenCode](https://opencode.ai) as the
provider `tinytitan` ("TinyTitan"), with Qwen3.8-Flash-Next 4-bit as the default
model. It is a config file, not a package: copy it, or merge its `provider` block,
into `~/.config/opencode/opencode.json`. It is not at the repository root on
purpose — a root `opencode.json` would change the default model of everyone who
runs OpenCode inside this checkout.

```bash
cp plugins/opencode-tinytitan/opencode.json ~/.config/opencode/opencode.json   # no config yet
.build/release/TinyTitanServer --models-dir models \
  --model models/qwen3.8-flash-next_125B_A6B_4Bit --max-context 262144
opencode -m tinytitan/qwen3.8-flash-next-mtp_4-Bit
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

## Context beyond 256K

262144 needs no flag: it is the default. Larger windows need YaRN on the server
(`--rope-scaling yarn --max-context 524288|1048576|2097152`) and a matching
`limit.context` here.
