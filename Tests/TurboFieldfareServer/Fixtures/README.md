# Client request fixtures

## OpenCode

Sanitized request bodies captured on 2026-07-23 from `opencode-ai@1.15.11`.
That source release pins `ai` 6.0.168,
`@ai-sdk/openai-compatible` 2.0.41, and `@ai-sdk/provider` 3.0.8.

- `opencode-1.15.11-initial.json`: 11,521-byte initial streamed request.
- `opencode-1.15.11-tool-result.json`: 12,153-byte follow-up containing the
  fake server's assistant `read` call and OpenCode's tool result.

## DeepSeek Harness

Sanitized request bodies captured on 2026-08-21 from
`@deepseek-ai/dsh@0.1.1-rc.1` (headless profile, Node v24.19.0) driven against
the compatibility harness through a logging proxy. dsh's LLM layer is
`@earendil-works/pi-ai` over `openai-completions`.

- `dsh-0.1.1-rc.1-initial.json`: initial streamed agent request with all 25
  default tools. The `workflow` tool's `args` parameter is a bare
  `{type: object}` node carrying `additionalProperties` — the schema shape
  that crashed chat-template rendering (PR 138).
- `dsh-0.1.1-rc.1-tool-result.json`: follow-up containing the fake server's
  assistant `bash` call and dsh's tool result.

The workspace path is sanitized to `/tmp/dsh-workspace`.

Temporary paths and call IDs are deterministic and sanitized. The fixtures
contain no credentials or user repository content.
