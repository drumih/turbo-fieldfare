# OpenCode request fixtures

Sanitized request bodies captured on 2026-07-23 from `opencode-ai@1.15.11`.
That source release pins `ai` 6.0.168,
`@ai-sdk/openai-compatible` 2.0.41, and `@ai-sdk/provider` 3.0.8.

- `opencode-1.15.11-initial.json`: 11,521-byte initial streamed request.
- `opencode-1.15.11-tool-result.json`: 12,153-byte follow-up containing the
  fake server's assistant `read` call and OpenCode's tool result.

Temporary paths and call IDs are deterministic and sanitized. The fixtures
contain no credentials or user repository content.
