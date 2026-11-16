import TinyTitanCore
import Foundation
import TinyTitan

/// Helpers for TINYTITAN_STRIP_CLI_PROMPT: drop coding-CLI boilerplate (agent
/// system prompts, tool definitions, tool-call history, and in-message
/// <system-reminder> scaffolding) before encoding, keeping only the real
/// user/assistant conversation. Prefill then processes a few hundred tokens
/// instead of the CLIs' multi-thousand-token agent prompt + tool set.
///
/// The structural part (dropping system/developer/tool roles, tools, and
/// assistant tool calls) keys off stable OpenAI wire-format roles and stays
/// correct even if a CLI rewrites its bloat prompt. The content scraper for
/// <system-reminder> blocks is heuristic: it never emits an empty user turn
/// (the real prompt cannot be lost), and every request logs a strip report so
/// a future template change is visible in the server log instead of silently
/// degrading.
enum CLIStrip {
    /// Bump when the heuristic changes so strip reports stay attributable.
    static let version = 2

    struct Stats: Equatable, Sendable {
        var systemDropped = 0
        var developerDropped = 0
        var toolRoleDropped = 0
        var toolsDropped = 0
        var toolCallsDropped = 0
        var reminderCharsRemoved = 0
        var emptyMessageFallbacks = 0
        var emptyRequestFallback = false
    }

    /// TINYTITAN_STRIP_CLI_PROMPT=1 (or "on", "true", "yes") enables stripping.
    static func isEnabled() -> Bool {
        switch ProcessInfo.processInfo.environment["TINYTITAN_STRIP_CLI_PROMPT"]?.lowercased() {
        case "1", "on", "true", "yes": return true
        default: return false
        }
    }

    /// Comma-separated block tag list from TINYTITAN_STRIP_TAGS, defaulting to
    /// ["system-reminder"] (qwen-code's scaffolding). Lets a template rename
    /// be handled by env var instead of a rebuild.
    static func tags() -> [String] {
        guard let raw = ProcessInfo.processInfo.environment["TINYTITAN_STRIP_TAGS"] else {
            return ["system-reminder"]
        }
        let parsed = raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        return parsed.isEmpty ? ["system-reminder"] : parsed
    }

    /// Filter a validated request. Returns the reduced messages (user and
    /// assistant turns only, reminder blocks removed, tool calls cleared),
    /// an empty tool list, and the strip stats for the log report.
    static func filter(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    )
        -> (messages: [GFTokenizer.Message], tools: [GFTokenizer.FunctionDefinition], stats: Stats)
    {
        var stats = Stats()
        stats.toolsDropped = tools.count
        let tags = tags()
        let stripped = messages.compactMap { message -> GFTokenizer.Message? in
            switch message.role {
            case .system:
                stats.systemDropped += 1
                return nil
            case .developer:
                stats.developerDropped += 1
                return nil
            case .tool:
                stats.toolRoleDropped += 1
                return nil
            case .user, .assistant:
                break
            }
            stats.toolCallsDropped += message.toolCalls.count
            let content: String?
            if let text = message.content {
                let cleaned = stripReminderBlocks(text, tags: tags)
                stats.reminderCharsRemoved += text.count - cleaned.count
                if cleaned.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    // Per-message guard: never hand the model an empty turn;
                    // keep the original content instead (slow but correct).
                    stats.emptyMessageFallbacks += 1
                    content = message.content
                } else {
                    content = cleaned
                }
            } else {
                content = nil
            }
            if message.role == .assistant,
                content?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true
            {
                // An assistant turn whose only content was tool calls now
                // renders as an empty turn; drop it instead of emitting a
                // hollow <|im_start|>assistant block.
                return nil
            }
            return GFTokenizer.Message(
                role: message.role,
                content: content,
                toolCalls: [],
                toolCallID: nil,
                name: message.name)
        }
        guard !stripped.isEmpty else {
            // Request-wide guard: a request that was only system/developer
            // guidance renders unchanged rather than as an empty prompt.
            stats.emptyRequestFallback = true
            return (messages, [], stats)
        }
        return (stripped, [], stats)
    }

    /// Remove <tag>...</tag> blocks (e.g. qwen-code's <system-reminder>) from
    /// text. The real user prompt is the surrounding text.
    static func stripReminderBlocks(_ text: String, tags: [String]) -> String {
        var result = ""
        var remaining = text[...]
        while let (tag, open) = nextBlockStart(remaining, tags: tags) {
            result += remaining[..<open.lowerBound]
            remaining = remaining[open.upperBound...]
            let closeMarker = "</\(tag)>"
            guard let close = remaining.range(of: closeMarker, options: .caseInsensitive) else {
                // Unterminated block: keep the remainder verbatim rather than
                // silently dropping the real user prompt.
                result += remaining
                return result
            }
            remaining = remaining[close.upperBound...]
        }
        result += remaining
        return result
    }

    /// Earliest <tag> block start among the configured tags.
    private static func nextBlockStart(_ text: Substring, tags: [String])
        -> (tag: String, range: Range<String.Index>)?
    {
        var best: (tag: String, range: Range<String.Index>)?
        for tag in tags {
            let marker = "<\(tag)>"
            if let range = text.range(of: marker, options: .caseInsensitive),
                best.map({ range.lowerBound < $0.range.lowerBound }) ?? true
            {
                best = (tag, range)
            }
        }
        return best
    }
}
