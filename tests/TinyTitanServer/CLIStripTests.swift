import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

@Suite("CLI prompt strip")
struct CLIStripTests {
    // MARK: - reminder block scraping

    @Test func stripsSingleReminderBlockKeepingPrompt() {
        let input =
            "<system-reminder>\nskills list\n</system-reminder>\nWhat is the capital of France?"
        let out = CLIStrip.stripReminderBlocks(input, tags: ["system-reminder"])
        #expect(out == "\nWhat is the capital of France?")
    }

    @Test func stripsMultipleReminderBlocks() {
        let input = """
            <system-reminder>skills</system-reminder>
            <system-reminder>context</system-reminder>
            <system-reminder>date</system-reminder>
            What is the capital of France?
            """
        let out = CLIStrip.stripReminderBlocks(input, tags: ["system-reminder"])
        #expect(out.contains("What is the capital of France?"))
        #expect(!out.contains("skills"))
        #expect(!out.contains("context"))
        #expect(!out.contains("date"))
    }

    @Test func unterminatedBlockKeepsRemainder() {
        let input = "<system-reminder>\nskills\nWhat is the capital of France?"
        let out = CLIStrip.stripReminderBlocks(input, tags: ["system-reminder"])
        // An unterminated block keeps the remainder verbatim so the real user
        // prompt is never dropped.
        #expect(out == "\nskills\nWhat is the capital of France?")
    }

    @Test func unrecognizedTagLeavesTextUntouched() {
        let input = "<system-note>skills</system-note>\nWhat is the capital of France?"
        let out = CLIStrip.stripReminderBlocks(input, tags: ["system-reminder"])
        #expect(out == input)
    }

    @Test func customTagListStripsRenamedBlocks() {
        let input = "<system-note>skills</system-note>\nWhat is the capital of France?"
        let out = CLIStrip.stripReminderBlocks(input, tags: ["system-reminder", "system-note"])
        #expect(out == "\nWhat is the capital of France?")
    }

    // MARK: - full request filter

    /// Synthetic qwen-code-style request: agent system prompt, a user message
    /// whose content is mostly <system-reminder> scaffolding plus the real
    /// prompt, and 59 tool definitions.
    @Test func syntheticQwenRequestStripsToRealConversation() throws {
        let system = GFTokenizer.Message(
            role: .system, content: "You are Qwen Code, a non-interactive CLI agent.")
        let user = GFTokenizer.Message(
            role: .user,
            content: """
                <system-reminder>skills list</system-reminder>
                <system-reminder>workspace snapshot</system-reminder>
                <system-reminder>current date</system-reminder>
                What is the capital of France?
                """)
        let tools = (0..<59).map { i in
            GFTokenizer.FunctionDefinition(
                name: "tool_\(i)",
                description: "tool \(i)",
                parameters: JSONValue.object([:]))
        }
        let result = CLIStrip.filter(messages: [system, user], tools: tools)

        #expect(result.messages.count == 1)
        #expect(result.messages[0].role == .user)
        #expect(result.messages[0].content == "\n\n\nWhat is the capital of France?")
        #expect(result.tools.isEmpty)
        #expect(result.stats.systemDropped == 1)
        #expect(result.stats.toolsDropped == 59)
        #expect(result.stats.reminderCharsRemoved > 0)
        #expect(result.stats.emptyMessageFallbacks == 0)
        #expect(result.stats.emptyRequestFallback == false)
    }

    @Test func stripsAssistantToolCallsAndToolRoleHistory() throws {
        let assistantCall = GFTokenizer.Message(
            role: .assistant,
            content: nil,
            toolCalls: [
                GFTokenizer.HistoricalToolCall(
                    id: "call_1",
                    name: "read_file",
                    arguments: JSONValue.object(["path": .string("a.txt")]))
            ])
        let toolResult = GFTokenizer.Message(
            role: .tool,
            content: "file contents",
            toolCallID: "call_1")
        let user = GFTokenizer.Message(role: .user, content: "Continue")
        let result = CLIStrip.filter(messages: [user, assistantCall, toolResult], tools: [])

        // The assistant turn carried only a tool call; once cleared it renders
        // as an empty turn and is dropped, and the tool result is dropped too.
        #expect(result.messages.map(\.role) == [.user])
        #expect(result.stats.toolRoleDropped == 1)
        #expect(result.stats.toolCallsDropped == 1)
    }

    // MARK: - fail-safe guards

    @Test func userMessageMadeEmptyByScrapingFallsBackToOriginal() throws {
        // The entire user message is a reminder block; stripping would leave
        // an empty turn, so the guard keeps the original content verbatim.
        let user = GFTokenizer.Message(
            role: .user,
            content: "<system-reminder>\nall junk\n</system-reminder>")
        let result = CLIStrip.filter(messages: [user], tools: [])

        #expect(result.messages.count == 1)
        #expect(result.messages[0].content == "<system-reminder>\nall junk\n</system-reminder>")
        #expect(result.stats.emptyMessageFallbacks == 1)
    }

    @Test func systemOnlyRequestFallsBackToOriginal() throws {
        let system = GFTokenizer.Message(role: .system, content: "guidance only")
        let result = CLIStrip.filter(messages: [system], tools: [])

        #expect(result.messages.count == 1)
        #expect(result.messages[0].role == .system)
        #expect(result.stats.emptyRequestFallback == true)
    }

    // MARK: - env configuration

    @Test func defaultTagIsSystemReminder() {
        setenv("TINYTITAN_STRIP_TAGS", "", 1)
        defer { unsetenv("TINYTITAN_STRIP_TAGS") }
        #expect(CLIStrip.tags() == ["system-reminder"])
    }

    @Test func envTagListParsesCommaSeparated() {
        setenv("TINYTITAN_STRIP_TAGS", "system-reminder, system-note ,foo", 1)
        defer { unsetenv("TINYTITAN_STRIP_TAGS") }
        #expect(CLIStrip.tags() == ["system-reminder", "system-note", "foo"])
    }

    @Test func stripDisabledByDefault() {
        unsetenv("TINYTITAN_STRIP_CLI_PROMPT")
        #expect(CLIStrip.isEnabled() == false)
        setenv("TINYTITAN_STRIP_CLI_PROMPT", "1", 1)
        #expect(CLIStrip.isEnabled() == true)
        setenv("TINYTITAN_STRIP_CLI_PROMPT", "on", 1)
        #expect(CLIStrip.isEnabled() == true)
        setenv("TINYTITAN_STRIP_CLI_PROMPT", "0", 1)
        #expect(CLIStrip.isEnabled() == false)
        unsetenv("TINYTITAN_STRIP_CLI_PROMPT")
    }
}
