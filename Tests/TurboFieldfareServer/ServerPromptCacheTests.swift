import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func capturedOpenCodeToolResultUsesFrozenToolBoundary() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = try validatedFixture("opencode-1.15.11-initial.json")
        let continuation = try validatedFixture("opencode-1.15.11-tool-result.json")
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: initial.messages,
            tools: initial.tools)
        let assistant = continuation.messages[initial.messages.count]
        let prefix = try tokenizer.encodeToolChat(
            messages: initial.messages + [assistant],
            tools: initial.tools)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let generatedCall = Array(prefix[callStart...callEnd])
        let kvBacked = initialPrompt + generatedCall
        let historicalCall = try #require(assistant.toolCalls.first)
        let parsedCall = ParsedToolCall(
            id: historicalCall.id,
            name: historicalCall.name,
            arguments: historicalCall.arguments,
            argumentsJSON: try historicalCall.arguments.encoded())
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "",
            calls: [parsedCall],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))
        let rendered = try tokenizer.encodeToolChat(
            messages: continuation.messages,
            tools: continuation.tools)

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected captured OpenCode tool-result hit")
            return
        }
        let bridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: initial.messages,
            assistant: assistant,
            incomingMessages: continuation.messages,
            tools: continuation.tools)
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(bridge.first == tokenizer.toolResponseID)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            #expect(cache.entry == nil)
        }

        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + tokenizer.encode("answer", addBOS: false),
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            GFTokenizer.Message(role: .user, content: "changed"),
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        #expect(cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) == .miss)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer ",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        #expect(cache.entry == nil)
    }

    /// A verbatim-repeated tool call must keep its cache; the unique-match
    /// rule refused this shape and dropped the cache from that turn on.
    @Test func repeatedIdenticalToolCallKeepsTheBoundaryResolvable() async throws {
        let tokenizer = try await GFTokenizer.load()
        let tools = [GFTokenizer.FunctionDefinition(
            name: "ls",
            description: "list a directory",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "path": .object(["type": .string("string")]),
                ]),
            ]))]
        let call = { (id: String) in
            GFTokenizer.HistoricalToolCall(
                id: id,
                name: "ls",
                arguments: .object(["path": .string(".")]))
        }
        let first = GFTokenizer.Message(
            role: .assistant, content: nil, toolCalls: [call("call_1")])
        let firstResult = GFTokenizer.Message(
            role: .tool, content: "pkg/", toolCallID: "call_1", name: "ls")
        // Byte-identical to the first call; only the call id differs.
        let repeated = GFTokenizer.Message(
            role: .assistant, content: nil, toolCalls: [call("call_2")])
        let repeatedResult = GFTokenizer.Message(
            role: .tool, content: "pkg/", toolCallID: "call_2", name: "ls")
        let cachedMessages = [
            GFTokenizer.Message(role: .user, content: "list the tree"),
            first,
            firstResult,
        ]

        let bridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: cachedMessages,
            assistant: repeated,
            incomingMessages: cachedMessages + [repeated, repeatedResult],
            tools: tools)

        // Both twins are followed by a response; only the counts tell a
        // twin-anchored bridge apart.
        #expect(bridge.first == tokenizer.toolResponseID)
        #expect(bridge.filter { $0 == tokenizer.toolResponseID }.count == 1)
        #expect(!bridge.contains(tokenizer.toolCallStartID))

        // A diverged cached history cannot anchor by position and still refuses.
        let diverged = [GFTokenizer.Message(role: .user, content: "list the other tree")]
            + cachedMessages.dropFirst()
        #expect(throws: (any Error).self) {
            try tokenizer.encodeToolResultContinuation(
                cachedMessages: diverged,
                assistant: repeated,
                incomingMessages: cachedMessages + [repeated, repeatedResult],
                tools: tools)
        }
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }

    private func validatedFixture(_ name: String) throws -> ValidatedChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Fixtures"))
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(contentsOf: url))
        return try OpenAIRequestValidator.validate(
            request,
            modelID: "gemma-4-26b-a4b-it")
    }
}
