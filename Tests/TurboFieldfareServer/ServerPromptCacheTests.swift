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

    /// A model that repeats a tool call verbatim must keep its cache.
    ///
    /// The repeat renders to a token sequence the history already contains, so
    /// the old unique-match rule refused the bridge and the conversation
    /// stayed uncached from the first verbatim repeat on; the boundary is
    /// resolved by position instead. A bridge that cannot render is still a
    /// named condition rather than a swallowed one — the divergent-history
    /// shape at the end still refuses — but with message-equal leading history
    /// the pinned render is deterministic, so `.bridgeRenderFailed` inside
    /// `match` is defense in depth rather than a reachable outcome.
    @Test func aRepeatedToolCallResolvesByPositionAndHits() async throws {
        let tokenizer = try await GFTokenizer.load()
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_repeat", name: "read", arguments: .object(["path": .string("a.txt")]))
        let tools = [GFTokenizer.FunctionDefinition(
            name: "read",
            description: "Read a file.",
            parameters: .object(["type": .string("object")]))]
        let earlier: [GFTokenizer.Message] = [
            GFTokenizer.Message(role: .user, content: "read a.txt"),
            GFTokenizer.Message(role: .assistant, content: nil, toolCalls: [call]),
            GFTokenizer.Message(role: .tool, content: "contents", toolCallID: call.id),
            GFTokenizer.Message(role: .user, content: "read it again"),
        ]
        let cached = request(messages: earlier, tools: tools)
        let assistant = GFTokenizer.Message(
            role: .assistant, content: nil, toolCalls: [call])
        let prefix = try tokenizer.encodeToolChat(
            messages: earlier + [assistant], tools: tools)
        // The KV holds what the model generated, not a fresh render of it, so
        // the cheap prefix path cannot match and the bridge is what decides.
        let initialPrompt = try tokenizer.encodeToolChat(
            messages: earlier, tools: tools)
        let callStart = try #require(prefix.lastIndex(of: tokenizer.toolCallStartID))
        let callEnd = try #require(prefix.lastIndex(of: tokenizer.toolCallEndID))
        let kvBacked = initialPrompt + Array(prefix[callStart...callEnd])
        let parsed = ParsedToolCall(
            id: call.id, name: call.name, arguments: call.arguments,
            argumentsJSON: try call.arguments.encoded())
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: cached,
            content: "",
            calls: [parsed],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.toolResponseID,
                reason: .toolCalls))

        let continuation = request(
            messages: earlier + [
                assistant,
                GFTokenizer.Message(
                    role: .tool, content: "contents", toolCallID: call.id),
            ],
            tools: tools)
        let bridge = try tokenizer.encodeToolResultContinuation(
            cachedMessages: earlier,
            assistant: assistant,
            incomingMessages: continuation.messages,
            tools: tools)
        // The bridge continues from the cached repeat, not its earlier
        // identical twin. Both twins are followed by a tool response, so the
        // first-token check alone cannot tell them apart; a twin-anchored
        // bridge would replay the first response and the repeated call turn.
        #expect(bridge.first == tokenizer.toolResponseID)
        #expect(bridge.filter { $0 == tokenizer.toolResponseID }.count == 1)
        #expect(!bridge.contains(tokenizer.toolCallStartID))

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: try tokenizer.encodeToolChat(
                messages: continuation.messages, tools: tools),
            tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected repeated-call tool-result hit")
            return
        }
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)

        // A cached history that diverged from the incoming one cannot anchor
        // the repeated sequence by position, and the ambiguous fallback still
        // refuses rather than guessing a twin.
        let diverged = [GFTokenizer.Message(role: .user, content: "read b.txt")]
            + earlier.dropFirst()
        #expect(throws: (any Error).self) {
            try tokenizer.encodeToolResultContinuation(
                cachedMessages: diverged,
                assistant: assistant,
                incomingMessages: continuation.messages,
                tools: tools)
        }
    }

    /// The prompt-cache domain used to cover six configuration fields and
    /// nothing else, while roughly nineteen `TURBO_FIELDFARE_VISION_*` switches
    /// select kernels - one of them set to `0` picks a different attention path
    /// and costs 55% of the encode. A prefix built under one selection could be
    /// resumed under another with nothing recording the change.
    @Test func runtimeIdentityCoversTheSwitchesThatSelectKernels() {
        let runtime = RuntimeConfiguration()
        let plain = ServerModelSession.runtimeIdentityString(
            runtime: runtime, environment: [:])
        let switched = ServerModelSession.runtimeIdentityString(
            runtime: runtime,
            environment: ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "0"])
        #expect(plain != switched, "a kernel switch did not change cache identity")

        // Unrelated variables stay out of it, or every shell would invalidate.
        #expect(ServerModelSession.runtimeIdentityString(
            runtime: runtime, environment: ["PATH": "/usr/bin"]) == plain)

        // Stable for the same input, and order-independent.
        #expect(ServerModelSession.runtimeIdentityString(
            runtime: runtime,
            environment: ["TURBO_FIELDFARE_B": "1", "TURBO_FIELDFARE_A": "1"])
            == ServerModelSession.runtimeIdentityString(
                runtime: runtime,
                environment: ["TURBO_FIELDFARE_A": "1", "TURBO_FIELDFARE_B": "1"]))

        // And the value matters, not just the name.
        #expect(ServerModelSession.runtimeIdentityString(
            runtime: runtime, environment: ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1"])
            != switched)
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
        // A changed first message is a diverged history, and the miss now says
        // so rather than arriving as an anonymous one.
        #expect(cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer).missReason == .historyDiverged)
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

    /// Why `ServerModelSession.generate` must never publish a multimodal turn.
    ///
    /// This cache identifies a prefix by its flattened message text, and image
    /// parts contribute nothing to that. An image turn's KV holds rows built
    /// from projected image features, but its cached `inputMessages` are just
    /// the text. A later text-only continuation of that conversation — a client
    /// that resends the history without the `image_url` parts — then matches,
    /// and resumes from a prefix whose content it never sent.
    ///
    /// This publishes an image turn directly, bypassing the guard, to show the
    /// cache handing that prefix over. It fails the day the cache learns image
    /// identity, at which point the guard can be relaxed deliberately rather
    /// than by accident.
    @Test func anImageBuiltPrefixWouldBeHandedToATextContinuation() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache()
        let opening = [GFTokenizer.Message(role: .user, content: "describe this")]

        let imageTurn = ValidatedChatRequest(
            messages: opening,
            multimodalMessages: [MultimodalMessage(
                role: .user,
                content: [.text("describe this"), .image(id: UUID())])],
            imageFiles: [UUID(): URL(fileURLWithPath: "/dev/null")],
            imageIdentities: [["sha-of-the-image"]],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
        // The condition the session's guard tests.
        #expect(imageTurn.multimodalMessages != nil)

        let prefix = tokenizer.encode(
            try tokenizer.applyChatTemplate(opening), addBOS: false)
        cache.publish(
            domain: domain,
            request: imageTurn,
            content: "a description",
            calls: [],
            result: rawResult(prompt: prefix,
                              kvBacked: prefix + tokenizer.encode("a description", addBOS: false),
                              boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn))

        // The same conversation, continued, with the image part dropped.
        let continuation = request(messages: opening + [
            GFTokenizer.Message(role: .assistant, content: "a description"),
            GFTokenizer.Message(role: .user, content: "and now?"),
        ])
        #expect(continuation.multimodalMessages == nil)

        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages), addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)
        // Refused by identity now, not by a blanket rule: the entry remembers
        // which image built it, so a client that drops the image parts cannot
        // resume onto a KV holding projected features it never sent. This is
        // what replaced the publish guard in `ServerModelSession.generate`.
        #expect(match.missReason == .imagesDiverged,
                "an image-built prefix was handed to a text continuation")
    }

    /// Image parts contribute nothing to a message's flattened text, so two
    /// prompts differing only in their image compare equal as messages. The
    /// cache must separate them on image content hash.
    @Test func differentImageWithIdenticalTextDoesNotReuseThePrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages = [GFTokenizer.Message(role: .user, content: "describe")]
        let initial = request(messages: messages, imageIdentities: [["image-a"]])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain, request: initial, content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        let follow = messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "and now"),
        ]
        let sameImage = cache.match(
            domain: domain,
            request: request(messages: follow,
                             imageIdentities: [["image-a"], [], []]),
            renderedPromptIDs: nil, tokenizer: tokenizer)
        guard case .hit = sameImage else {
            Issue.record("the same image must continue from the cached prefix")
            return
        }

        let otherImage = cache.match(
            domain: domain,
            request: request(messages: follow,
                             imageIdentities: [["image-b"], [], []]),
            renderedPromptIDs: nil, tokenizer: tokenizer)
        #expect(otherImage.missReason == .imagesDiverged)

        let noImage = cache.match(
            domain: domain,
            request: request(messages: follow, imageIdentities: [[], [], []]),
            renderedPromptIDs: nil, tokenizer: tokenizer)
        #expect(noImage.missReason == .imagesDiverged)
    }

    /// A continuation carrying its own image cannot use the text bridge, but the
    /// cached prefix is still valid, so it resumes after a render instead of
    /// being discarded.
    @Test func continuationCarryingItsOwnImageResumesAfterRender() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages = [GFTokenizer.Message(role: .user, content: "describe")]
        let initial = request(messages: messages, imageIdentities: [["image-a"]])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain, request: initial, content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        let follow = messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "and this one"),
        ]
        let match = cache.match(
            domain: domain,
            request: request(messages: follow,
                             imageIdentities: [["image-a"], [], ["image-b"]]),
            renderedPromptIDs: nil, tokenizer: tokenizer)
        // The history and its images match, so this resumes via a render
        // rather than throwing the cached prefix away.
        #expect(match == .renderThenResume(cachedPromptTokens: kvBacked.count))
    }

    // MARK: - Image identity revalidation

    /// Publishing after each turn must keep an image conversation reusable
    /// indefinitely, not only for the first follow-up.
    @Test func imageConversationKeepsReusingAcrossManyTurns() async throws {
        let tokenizer = try await GFTokenizer.load()
        var messages = [GFTokenizer.Message(role: .user, content: "describe")]
        var identities: [[String]] = [["image-a"]]
        var cache = ServerPromptCache()

        var prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        var kvBacked = prompt + tokenizer.encode("answer 0", addBOS: false)
        cache.publish(
            domain: domain,
            request: request(messages: messages, imageIdentities: identities),
            content: "answer 0", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        for turn in 1...5 {
            messages.append(GFTokenizer.Message(
                role: .assistant, content: "answer \(turn - 1)"))
            messages.append(GFTokenizer.Message(
                role: .user, content: "follow up \(turn)"))
            identities.append([])
            identities.append([])
            let follow = request(messages: messages, imageIdentities: identities)
            let match = cache.match(
                domain: domain, request: follow,
                renderedPromptIDs: nil, tokenizer: tokenizer)
            guard case .hit(let effective, let cached) = match else {
                Issue.record("turn \(turn) lost the cached image prefix")
                return
            }
            #expect(cached == kvBacked.count)
            prompt = effective
            kvBacked = effective + tokenizer.encode("answer \(turn)", addBOS: false)
            cache.publish(
                domain: domain, request: follow,
                content: "answer \(turn)", calls: [],
                result: rawResult(prompt: prompt, kvBacked: kvBacked,
                                  boundary: tokenizer.endOfTurnID,
                                  reason: .endOfTurn))
        }
    }

    /// Identity must distinguish a message's images by content, by count, and
    /// by order within the message.
    @Test func multipleImagesAreDistinguishedByContentCountAndOrder() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages = [GFTokenizer.Message(role: .user, content: "compare")]
        let all = ["img-1", "img-2", "img-3", "img-4"]
        let initial = request(messages: messages, imageIdentities: [all])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain, request: initial, content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        let follow = messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "again"),
        ]
        func match(_ leading: [String]) -> ServerPromptCacheMatch {
            cache.match(
                domain: domain,
                request: request(messages: follow,
                                 imageIdentities: [leading, [], []]),
                renderedPromptIDs: nil, tokenizer: tokenizer)
        }
        guard case .hit = match(all) else {
            Issue.record("identical four-image prefix must reuse")
            return
        }
        #expect(match(["img-1", "img-2", "img-4", "img-3"]).missReason == .imagesDiverged)
        #expect(match(["img-1", "img-2", "img-3"]).missReason == .imagesDiverged)
        #expect(match(all + ["img-5"]).missReason == .imagesDiverged)
        #expect(match(["img-1", "img-9", "img-3", "img-4"]).missReason == .imagesDiverged)
    }

    /// The same image moved to a different message is a different prompt, even
    /// though the flattened text and the multiset of images are unchanged.
    @Test func movingAnImageBetweenMessagesMisses() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages = [
            GFTokenizer.Message(role: .user, content: "one"),
            GFTokenizer.Message(role: .assistant, content: "ok"),
            GFTokenizer.Message(role: .user, content: "two"),
        ]
        let initial = request(messages: messages,
                              imageIdentities: [["image-a"], [], []])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain, request: initial, content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        let follow = messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "three"),
        ]
        let moved = cache.match(
            domain: domain,
            request: request(messages: follow,
                             imageIdentities: [[], [], ["image-a"], [], []]),
            renderedPromptIDs: nil, tokenizer: tokenizer)
        #expect(moved.missReason == .imagesDiverged)
    }

    /// A multimodal request whose identity array does not line up with its
    /// messages must neither publish nor match, since it cannot be told apart
    /// from a request carrying a different image.
    @Test func malformedImageIdentityIsFailClosed() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages = [GFTokenizer.Message(role: .user, content: "describe")]
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        let malformed = ValidatedChatRequest(
            messages: messages,
            multimodalMessages: [MultimodalMessage(
                role: .user, content: [.image(id: UUID()), .text("describe")])],
            imageFiles: [UUID(): URL(fileURLWithPath: "/dev/null")],
            imageIdentities: [],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain, request: malformed, content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        #expect(cache.entry == nil)

        cache.publish(
            domain: domain,
            request: request(messages: messages, imageIdentities: [["image-a"]]),
            content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        let follow = messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "again"),
        ]
        let malformedFollow = ValidatedChatRequest(
            messages: follow,
            multimodalMessages: [MultimodalMessage(
                role: .user, content: [.image(id: UUID())])],
            imageFiles: [UUID(): URL(fileURLWithPath: "/dev/null")],
            imageIdentities: [],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
        #expect(cache.match(domain: domain, request: malformedFollow,
                            renderedPromptIDs: nil,
                            tokenizer: tokenizer).missReason
                == .missingImageIdentity)
    }

    /// A text-only conversation must keep hitting after the image work; the
    /// identity comparison has to be inert when no image is present.
    @Test func textOnlyConversationIsUnaffectedByImageIdentity() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages = [GFTokenizer.Message(role: .user, content: "first")]
        let initial = request(messages: messages)
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain, request: initial, content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        let follow = request(messages: messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        guard case .hit = cache.match(
            domain: domain, request: follow,
            renderedPromptIDs: nil, tokenizer: tokenizer) else {
            Issue.record("text-only continuation must still reuse the prefix")
            return
        }
    }

    /// Nothing privileges the first turn. An image introduced at turn three is
    /// prefilled once as a miss, joins the cached prefix, and is reused by every
    /// later text turn exactly like a turn-one image.
    @Test func imageIntroducedAtALaterTurnJoinsTheCachedPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache()
        var messages = [GFTokenizer.Message(role: .user, content: "turn 1")]
        var identities: [[String]] = [[]]

        func publish(_ request: ValidatedChatRequest,
                     _ content: String) -> [Int32] {
            let prompt = (try? tokenizer.applyChatTemplate(request.messages))
                .map { tokenizer.encode($0, addBOS: false) } ?? []
            let kvBacked = prompt + tokenizer.encode(content, addBOS: false)
            cache.publish(
                domain: domain, request: request, content: content, calls: [],
                result: rawResult(prompt: prompt, kvBacked: kvBacked,
                                  boundary: tokenizer.endOfTurnID,
                                  reason: .endOfTurn))
            return kvBacked
        }

        // Turns 1 and 2 are text only.
        _ = publish(request(messages: messages, imageIdentities: identities),
                    "answer 1")
        messages += [
            GFTokenizer.Message(role: .assistant, content: "answer 1"),
            GFTokenizer.Message(role: .user, content: "turn 2"),
        ]
        identities += [[], []]
        _ = publish(request(messages: messages, imageIdentities: identities),
                    "answer 2")

        // Turn 3 introduces an image. The text bridge cannot render it, so this
        // resumes after a render rather than discarding the prefix.
        messages += [
            GFTokenizer.Message(role: .assistant, content: "answer 2"),
            GFTokenizer.Message(role: .user, content: "look at this"),
        ]
        identities += [[], ["image-late"]]
        let introducing = request(messages: messages, imageIdentities: identities)
        guard case .renderThenResume = cache.match(
            domain: domain, request: introducing,
            renderedPromptIDs: nil, tokenizer: tokenizer) else {
            Issue.record("introducing an image should resume after a render")
            return
        }
        let kvWithImage = publish(introducing, "answer 3")

        // Turn 4 onward reuses the prefix that now contains the late image.
        for turn in 4...6 {
            messages += [
                GFTokenizer.Message(role: .assistant,
                                    content: "answer \(turn - 1)"),
                GFTokenizer.Message(role: .user, content: "turn \(turn)"),
            ]
            identities += [[], []]
            let follow = request(messages: messages, imageIdentities: identities)
            guard case .hit(let effective, let cached) = cache.match(
                domain: domain, request: follow,
                renderedPromptIDs: nil, tokenizer: tokenizer) else {
                Issue.record("turn \(turn) lost the late image prefix")
                return
            }
            if turn == 4 { #expect(cached == kvWithImage.count) }
            let kvBacked = effective + tokenizer.encode("answer \(turn)",
                                                        addBOS: false)
            cache.publish(
                domain: domain, request: follow, content: "answer \(turn)",
                calls: [],
                result: rawResult(prompt: effective, kvBacked: kvBacked,
                                  boundary: tokenizer.endOfTurnID,
                                  reason: .endOfTurn))
        }

        // And the late image is still identity-checked, not merely carried.
        messages += [
            GFTokenizer.Message(role: .assistant, content: "answer 6"),
            GFTokenizer.Message(role: .user, content: "turn 7"),
        ]
        var swapped = identities + [[], []]
        swapped[5] = ["image-other"]
        #expect(cache.match(
            domain: domain,
            request: request(messages: messages, imageIdentities: swapped),
            renderedPromptIDs: nil,
            tokenizer: tokenizer).missReason == .imagesDiverged)
    }

    /// The rendered-prefix fast path must not be usable for an image request:
    /// placeholder tokens are identical whatever image they stand for, so
    /// rendered ids cannot tell two images apart. Previously this was safe only
    /// because the caller passed nil for multimodal requests.
    @Test func renderedPrefixFastPathRefusesImageRequests() async throws {
        let tokenizer = try await GFTokenizer.load()
        let messages = [GFTokenizer.Message(role: .user, content: "describe")]
        let initial = request(messages: messages, imageIdentities: [["image-a"]])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain, request: initial, content: "answer", calls: [],
            result: rawResult(prompt: prompt, kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        // Same rendered ids, different image: the fast path would hit on tokens
        // alone, so it must be refused.
        let follow = messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "again"),
        ]
        let rendered = kvBacked + tokenizer.encodeTextContinuation(userContent: "again")
        let other = cache.match(
            domain: domain,
            request: request(messages: follow,
                             imageIdentities: [["image-b"], [], []]),
            renderedPromptIDs: rendered, tokenizer: tokenizer)
        #expect(other.missReason == .imagesDiverged)
    }

    // MARK: - Slots

    /// The measured case this exists for.
    ///
    /// Two conversations interleave: a background extraction that runs on a
    /// cadence, and an assistant a person is typing into. With one slot each
    /// one's request destroys the other's prefix, so neither ever continues —
    /// a 62-minute meeting kept continuation on 6 of 19 calls. With two, both
    /// continue across the other's traffic.
    @Test func twoInterleavedConversationsBothKeepTheirPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        var single = ServerPromptCache(slotCount: 1)
        var paired = ServerPromptCache(slotCount: 2)

        var extraction = [GFTokenizer.Message(role: .user, content: "chunk 1")]
        var assistant = [GFTokenizer.Message(role: .user, content: "what was decided?")]
        var singleHits = 0
        var pairedHits = 0

        for turn in 1...4 {
            for (label, messages) in [("extraction", extraction),
                                      ("assistant", assistant)] {
                let request = request(messages: messages)
                let rendered = tokenizer.encode(
                    try tokenizer.applyChatTemplate(messages), addBOS: false)

                if case .hit = single.match(
                    domain: domain, request: request,
                    renderedPromptIDs: rendered, tokenizer: tokenizer) {
                    singleHits += 1
                }
                single.publish(
                    domain: domain, request: request,
                    content: "\(label) \(turn)", calls: [],
                    result: rawResult(
                        prompt: rendered,
                        kvBacked: rendered + tokenizer.encode(
                            "\(label) \(turn)", addBOS: false),
                        boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

                let resolution = paired.resolve(
                    domain: domain, request: request,
                    renderedPromptIDs: rendered, tokenizer: tokenizer)
                if case .hit = resolution.match { pairedHits += 1 }
                paired.publish(
                    slot: resolution.slot,
                    domain: domain, request: request,
                    content: "\(label) \(turn)", calls: [],
                    result: rawResult(
                        prompt: rendered,
                        kvBacked: rendered + tokenizer.encode(
                            "\(label) \(turn)", addBOS: false),
                        boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

                let reply = GFTokenizer.Message(
                    role: .assistant, content: "\(label) \(turn)")
                let next = GFTokenizer.Message(
                    role: .user, content: "\(label) follow \(turn)")
                if label == "extraction" {
                    extraction += [reply, next]
                } else {
                    assistant += [reply, next]
                }
            }
        }

        // One slot: every request is preceded by the other conversation's, so
        // nothing ever continues. This is the bug, asserted rather than
        // described.
        #expect(singleHits == 0)
        // Two slots: turn 1 is cold for each conversation, every later turn
        // continues. Six of eight against none.
        #expect(pairedHits == 6)
    }

    /// A slot is chosen by recency, but a hit is decided by content. The two
    /// conversations here are the same length and shape, so only the history
    /// comparison can tell them apart — a slot picked wrongly must miss, never
    /// hand over the other conversation's prefix.
    @Test func aSlotGuessNeverServesTheWrongConversation() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache(slotCount: 2)
        var lineages: [[GFTokenizer.Message]] = [
            [GFTokenizer.Message(role: .user, content: "alpha")],
            [GFTokenizer.Message(role: .user, content: "bravo")],
        ]
        var kvByLineage: [[Int32]] = [[], []]

        for index in 0..<2 {
            let request = request(messages: lineages[index])
            let rendered = tokenizer.encode(
                try tokenizer.applyChatTemplate(lineages[index]), addBOS: false)
            let kv = rendered + tokenizer.encode("reply \(index)", addBOS: false)
            kvByLineage[index] = kv
            let resolution = cache.resolve(
                domain: domain, request: request,
                renderedPromptIDs: rendered, tokenizer: tokenizer)
            #expect(resolution.missReason != nil)
            cache.publish(
                slot: resolution.slot, domain: domain, request: request,
                content: "reply \(index)", calls: [],
                result: rawResult(prompt: rendered, kvBacked: kv,
                                  boundary: tokenizer.endOfTurnID,
                                  reason: .endOfTurn))
            lineages[index] += [
                GFTokenizer.Message(role: .assistant, content: "reply \(index)"),
                GFTokenizer.Message(role: .user, content: "again \(index)"),
            ]
        }

        // Each continuation resumes onto the KV its own history produced, not
        // the other's, whichever slot recency happened to offer first.
        for index in 0..<2 {
            let request = request(messages: lineages[index])
            let resolution = cache.resolve(
                domain: domain, request: request,
                renderedPromptIDs: nil, tokenizer: tokenizer)
            guard case .hit(let effective, let cached) = resolution.match else {
                Issue.record("lineage \(index) lost its prefix")
                return
            }
            #expect(cached == kvByLineage[index].count)
            #expect(Array(effective.prefix(cached)) == kvByLineage[index])
        }
    }

    /// An unkeyed single-shot caller — a summariser that never continues — is
    /// what evicted the conversations in the measured case. A key stops it: the
    /// named lineages survive while the unnamed traffic recycles the slot left
    /// for it.
    @Test func aKeyedConversationSurvivesUnkeyedTraffic() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache(slotCount: 3)

        func run(_ messages: [GFTokenizer.Message],
                 key: String?,
                 reply: String) -> ServerPromptCacheResolution {
            let request = request(messages: messages, promptCacheKey: key)
            let rendered = (try? tokenizer.applyChatTemplate(messages))
                .map { tokenizer.encode($0, addBOS: false) } ?? []
            let resolution = cache.resolve(
                domain: domain, request: request,
                renderedPromptIDs: rendered, tokenizer: tokenizer)
            cache.publish(
                slot: resolution.slot, domain: domain, request: request,
                content: reply, calls: [],
                result: rawResult(
                    prompt: rendered,
                    kvBacked: rendered + tokenizer.encode(reply, addBOS: false),
                    boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
            return resolution
        }

        var extraction = [GFTokenizer.Message(role: .user, content: "chunk 1")]
        var chat = [GFTokenizer.Message(role: .user, content: "question 1")]
        let extractionSlot = run(extraction, key: "extraction", reply: "e1").slot
        let chatSlot = run(chat, key: "assistant", reply: "a1").slot
        #expect(extractionSlot != chatSlot)

        // Four unrelated single-shot calls, none of which ever continues. Each
        // one is a fresh conversation, so each misses; all four share the one
        // slot no key claimed.
        for index in 1...4 {
            let summary = run(
                [GFTokenizer.Message(role: .user, content: "summarise \(index)")],
                key: nil,
                reply: "s\(index)")
            #expect(summary.slot != extractionSlot)
            #expect(summary.slot != chatSlot)
        }

        // Both named conversations continue, after all of that.
        extraction += [
            GFTokenizer.Message(role: .assistant, content: "e1"),
            GFTokenizer.Message(role: .user, content: "chunk 2"),
        ]
        chat += [
            GFTokenizer.Message(role: .assistant, content: "a1"),
            GFTokenizer.Message(role: .user, content: "question 2"),
        ]
        let resumedExtraction = run(extraction, key: "extraction", reply: "e2")
        let resumedChat = run(chat, key: "assistant", reply: "a2")
        #expect(resumedExtraction.slot == extractionSlot)
        #expect(resumedChat.slot == chatSlot)
        guard case .hit = resumedExtraction.match, case .hit = resumedChat.match else {
            Issue.record("a keyed conversation was evicted by unkeyed traffic")
            return
        }
    }

    /// A key names one lineage, so a keyed request looks at its own slot and
    /// nowhere else — including when a different key's slot happens to hold a
    /// conversation that would match.
    @Test func aKeyLooksOnlyAtItsOwnSlot() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache(slotCount: 2)
        let opening = [GFTokenizer.Message(role: .user, content: "first")]
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(opening), addBOS: false)
        let kv = rendered + tokenizer.encode("answer", addBOS: false)
        let published = request(messages: opening, promptCacheKey: "one")
        let resolution = cache.resolve(
            domain: domain, request: published,
            renderedPromptIDs: rendered, tokenizer: tokenizer)
        cache.publish(
            slot: resolution.slot, domain: domain, request: published,
            content: "answer", calls: [],
            result: rawResult(prompt: rendered, kvBacked: kv,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        #expect(cache.key(slot: resolution.slot) == "one")

        let follow = opening + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ]
        // The same conversation under a different name is a different lineage:
        // it takes the free slot rather than the one "one" holds.
        let other = cache.resolve(
            domain: domain,
            request: request(messages: follow, promptCacheKey: "two"),
            renderedPromptIDs: nil, tokenizer: tokenizer)
        #expect(other.slot != resolution.slot)
        #expect(other.missReason == .noEntry)

        // Under its own name it continues.
        let same = cache.resolve(
            domain: domain,
            request: request(messages: follow, promptCacheKey: "one"),
            renderedPromptIDs: nil, tokenizer: tokenizer)
        #expect(same.slot == resolution.slot)
        guard case .hit = same.match else {
            Issue.record("a keyed conversation did not continue in its own slot")
            return
        }
    }

    /// The two halves of what a key means, which are easy to conflate.
    ///
    /// A key selects a *slot*, not a history. So reusing a name over time is
    /// the intended lifecycle — a conversation that ends and is replaced under
    /// the same name recycles its own slot and disturbs nothing else — while
    /// two conversations alive at once under one name resolve to that same
    /// single slot and overwrite each other on every call, leaving the other
    /// slots idle.
    ///
    /// The second half is the one way a client is worse off for sending a key
    /// than for sending none, so it is pinned here rather than left to the
    /// prose. It was also written wrongly first in
    /// `ServerPromptCacheSlotModelTests`, where giving both conversations one
    /// key looked like a product bug for a minute.
    @Test func oneKeyHoldsOneConversationAtATime() async throws {
        let tokenizer = try await GFTokenizer.load()

        func publish(into cache: inout ServerPromptCache,
                     _ text: String,
                     key: String?) -> ServerPromptCacheResolution {
            let messages = [GFTokenizer.Message(role: .user, content: text)]
            let request = request(messages: messages, promptCacheKey: key)
            let rendered = (try? tokenizer.applyChatTemplate(messages))
                .map { tokenizer.encode($0, addBOS: false) } ?? []
            let resolution = cache.resolve(
                domain: domain, request: request,
                renderedPromptIDs: rendered, tokenizer: tokenizer)
            cache.publish(
                slot: resolution.slot, domain: domain, request: request,
                content: "reply", calls: [],
                result: rawResult(
                    prompt: rendered,
                    kvBacked: rendered + tokenizer.encode("reply", addBOS: false),
                    boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
            return resolution
        }

        // Sequential reuse: each run is a new conversation under one stable
        // name, and each recycles the same slot without touching the others.
        var sequential = ServerPromptCache(slotCount: 3)
        _ = publish(into: &sequential, "neighbour", key: nil)
        let firstRun = publish(into: &sequential, "run 1", key: "nightly")
        for run in 2...5 {
            let later = publish(into: &sequential, "run \(run)", key: "nightly")
            #expect(later.slot == firstRun.slot,
                    "a stable name must keep recycling its own slot")
        }
        // The unkeyed neighbour is still there, untouched by five runs.
        #expect(sequential.occupiedSlotCount == 2)

        // Concurrent reuse: three live conversations sharing one name collapse
        // onto a single slot while the other two sit idle.
        var concurrent = ServerPromptCache(slotCount: 3)
        var slotsUsed: Set<Int> = []
        for conversation in 1...3 {
            slotsUsed.insert(
                publish(into: &concurrent, "conversation \(conversation)",
                        key: "shared").slot)
        }
        #expect(slotsUsed.count == 1, "one name cannot hold three lineages")
        #expect(concurrent.occupiedSlotCount == 1,
                "the other slots are left idle, which is the cost")

        // The same three conversations without a name spread across the slots
        // and all three are retained. This is the comparison that makes the
        // shared key worse than no key at all.
        var unkeyed = ServerPromptCache(slotCount: 3)
        for conversation in 1...3 {
            _ = publish(into: &unkeyed, "conversation \(conversation)", key: nil)
        }
        #expect(unkeyed.occupiedSlotCount == 3)
    }

    /// With every slot named and a new name arriving, something has to go. The
    /// least recently used name goes, and the slot is renamed on publish rather
    /// than answering for two conversations at once.
    @Test func aNewKeyRecyclesTheLeastRecentlyUsedSlot() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache(slotCount: 2)

        func run(_ text: String, key: String) -> Int {
            let messages = [GFTokenizer.Message(role: .user, content: text)]
            let request = request(messages: messages, promptCacheKey: key)
            let rendered = (try? tokenizer.applyChatTemplate(messages))
                .map { tokenizer.encode($0, addBOS: false) } ?? []
            let resolution = cache.resolve(
                domain: domain, request: request,
                renderedPromptIDs: rendered, tokenizer: tokenizer)
            cache.publish(
                slot: resolution.slot, domain: domain, request: request,
                content: "reply", calls: [],
                result: rawResult(
                    prompt: rendered,
                    kvBacked: rendered + tokenizer.encode("reply", addBOS: false),
                    boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
            return resolution.slot
        }

        let a = run("a", key: "a")
        let b = run("b", key: "b")
        #expect(a != b)
        // "a" is touched again, so "b" becomes the least recently used name.
        _ = run("a again", key: "a")
        let c = run("c", key: "c")
        #expect(c == b, "a new key must recycle the least recently used one")
        #expect(cache.key(slot: c) == "c")
        #expect(cache.key(slot: a) == "a")
        // Only two slots exist, so three names cannot all be held: "b" is gone
        // and comes back cold rather than answering from "c"'s lineage.
        #expect(!cache.slots.contains { $0.key == "b" })
    }

    /// Every slot spent means the count is the ceiling on retained
    /// conversations, not a hint. A third lineage on a two-slot cache costs the
    /// older of the two.
    @Test func slotCountIsTheCeilingOnRetainedConversations() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache(slotCount: 2)
        var lineages: [[GFTokenizer.Message]] = []

        func publish(_ messages: [GFTokenizer.Message]) -> Int {
            let request = request(messages: messages)
            let rendered = (try? tokenizer.applyChatTemplate(messages))
                .map { tokenizer.encode($0, addBOS: false) } ?? []
            let resolution = cache.resolve(
                domain: domain, request: request,
                renderedPromptIDs: rendered, tokenizer: tokenizer)
            cache.publish(
                slot: resolution.slot, domain: domain, request: request,
                content: "reply", calls: [],
                result: rawResult(
                    prompt: rendered,
                    kvBacked: rendered + tokenizer.encode("reply", addBOS: false),
                    boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
            return resolution.slot
        }

        for index in 0..<3 {
            var messages = [GFTokenizer.Message(
                role: .user, content: "lineage \(index)")]
            _ = publish(messages)
            messages += [
                GFTokenizer.Message(role: .assistant, content: "reply"),
                GFTokenizer.Message(role: .user, content: "more \(index)"),
            ]
            lineages.append(messages)
        }
        #expect(cache.occupiedSlotCount == 2)

        // The two most recent lineages continue; the oldest was evicted to make
        // room for the third and misses.
        var hits = 0
        for messages in lineages {
            let resolution = cache.resolve(
                domain: domain, request: request(messages: messages),
                renderedPromptIDs: nil, tokenizer: tokenizer)
            if case .hit = resolution.match { hits += 1 }
        }
        #expect(hits == 2)
    }

    /// A one-slot cache must behave exactly as the single-prefix cache did,
    /// down to which request evicts which. The default is one, so this is what
    /// every existing deployment gets.
    @Test func oneSlotIsTheHistoricalSinglePrefixBehavior() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache(slotCount: 1)
        let opening = [GFTokenizer.Message(role: .user, content: "first")]
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(opening), addBOS: false)
        let kv = rendered + tokenizer.encode("answer", addBOS: false)
        cache.publish(
            slot: 0, domain: domain, request: request(messages: opening),
            content: "answer", calls: [],
            result: rawResult(prompt: rendered, kvBacked: kv,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        // An unrelated request takes the only slot, exactly as it always did.
        let interloper = request(
            messages: [GFTokenizer.Message(role: .user, content: "unrelated")])
        let taken = cache.resolve(
            domain: domain, request: interloper,
            renderedPromptIDs: nil, tokenizer: tokenizer)
        #expect(taken.slot == 0)
        cache.publish(
            slot: 0, domain: domain, request: interloper,
            content: "other", calls: [],
            result: rawResult(prompt: rendered, kvBacked: kv,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        let follow = request(messages: opening + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        #expect(cache.resolve(
            domain: domain, request: follow,
            renderedPromptIDs: nil, tokenizer: tokenizer).missReason != nil)
    }

    /// Searching several slots produces several refusals, and the one reported
    /// has to be the informative one. Every slot holding somebody else's
    /// conversation answers `historyDiverged`, so picking by search order would
    /// let a neighbouring slot's truthful shrug bury what the request's own
    /// slot had to say.
    @Test func aMissReportsTheMostSpecificReasonNotTheNearestSlots() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = ServerPromptCache(slotCount: 2)
        let messages = [GFTokenizer.Message(role: .user, content: "shared")]
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kv = rendered + tokenizer.encode("answer", addBOS: false)

        // Slot 0 holds this conversation with an image; slot 1 holds something
        // unrelated and is the more recently used of the two.
        cache.publish(
            slot: 0, domain: domain,
            request: request(messages: messages, imageIdentities: [["image-a"]]),
            content: "answer", calls: [],
            result: rawResult(prompt: rendered, kvBacked: kv,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        cache.publish(
            slot: 1, domain: domain,
            request: request(messages: [
                GFTokenizer.Message(role: .user, content: "unrelated"),
            ]),
            content: "other", calls: [],
            result: rawResult(prompt: rendered, kvBacked: kv,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))

        // The same conversation continued with a different image. Slot 0 can
        // say so; slot 1 can only say the history diverged.
        let follow = request(
            messages: messages + [
                GFTokenizer.Message(role: .assistant, content: "answer"),
                GFTokenizer.Message(role: .user, content: "again"),
            ],
            imageIdentities: [["image-b"], [], []])
        let resolution = cache.resolve(
            domain: domain, request: follow,
            renderedPromptIDs: nil, tokenizer: tokenizer)
        #expect(resolution.missReason == .imagesDiverged)

        // And the reverse arrangement gives the same answer, so this is the
        // reason winning on its merits rather than on where it sat.
        var reversed = ServerPromptCache(slotCount: 2)
        reversed.publish(
            slot: 1, domain: domain,
            request: request(messages: messages, imageIdentities: [["image-a"]]),
            content: "answer", calls: [],
            result: rawResult(prompt: rendered, kvBacked: kv,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        reversed.publish(
            slot: 0, domain: domain,
            request: request(messages: [
                GFTokenizer.Message(role: .user, content: "unrelated"),
            ]),
            content: "other", calls: [],
            result: rawResult(prompt: rendered, kvBacked: kv,
                              boundary: tokenizer.endOfTurnID, reason: .endOfTurn))
        #expect(reversed.resolve(
            domain: domain, request: follow,
            renderedPromptIDs: nil, tokenizer: tokenizer).missReason
            == .imagesDiverged)
    }

    /// What a slot costs, checked against the architecture rather than trusted.
    ///
    /// The figure the flag's help text and the startup log quote is this one,
    /// and it is the whole reason the slot count is capped: a slot is a KV
    /// lineage, not bookkeeping. Derived here from Gemma 4's own shape so the
    /// claim moves if the architecture does.
    ///
    ///   25 SWA layers  x 2 (K and V) x 1,304 tokens x 8 heads  x 256 dims x 2 B
    ///    5 full layers x 2           x context     x 2 heads x 512 dims x 2 B
    ///
    /// The SWA capacity is the sliding window plus the largest prefill chunk,
    /// which is the vision pooled-token cap (280) rather than the 256-token
    /// text chunk.
    @Test func aSlotCostsOneWholeKVLineage() {
        let config = ArchConfig.gemma4_26B_A4B
        let ringCapacity = config.slidingWindow + 280
        let swa = 25 * 2 * ringCapacity * config.numKVHeads * config.headDim * 2
        let full = 5 * 2 * 16_384 * config.numFullKVHeads * config.fullHeadDim * 2

        let measured = KVCacheManager.storageBytes(
            config: config,
            maxContext: 16_384,
            fp16RingEnabled: true,
            slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 280)
        #expect(measured == swa + full)
        // ~575 MiB per slot at the default context: the number the help text
        // quotes, and the reason four is the cap rather than sixteen.
        #expect(measured / (1_024 * 1_024) == 574)

        // What a slot actually holds is smaller, because the buffers fault in
        // on write: the ring-capped SWA layers are reached early and the linear
        // full-attention layers grow with position. The server doc quotes
        // ~333 MiB for a 4,000-token conversation; deriving it here means the
        // doc breaks with the architecture rather than quietly going stale.
        let resident = swa + full * 4_000 / 16_384
        #expect(Int((Double(resident) / (1_024 * 1_024)).rounded()) == 333)

        // Note what the split implies: the early-reached ring is the larger
        // half, so the saving is real but not dramatic at ordinary prompt
        // sizes — 58% of the ceiling at a quarter of the context, not 25%. A
        // conversation has to be short to be cheap, and one long enough to be
        // worth caching is most of the way to the ceiling already.
        #expect(resident * 100 / measured == 57)
        #expect(swa > full * 4_000 / 16_384)

        // Only the full-attention layers scale with the context; the ring caps
        // the other twenty-five. Quadrupling the context does not quadruple the
        // slot, and an operator sizing a machine needs that to be true.
        let larger = KVCacheManager.storageBytes(
            config: config,
            maxContext: 65_536,
            fp16RingEnabled: true,
            slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: 280)
        #expect(larger == swa + full * 4)
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = [],
        imageIdentities: [[String]] = [],
        promptCacheKey: String? = nil
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            imageIdentities: imageIdentities,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16,
            promptCacheKey: promptCacheKey)
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
