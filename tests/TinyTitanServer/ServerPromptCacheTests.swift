import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

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
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
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

        let continuation = request(
            messages: initial.messages + [
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

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            let publication = cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            // Unsafe stop reasons must not publish a cacheable entry
            // (S17). `publish` returns nil when the entry is rejected.
            #expect(publication == nil)
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
        #expect(
            cache.match(
                domain: domain,
                request: changed,
                renderedPromptIDs: rendered,
                tokenizer: tokenizer) == .miss)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        let publication = cache.publish(
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
        // A stop-string-filtered tail must not be published as a cacheable
        // entry: the cached KV would resume from a partial stop string (S17).
        #expect(publication == nil)
    }

    @Test func multiPrefixChoosesLongestExactPrefixAndUsesLRUEviction() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        var cache = ServerPromptCache(maximumEntries: 2)
        let shortPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "short",
            calls: [],
            result: rawResult(
                prompt: [1, 2],
                kvBacked: [1, 2],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let short = try #require(shortPublication)
        let longPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "long",
            calls: [],
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let long = try #require(longPublication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4],
            tokenizer: tokenizer)
        #expect(
            match
                == .hit(
                    entryID: long.entry.id,
                    effectivePromptIDs: [1, 2, 3, 4],
                    cachedPromptTokens: 3))

        let newestPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "newest",
            calls: [],
            result: rawResult(
                prompt: [9],
                kvBacked: [9],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let newest = try #require(newestPublication)
        #expect(newest.evictedEntryIDs == [short.entry.id])
        #expect(cache.entries.map(\.id) == [long.entry.id, newest.entry.id])
    }

    /// An identical-prompt replay of a published entry reports the ENTIRE
    /// prompt as cached (`cachedPromptTokens == rendered.count`): at the
    /// session layer (S12) that leaves nothing to prefill, which is what
    /// "a prompt-cache hit short-circuits prefill" means. The end-to-end
    /// resume-path half of that contract lives inside `ServerModelSession`
    /// (its private init and hardcoded production-arch load make it
    /// untestable with the synthetic toy); this pins the cache layer's half.
    @Test func identicalReplayReportsEntirePromptAsCached() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first")
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let entry = try #require(publication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt,
            tokenizer: tokenizer)
        #expect(
            match
                == .hit(
                    entryID: entry.entry.id,
                    effectivePromptIDs: prompt,
                    cachedPromptTokens: prompt.count))
    }

    /// Regression: the cache keys on the post-strip view of a request, so a
    /// "<model>-fast" continuation re-renders its tail through CLIStrip too.
    /// Keying on the raw request instead produced a bridge that still carried
    /// the CLI's <system-reminder> scaffolding — an unstripped tail spliced
    /// onto a stripped prefix, so the cached turn silently lost the alias's
    /// strip and stopped reproducing a fresh prefill of the same request.
    @Test func strippedContinuationBridgeDropsReminderScaffolding() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let bloat = GFTokenizer.Message(role: .system, content: "you are an agent")
        let rawFirst = GFTokenizer.Message(
            role: .user,
            content: "first<system-reminder>\ncwd is /tmp\n</system-reminder>")
        let rawSecond = GFTokenizer.Message(
            role: .user,
            content: "second<system-reminder>\nfile changed\n</system-reminder>")

        // Turn 1, exactly as ServerInference composes it: strip, then key the
        // cache on the filtered view that was actually encoded.
        let firstStrip = CLIStrip.filter(messages: [bloat, rawFirst], tools: [])
        let initial = request(messages: [bloat, rawFirst])
            .replacingMessages(firstStrip.messages, tools: firstStrip.tools)
        #expect(initial.messages.map(\.content) == ["first"])

        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let kvBacked = initialPrompt + tokenizer.encode("answer", addBOS: false)
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

        // Turn 2 arrives with the bloat and both reminder blocks intact.
        let rawContinuation = [
            bloat,
            rawFirst,
            GFTokenizer.Message(role: .assistant, content: "answer"),
            rawSecond,
        ]
        let secondStrip = CLIStrip.filter(messages: rawContinuation, tools: [])
        let continuation = request(messages: rawContinuation)
            .replacingMessages(secondStrip.messages, tools: secondStrip.tools)
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected text continuation hit on the stripped view")
            return
        }
        // The bridge is the *stripped* user turn; the reminder block never
        // reaches the model, and the raw turn would have produced a longer one.
        #expect(cached == kvBacked.count)
        #expect(
            effective == kvBacked
                + tokenizer.encodeTextContinuation(userContent: "second"))
        #expect(
            effective != kvBacked
                + tokenizer.encodeTextContinuation(userContent: rawSecond.content ?? ""))

        // And the shape of the defect this guards: an entry keyed on the raw
        // messages still describes a KV range prefilled from the *stripped*
        // ones, so its continuation bridge carries the reminder block — an
        // unstripped tail on a stripped prefix.
        var rawKeyed = ServerPromptCache()
        rawKeyed.publish(
            domain: domain,
            request: request(messages: [bloat, rawFirst]),
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let rawMatch = rawKeyed.match(
            domain: domain,
            request: request(messages: rawContinuation),
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)
        guard case .hit(_, let rawEffective, _) = rawMatch else {
            Issue.record("expected the raw-keyed cache to still hit")
            return
        }
        #expect(
            rawEffective == kvBacked
                + tokenizer.encodeTextContinuation(userContent: rawSecond.content ?? ""))
        #expect(rawEffective != effective)
    }

    /// The post-strip view swaps only the messages and tools; every other
    /// validated field must survive, or the cached turn would silently change
    /// sampling or streaming behavior.
    @Test func replacingMessagesPreservesTheRestOfTheRequest() {
        let original = request(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [])
        let replaced = original.replacingMessages(
            [GFTokenizer.Message(role: .user, content: "stripped")],
            tools: [])

        #expect(replaced.messages.map(\.content) == ["stripped"])
        #expect(replaced.stream == original.stream)
        #expect(replaced.includeUsage == original.includeUsage)
        #expect(replaced.maximumCompletionTokens == original.maximumCompletionTokens)
        #expect(replaced.stripCLIPrompt == original.stripCLIPrompt)
        #expect(
            replaced.generationConfig.maxNewTokens
                == original.generationConfig.maxNewTokens)
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
    ) -> TinyTitanCore.RawDecodeResult {
        TinyTitanCore.RawDecodeResult(
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
}
