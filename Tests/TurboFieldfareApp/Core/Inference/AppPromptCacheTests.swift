import Foundation
import Testing
import TurboFieldfareDecodeProtocol

@testable import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AppPromptCacheTests {
    private let domain = AppPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 4_096,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    /// The KV holds sampled token IDs, but the next request re-encodes the
    /// assistant turn from text. The plain-prefix path therefore cannot match;
    /// the bridge is what makes a follow-up prefill only the new turn.
    @Test func textContinuationBridgePrefillsOnlyTheNewTurn() async throws {
        let tokenizer = try await GFTokenizer.load()
        let first: [DecodeChatMessage] = [.init(role: .user, content: "first")]
        let firstPrompt = try render(first, tokenizer)
        let kvBacked = firstPrompt + tokenizer.encode("answer", addBOS: false)

        var cache = AppPromptCache()
        cache.publish(
            domain: domain,
            messages: first,
            content: "answer",
            result: rawResult(prompt: firstPrompt,
                              kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn))

        let second = first + [
            .init(role: .assistant, content: "answer"),
            .init(role: .user, content: "second"),
        ]
        let rendered = try render(second, tokenizer)
        let match = cache.match(domain: domain,
                                messages: second,
                                renderedPromptIDs: rendered,
                                tokenizer: tokenizer)

        guard case .hit(let effective, let cached) = match else {
            Issue.record("expected a text-continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        // The plain-prefix path genuinely cannot serve this; if it ever could,
        // the bridge would be dead code.
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    /// Reuse must keep working as the conversation grows: turn three resumes
    /// from the prefix turn two published, not from the original prompt.
    @Test func chainedContinuationReusesTheGrowingPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        var cache = AppPromptCache()

        let first: [DecodeChatMessage] = [.init(role: .user, content: "first")]
        let firstPrompt = try render(first, tokenizer)
        let firstKV = firstPrompt + tokenizer.encode("answer one", addBOS: false)
        cache.publish(
            domain: domain,
            messages: first,
            content: "answer one",
            result: rawResult(prompt: firstPrompt,
                              kvBacked: firstKV,
                              boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn))

        let second = first + [
            .init(role: .assistant, content: "answer one"),
            .init(role: .user, content: "second"),
        ]
        guard case .hit(let secondEffective, let secondCached) = cache.match(
            domain: domain,
            messages: second,
            renderedPromptIDs: try render(second, tokenizer),
            tokenizer: tokenizer) else {
            Issue.record("expected a hit on the second turn")
            return
        }
        #expect(secondCached == firstKV.count)

        // Publish what the second generation would leave in KV.
        let secondKV = secondEffective + tokenizer.encode("answer two", addBOS: false)
        cache.publish(
            domain: domain,
            messages: second,
            content: "answer two",
            result: rawResult(prompt: secondEffective,
                              kvBacked: secondKV,
                              boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn))

        let third = second + [
            .init(role: .assistant, content: "answer two"),
            .init(role: .user, content: "third"),
        ]
        guard case .hit(let thirdEffective, let thirdCached) = cache.match(
            domain: domain,
            messages: third,
            renderedPromptIDs: try render(third, tokenizer),
            tokenizer: tokenizer) else {
            Issue.record("expected a hit on the third turn")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "third")
        #expect(thirdCached == secondKV.count)
        #expect(thirdEffective == secondKV + bridge)
        // Each turn reuses strictly more than the last.
        #expect(thirdCached > secondCached)
    }

    /// Regression guard for a silent performance cliff: the bridge compares
    /// assistant text byte-for-byte, so any normalisation upstream (trimming,
    /// whitespace collapsing) drops reuse back to a full re-prefill.
    @Test func normalisedAssistantTextMissesTheBridge() async throws {
        let tokenizer = try await GFTokenizer.load()
        let first: [DecodeChatMessage] = [.init(role: .user, content: "first")]
        let firstPrompt = try render(first, tokenizer)
        let kvBacked = firstPrompt + tokenizer.encode("answer", addBOS: false)

        var cache = AppPromptCache()
        cache.publish(
            domain: domain,
            messages: first,
            content: "answer\n",
            result: rawResult(prompt: firstPrompt,
                              kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn))

        let trimmed = first + [
            .init(role: .assistant, content: "answer"),
            .init(role: .user, content: "second"),
        ]
        #expect(cache.match(domain: domain,
                            messages: trimmed,
                            renderedPromptIDs: try render(trimmed, tokenizer),
                            tokenizer: tokenizer) == .miss)

        let verbatim = first + [
            .init(role: .assistant, content: "answer\n"),
            .init(role: .user, content: "second"),
        ]
        guard case .hit = cache.match(domain: domain,
                                      messages: verbatim,
                                      renderedPromptIDs: try render(verbatim, tokenizer),
                                      tokenizer: tokenizer) else {
            Issue.record("verbatim assistant text must still hit")
            return
        }
    }

    /// A generation that stopped at the token cap never emitted the end-of-turn
    /// token, so the bridge has to supply the uncommitted boundary itself.
    @Test func maxTokensStopPrependsTheUncommittedBoundary() async throws {
        let tokenizer = try await GFTokenizer.load()
        let first: [DecodeChatMessage] = [.init(role: .user, content: "first")]
        let firstPrompt = try render(first, tokenizer)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = firstPrompt + generated
        let boundary = try #require(generated.last)

        var cache = AppPromptCache()
        cache.publish(
            domain: domain,
            messages: first,
            content: "answer",
            result: rawResult(prompt: firstPrompt,
                              kvBacked: kvBacked,
                              boundary: boundary,
                              reason: .maxTokens))

        let second = first + [
            .init(role: .assistant, content: "answer"),
            .init(role: .user, content: "second"),
        ]
        guard case .hit(let effective, let cached) = cache.match(
            domain: domain,
            messages: second,
            renderedPromptIDs: try render(second, tokenizer),
            tokenizer: tokenizer) else {
            Issue.record("expected a maxTokens continuation hit")
            return
        }
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + [boundary]
                + tokenizer.encodeTextContinuation(userContent: "second"))
    }

    /// Only a clean end-of-turn or token-cap finish leaves a prefix whose KV
    /// boundary is known. A stop-string or EOS finish must not be published;
    /// cancellation cannot reach here at all, because it throws before a
    /// `RawDecodeResult` exists.
    @Test(arguments: [StopReason.stopString, .eos, .toolCalls])
    func unsafeStopReasonIsNeverPublished(_ reason: StopReason) async throws {
        let tokenizer = try await GFTokenizer.load()
        let first: [DecodeChatMessage] = [.init(role: .user, content: "first")]
        let firstPrompt = try render(first, tokenizer)

        var cache = AppPromptCache()
        cache.publish(
            domain: domain,
            messages: first,
            content: "partial",
            result: rawResult(prompt: firstPrompt,
                              kvBacked: firstPrompt + tokenizer.encode("partial", addBOS: false),
                              boundary: tokenizer.endOfTurnID,
                              reason: reason))
        #expect(cache.entry == nil)
    }

    @Test func domainMismatchMisses() async throws {
        let tokenizer = try await GFTokenizer.load()
        let first: [DecodeChatMessage] = [.init(role: .user, content: "first")]
        let firstPrompt = try render(first, tokenizer)
        let kvBacked = firstPrompt + tokenizer.encode("answer", addBOS: false)

        var cache = AppPromptCache()
        cache.publish(
            domain: domain,
            messages: first,
            content: "answer",
            result: rawResult(prompt: firstPrompt,
                              kvBacked: kvBacked,
                              boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn))

        // A context-length change alone must invalidate reuse: the KV was
        // allocated for the old window.
        var other = domain
        other = AppPromptCacheDomain(
            modelID: domain.modelID,
            sourceSnapshotHash: domain.sourceSnapshotHash,
            runtimeProfileHash: domain.runtimeProfileHash,
            maximumContext: 8_192,
            kvStorage: domain.kvStorage,
            fp16RingEnabled: domain.fp16RingEnabled,
            templateSHA256: domain.templateSHA256)

        let second = first + [
            .init(role: .assistant, content: "answer"),
            .init(role: .user, content: "second"),
        ]
        #expect(cache.match(domain: other,
                            messages: second,
                            renderedPromptIDs: try render(second, tokenizer),
                            tokenizer: tokenizer) == .miss)
    }

    @Test func invalidateDropsTheEntry() async throws {
        let tokenizer = try await GFTokenizer.load()
        let first: [DecodeChatMessage] = [.init(role: .user, content: "first")]
        let firstPrompt = try render(first, tokenizer)

        var cache = AppPromptCache()
        cache.publish(
            domain: domain,
            messages: first,
            content: "answer",
            result: rawResult(prompt: firstPrompt,
                              kvBacked: firstPrompt + tokenizer.encode("answer", addBOS: false),
                              boundary: tokenizer.endOfTurnID,
                              reason: .endOfTurn))
        #expect(cache.entry != nil)
        cache.invalidate()
        #expect(cache.entry == nil)
    }

    private func render(_ messages: [DecodeChatMessage],
                        _ tokenizer: GFTokenizer) throws -> [Int32] {
        tokenizer.encode(
            try tokenizer.applyChatTemplate(messages.map(GFTokenizer.Message.init)),
            addBOS: false)
    }

    private func rawResult(prompt: [Int32],
                           kvBacked: [Int32],
                           boundary: Int32,
                           reason: StopReason) -> RawDecodeResult {
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
}
