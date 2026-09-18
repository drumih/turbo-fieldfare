import Testing
@testable import TurboFieldfareServerCore

/// A cancelled request used to be excluded from logging entirely, on the
/// reasoning that a client hanging up is not a server fault. That is true, and it
/// left the request's last line as `generating` forever: reading the log, a
/// running request, an abandoned one and a crashed one were indistinguishable.
@Suite struct ServerLogTests {
    private func completion(promptTokens: Int,
                            cachedTokens: Int,
                            completionTokens: Int,
                            prefillSeconds: Double,
                            decodeSeconds: Double) -> ServerCompletion {
        ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: promptTokens,
                               completionTokens: completionTokens,
                               totalTokens: promptTokens + completionTokens,
                               cachedTokens: cachedTokens),
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds)
    }

    /// The two phases have wildly different rates at a long context — minutes
    /// of prefill against sub-second decode — so one aggregate duration hides
    /// which of them a change moved.
    @Test func completedMessageReportsPrefillAndDecodeSeparately() {
        let line = ServerLog.completedMessage(
            id: "chatcmpl-timing",
            duration: .seconds(676),
            completion: completion(promptTokens: 120_013,
                                   cachedTokens: 0,
                                   completionTokens: 1,
                                   prefillSeconds: 675.595,
                                   decodeSeconds: 0.250))

        #expect(line.contains("prompt=120013"))
        #expect(line.contains("cached=0"))
        #expect(line.contains("pp=675.595s"))
        #expect(line.contains("pp_tok_s=177.640"))
        #expect(line.contains("tg=0.250s"))
        #expect(line.contains("tg_tok_s=4.000"))
        #expect(line.contains("finish=stop"))
    }

    /// A prompt-cache hit prefills only the suffix, so counting the cached
    /// prefix as prefilled would report a prompt rate the runtime never
    /// achieved. Prompt reuse is the default server mode, so this is the
    /// common case rather than an edge one.
    @Test func promptThroughputExcludesCachedTokens() {
        let line = ServerLog.completedMessage(
            id: "chatcmpl-cached",
            duration: .seconds(3),
            completion: completion(promptTokens: 100_000,
                                   cachedTokens: 75_000,
                                   completionTokens: 4,
                                   prefillSeconds: 2.5,
                                   decodeSeconds: 0.5))

        #expect(line.contains("prompt=100000"))
        #expect(line.contains("cached=75000"))
        // 25,000 computed tokens over 2.5s, not 100,000 over 2.5s.
        #expect(line.contains("pp_tok_s=10000.000"))
        #expect(line.contains("tg_tok_s=8.000"))
    }

    /// A fully cached prompt does no prefill work at all, and a request that
    /// fails before decoding reports no decode time: both rates must read as
    /// zero rather than dividing by a zero elapsed time.
    @Test func zeroDurationsReportZeroRatesInsteadOfDividingByZero() {
        let line = ServerLog.completedMessage(
            id: "chatcmpl-allcached",
            duration: .seconds(1),
            completion: completion(promptTokens: 4_096,
                                   cachedTokens: 4_096,
                                   completionTokens: 0,
                                   prefillSeconds: 0,
                                   decodeSeconds: 0))

        #expect(line.contains("pp=0.000s"))
        #expect(line.contains("pp_tok_s=0.000"))
        #expect(line.contains("tg=0.000s"))
        #expect(line.contains("tg_tok_s=0.000"))
        #expect(!line.contains("nan"))
        #expect(!line.contains("inf"))
    }

    /// Both counts come from the runtime, so a cached count above the prompt
    /// would log a negative token count and a negative rate instead of the
    /// zero work that was actually done.
    @Test func moreCachedTokensThanPromptTokensClampsToZero() {
        let line = ServerLog.completedMessage(
            id: "chatcmpl-clamp",
            duration: .seconds(2),
            completion: completion(promptTokens: 10,
                                   cachedTokens: 25,
                                   completionTokens: 1,
                                   prefillSeconds: 1,
                                   decodeSeconds: 1))

        #expect(line.contains("pp_tok_s=0.000"))
        // The request id legitimately contains a hyphen, so assert on the rate
        // fields rather than scanning the whole line for a minus sign.
        #expect(!line.contains("pp_tok_s=-"))
        #expect(!line.contains("tg_tok_s=-"))
    }

    @Test func aCancelledRequestReadsAsCancelledRatherThanFailed() {
        let line = ServerLog.cancelledMessage(id: "chatcmpl-abc",
                                              phase: "generating",
                                              duration: .milliseconds(5369))

        #expect(line.contains("chatcmpl-abc"))
        #expect(line.contains("cancelled by client"))
        #expect(line.contains("phase=generating"))
        #expect(line.contains("5.369s"))
        // Distinct from a failure, so an abandoned client does not read as a
        // server fault in the log or in anything counting error lines.
        #expect(!line.contains("failed"))
    }
}
