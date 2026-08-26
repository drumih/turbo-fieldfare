import Testing
@testable import TurboFieldfareServerCore

/// A cancelled request used to be excluded from logging entirely, on the
/// reasoning that a client hanging up is not a server fault. That is true, and it
/// left the request's last line as `generating` forever: reading the log, a
/// running request, an abandoned one and a crashed one were indistinguishable.
@Suite struct ServerLogTests {
    @Test func completionMessageIncludesPromptAndGenerationTimings() {
        let completion = ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 120_013,
                               completionTokens: 1,
                               totalTokens: 120_014,
                               cachedTokens: 0),
            prefillSeconds: 675.595,
            decodeSeconds: 0.250)
        let line = ServerLog.completedMessage(
            id: "chatcmpl-timing",
            duration: .seconds(676),
            completion: completion)

        #expect(line.contains("pp=675.595s"))
        #expect(line.contains("tg=0.250s"))
        #expect(line.contains("tg_tok_s=4.000"))
    }

    /// `pp_tok_s` divides by the tokens actually prefilled, so a cache hit must
    /// not be counted as prefill work. Every context-ladder rung ran with an
    /// empty cache and reported `cached=0`, which leaves this subtraction
    /// unexercised by the benchmark; prompt reuse is the default server mode,
    /// so it is the common case in practice rather than an edge case.
    @Test func promptThroughputExcludesCachedTokens() {
        let completion = ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 100_000,
                               completionTokens: 4,
                               totalTokens: 100_004,
                               cachedTokens: 75_000),
            prefillSeconds: 2.5,
            decodeSeconds: 0.5)
        let line = ServerLog.completedMessage(
            id: "chatcmpl-cached",
            duration: .seconds(3),
            completion: completion)

        #expect(line.contains("prompt=100000"))
        #expect(line.contains("cached=75000"))
        // 25,000 computed tokens / 2.5s, not 100,000 / 2.5s.
        #expect(line.contains("pp_tok_s=10000.000"))
        #expect(line.contains("tg_tok_s=8.000"))
    }

    /// A fully cached prompt does no prefill work at all. The rate must read as
    /// zero rather than dividing by a zero elapsed time.
    @Test func aFullyCachedPromptReportsZeroPromptThroughput() {
        let completion = ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 4_096,
                               completionTokens: 1,
                               totalTokens: 4_097,
                               cachedTokens: 4_096),
            prefillSeconds: 0,
            decodeSeconds: 0.125)
        let line = ServerLog.completedMessage(
            id: "chatcmpl-allcached",
            duration: .seconds(1),
            completion: completion)

        #expect(line.contains("pp=0.000s"))
        #expect(line.contains("pp_tok_s=0.000"))
    }

    /// Cached tokens are reported by the runtime, so a value exceeding the
    /// prompt would otherwise produce a negative token count and a negative
    /// rate in the log.
    @Test func moreCachedTokensThanPromptTokensClampsToZero() {
        let completion = ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 10,
                               completionTokens: 1,
                               totalTokens: 11,
                               cachedTokens: 25),
            prefillSeconds: 1,
            decodeSeconds: 1)
        let line = ServerLog.completedMessage(
            id: "chatcmpl-clamp",
            duration: .seconds(2),
            completion: completion)

        #expect(line.contains("pp_tok_s=0.000"))
        // The request id legitimately contains a hyphen, so assert on the
        // computed rate fields rather than scanning the whole line.
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
