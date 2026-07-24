import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareServerCore

/// Metal-free fake for the runtime seam. Emits canned deltas and lets tests
/// script outcomes and observe concurrency.
final class FakeTokenGenerator: TokenGenerating, @unchecked Sendable {
    struct Call: Sendable {
        var input: GenerationInput
        var effectiveMaxTokens: Int
    }

    /// Records actor-style entry/exit so tests can assert serialization.
    actor OverlapTracker {
        private(set) var active = 0
        private(set) var maxActive = 0
        private(set) var events: [String] = []
        private(set) var calls: [Call] = []

        func record(_ call: Call) {
            calls.append(call)
        }

        func enter(_ tag: String) {
            active += 1
            maxActive = max(maxActive, active)
            events.append("+\(tag)")
        }

        func exit(_ tag: String) {
            active -= 1
            events.append("-\(tag)")
        }
    }

    let tracker = OverlapTracker()
    /// Canned deltas emitted via onDelta, in order.
    var deltas: [String] = ["Hello", ", world"]
    var finishReason: FinishReason = .stop
    var promptTokenCount = 8
    /// Artificial decode time so overlapping calls would be observable.
    var latencyNanoseconds: UInt64 = 20_000_000

    var calls: [Call] {
        get async { await tracker.calls }
    }

    /// When set, generate throws this after entering (simulates a mid-flight
    /// runtime failure).
    var thrownError: (any Error)?

    /// Tag derived from the input, used in overlap events.
    private func tag(for input: GenerationInput) -> String {
        switch input {
        case .chat(let messages): return messages.last?.content ?? "chat"
        case .raw(let prompt): return prompt
        }
    }

    func tokenCount(for input: GenerationInput) throws -> Int {
        promptTokenCount
    }

    func generate(_ input: GenerationInput,
                  sampling: SamplingParams,
                  effectiveMaxTokens: Int,
                  onDelta: (@Sendable (String) -> Void)?) async throws -> GenerationOutcome {
        let tag = tag(for: input)
        await tracker.record(Call(input: input, effectiveMaxTokens: effectiveMaxTokens))
        await tracker.enter(tag)
        if let thrownError {
            await tracker.exit(tag)
            throw thrownError
        }
        try? await Task.sleep(nanoseconds: latencyNanoseconds)
        var text = ""
        for delta in deltas {
            text += delta
            onDelta?(delta)
        }
        await tracker.exit(tag)
        return GenerationOutcome(text: text,
                                 promptTokens: promptTokenCount,
                                 completionTokens: deltas.count,
                                 finishReason: finishReason)
    }
}

@Suite("GenerationSession")
struct GenerationSessionTests {
    private func makeSession(_ generator: FakeTokenGenerator,
                             maxContext: Int = 64) -> GenerationSession {
        GenerationSession(modelID: "test-model", maxContext: maxContext, generator: generator)
    }

    @Test func assistantRoleAccepted() async throws {
        let generator = FakeTokenGenerator()
        let session = makeSession(generator)
        let outcome = try await session.chat(
            messages: [ChatMessage(role: "user", content: "hi"),
                       ChatMessage(role: "assistant", content: "hello"),
                       ChatMessage(role: "user", content: "again")],
            sampling: SamplingParams())
        #expect(outcome.text == "Hello, world")
        // assistant maps to the template's model turn; the input passed down
        // keeps the OpenAI role for the tokenizer to render.
        let calls = await generator.calls
        guard case .chat(let forwarded) = calls.first?.input else {
            Issue.record("expected chat input")
            return
        }
        #expect(forwarded[1].role == "assistant")
    }

    @Test func toolRoleRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.unsupportedRole("tool")) {
            try await session.chat(
                messages: [ChatMessage(role: "tool", content: "result")],
                sampling: SamplingParams())
        }
    }

    @Test func functionRoleRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.unsupportedRole("function")) {
            try await session.chat(
                messages: [ChatMessage(role: "function", content: "result")],
                sampling: SamplingParams())
        }
    }

    @Test func emptyMessagesRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.emptyMessages) {
            try await session.chat(messages: [], sampling: SamplingParams())
        }
    }

    @Test func deltasStreamedAndUsageComputed() async throws {
        let generator = FakeTokenGenerator()
        let session = makeSession(generator)
        let streamed = Mutex<[String]>([])
        let outcome = try await session.chat(
            messages: [ChatMessage(role: "user", content: "hi")],
            sampling: SamplingParams()) { delta in
                streamed.withLock { $0.append(delta) }
            }
        #expect(streamed.withLock { $0 } == ["Hello", ", world"])
        #expect(outcome.text == "Hello, world")
        #expect(outcome.promptTokens == 8)
        #expect(outcome.completionTokens == 2)
        #expect(outcome.usage.totalTokens == 10)
    }

    @Test func explicitMaxTokensOverflowRejected() async {
        let session = makeSession(FakeTokenGenerator(), maxContext: 10)
        await #expect(
            throws: GenerationRequestError.contextOverflow(prompt: 8, maxTokens: 3, maxContext: 10)
        ) {
            try await session.chat(
                messages: [ChatMessage(role: "user", content: "hi")],
                sampling: SamplingParams(maxTokens: 3))
        }
    }

    @Test func absentMaxTokensClampsToRemainingContext() async throws {
        let generator = FakeTokenGenerator()
        let session = makeSession(generator, maxContext: 100)
        _ = try await session.chat(
            messages: [ChatMessage(role: "user", content: "hi")],
            sampling: SamplingParams())
        let calls = await generator.calls
        #expect(calls.first?.effectiveMaxTokens == 92)
    }

    @Test func concurrentGenerationsSerializeInOrder() async throws {
        let generator = FakeTokenGenerator()
        let session = makeSession(generator)

        async let first = session.chat(
            messages: [ChatMessage(role: "user", content: "one")],
            sampling: SamplingParams())
        // Give the first call a head start so FIFO order is deterministic.
        try await Task.sleep(nanoseconds: 5_000_000)
        async let second = session.chat(
            messages: [ChatMessage(role: "user", content: "two")],
            sampling: SamplingParams())

        let outcomes = try await [first, second]
        #expect(outcomes.count == 2)

        let maxActive = await generator.tracker.maxActive
        let events = await generator.tracker.events
        #expect(maxActive == 1)
        #expect(events == ["+one", "-one", "+two", "-two"])
    }

    @Test func finishReasonStop() async throws {
        let generator = FakeTokenGenerator()
        generator.finishReason = .stop
        let outcome = try await makeSession(generator).complete(prompt: "p", sampling: SamplingParams())
        #expect(outcome.finishReason == .stop)
    }

    @Test func finishReasonLength() async throws {
        let generator = FakeTokenGenerator()
        generator.finishReason = .length
        let outcome = try await makeSession(generator).complete(prompt: "p", sampling: SamplingParams())
        #expect(outcome.finishReason == .length)
    }

    @Test func zeroMaxTokensRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.unsupportedParameter("max_tokens must be a positive integer")) {
            try await session.chat(messages: [ChatMessage(role: "user", content: "hi")],
                                   sampling: SamplingParams(maxTokens: 0))
        }
    }

    @Test func negativeMaxTokensRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.unsupportedParameter("max_tokens must be a positive integer")) {
            try await session.chat(messages: [ChatMessage(role: "user", content: "hi")],
                                   sampling: SamplingParams(maxTokens: -5))
        }
    }

    @Test func negativeTemperatureRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.unsupportedParameter("temperature must be >= 0")) {
            try await session.chat(messages: [ChatMessage(role: "user", content: "hi")],
                                   sampling: SamplingParams(temperature: -0.1))
        }
    }

    @Test func outOfRangeTopPRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.unsupportedParameter("top_p must be in (0, 1]")) {
            try await session.chat(messages: [ChatMessage(role: "user", content: "hi")],
                                   sampling: SamplingParams(topP: 0))
        }
        await #expect(throws: GenerationRequestError.unsupportedParameter("top_p must be in (0, 1]")) {
            try await session.chat(messages: [ChatMessage(role: "user", content: "hi")],
                                   sampling: SamplingParams(topP: 1.5))
        }
    }

    /// A waiter cancelled while parked in the slot queue must be removed and
    /// never run; the queued-behind request still gets served.
    @Test func cancelledWaiterNeverRuns() async throws {
        let generator = FakeTokenGenerator()
        generator.latencyNanoseconds = 50_000_000
        let session = makeSession(generator)

        async let first = session.chat(
            messages: [ChatMessage(role: "user", content: "one")],
            sampling: SamplingParams())
        try await Task.sleep(nanoseconds: 5_000_000)

        let parked = Task {
            try await session.chat(
                messages: [ChatMessage(role: "user", content: "parked")],
                sampling: SamplingParams())
        }
        try await Task.sleep(nanoseconds: 5_000_000)
        parked.cancel()

        await #expect(throws: CancellationError.self) { try await parked.value }
        _ = try await first

        let calls = await generator.calls
        #expect(calls.count == 1)
        let events = await generator.tracker.events
        #expect(events == ["+one", "-one"])

        // Slot was released: a fresh request runs to completion.
        _ = try await session.chat(messages: [ChatMessage(role: "user", content: "three")],
                                   sampling: SamplingParams())
        let finalCalls = await generator.calls
        #expect(finalCalls.count == 2)
    }

    /// A generator failure must free the slot; the next request is served.
    @Test func slotReleasedOnGenerationError() async throws {
        struct Boom: Error {}
        let generator = FakeTokenGenerator()
        let session = makeSession(generator)

        generator.thrownError = Boom()
        await #expect(throws: Boom.self) {
            try await session.chat(messages: [ChatMessage(role: "user", content: "one")],
                                   sampling: SamplingParams())
        }

        generator.thrownError = nil
        let outcome = try await session.chat(messages: [ChatMessage(role: "user", content: "two")],
                                             sampling: SamplingParams())
        #expect(outcome.text == "Hello, world")
    }

    /// max_tokens near Int.max must produce a 400-class error, not trap on
    /// checked `promptTokens + requested` overflow and kill the process.
    @Test func hugeMaxTokensRejectedWithoutOverflow() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(
            throws: GenerationRequestError.contextOverflow(prompt: 8, maxTokens: Int.max, maxContext: 64)
        ) {
            try await session.chat(
                messages: [ChatMessage(role: "user", content: "hi")],
                sampling: SamplingParams(maxTokens: Int.max))
        }
    }

    /// +Inf passes `temperature >= 0` and would otherwise fail deep in the
    /// runtime; reject it at the boundary like any other invalid value.
    @Test func infiniteTemperatureRejected() async {
        let session = makeSession(FakeTokenGenerator())
        await #expect(throws: GenerationRequestError.unsupportedParameter("temperature must be >= 0")) {
            try await session.chat(messages: [ChatMessage(role: "user", content: "hi")],
                                   sampling: SamplingParams(temperature: .infinity))
        }
    }

    /// A task cancelled exactly when the free slot is handed to it must not
    /// wedge the slot: releaseSlot is installed before the cancellation
    /// check, so the next request still runs.
    @Test func slotReleasedWhenCancelledAtHandoff() async throws {
        struct Timeout: Error {}
        let generator = FakeTokenGenerator()
        let session = makeSession(generator)

        let cancelled = Task {
            try await session.chat(messages: [ChatMessage(role: "user", content: "hi")],
                                   sampling: SamplingParams())
        }
        cancelled.cancel()
        _ = try? await cancelled.value

        // If the slot stayed wedged this hangs; bound the wait so the
        // regression fails instead of stalling the suite.
        try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask {
                let outcome = try await session.chat(
                    messages: [ChatMessage(role: "user", content: "next")],
                    sampling: SamplingParams())
                return outcome.text
            }
            group.addTask {
                try await Task.sleep(nanoseconds: 5_000_000_000)
                throw Timeout()
            }
            let text = try await group.next()
            group.cancelAll()
            #expect(text == "Hello, world")
        }
    }
}

@Suite("StopReason to finish_reason mapping")
struct StopReasonMappingTests {
    @Test func mapping() {
        // The mapping itself lives in the Metal-backed generator; these pin
        // the contract it implements against the runtime's StopReason cases.
        #expect(FinishReason.stop.rawValue == "stop")
        #expect(FinishReason.length.rawValue == "length")
    }
}
