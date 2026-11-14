import TinyTitan
import TinyTitanCore

/// Where a generation's decoded output goes, the same for every engine.
///
/// Both engines hand this what `StructuredAssistantDecoder` made of their
/// tokens, so a thought is routed the same way whichever one produced it.
/// The split it keeps is between what the person reads and what the model
/// thought on the way there:
///
///   * **Visible text** passes through the client's stop strings, and each
///     piece that survives is shown to whatever watches the answer -- the
///     watchdogs, on the GPU path. A stop string is a promise about the
///     answer; a model that mentions one while reasoning has not ended it.
///   * **Reasoning** goes straight to the client and is collected apart.
///     Nothing that judges the answer reads it: a model thinking at length
///     before a short reply is working, and the loop and stub detectors were
///     calibrated on answers.
///   * **Tool calls** are collected and forwarded as they are.
package struct AssistantOutput {
    private var stopMatcher: StreamingStopMatcher
    private let onEvent: @Sendable (ServerInferenceEvent) -> Void
    private let observeVisible: (String) -> Void
    private let observeReasoning: (String) -> Void
    package private(set) var content = ""
    package private(set) var reasoning = ""
    package private(set) var calls: [ParsedToolCall] = []
    /// Generated tokens whose text landed in the thought channel.
    ///
    /// Counted where the channel split happens rather than derived from the
    /// reasoning text afterwards: detokenizing and re-tokenizing is not an
    /// identity, so only the decoder's own per-token verdict is the number the
    /// usage object can honestly advertise as `reasoning_tokens`.
    package private(set) var reasoningTokens = 0

    /// `observeReasoning` sees the thought text and nothing else -- on the
    /// GPU path, the loop detector that watches reasoning on its own.
    package init(
        stops: [String],
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void,
        observeVisible: @escaping (String) -> Void = { _ in },
        observeReasoning: @escaping (String) -> Void = { _ in }
    ) {
        self.stopMatcher = StreamingStopMatcher(stops: stops)
        self.onEvent = onEvent
        self.observeVisible = observeVisible
        self.observeReasoning = observeReasoning
    }

    /// True once a client stop string has matched; generation should end.
    package var isStopped: Bool { stopMatcher.isStopped }

    /// The stop string that matched, which the Messages API names.
    package var matchedStop: String? { stopMatcher.matchedStop }

    /// `isToken` is false for the flush at the end of generation: released tail
    /// text is not a generated token, so it must not be counted as one.
    package mutating func publish(_ events: [StructuredAssistantEvent], isToken: Bool = true) {
        if isToken, events.contains(where: \.isReasoning) {
            reasoningTokens += 1
        }
        for event in events {
            switch event {
            case .content(let text):
                let visible = stopMatcher.push(text)
                guard !visible.isEmpty else { continue }
                content += visible
                onEvent(.content(visible))
                observeVisible(visible)
            case .reasoning(let text):
                reasoning += text
                onEvent(.reasoning(text))
                observeReasoning(text)
            case .toolCall(let call):
                calls.append(call)
                onEvent(.toolCall(call))
            }
        }
    }

    /// Releases what the stop matcher held back as a possible partial match.
    package mutating func finish() {
        let tail = stopMatcher.finish()
        guard !tail.isEmpty else { return }
        content += tail
        onEvent(.content(tail))
    }
}
