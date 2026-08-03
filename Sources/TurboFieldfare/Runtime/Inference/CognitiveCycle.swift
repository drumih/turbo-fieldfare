import Foundation

/// One completed pass of the cognitive cycle, in display order.
public struct CognitivePassSegment: Equatable, Sendable, Identifiable {
    public let id = UUID()
    public let kind: CognitiveStepKind
    public let text: String

    public init(kind: CognitiveStepKind, text: String) {
        self.kind = kind
        self.text = text
    }
}

/// The four passes of the advanced cognitive cycle.
public enum CognitiveStepKind: String, CaseIterable, Sendable, Equatable, Hashable {
    case plan
    case draft
    case critique
    case final

    /// Header shown in the output pane before each pass.
    public var header: String {
        switch self {
        case .plan: return "Plan"
        case .draft: return "Draft"
        case .critique: return "Critique"
        case .final: return "Final answer"
        }
    }
}

/// One generation step of a cognitive cycle.
public struct CognitiveStep: Equatable, Sendable {
    public let kind: CognitiveStepKind
    public let prompt: String

    public init(kind: CognitiveStepKind, prompt: String) {
        self.kind = kind
        self.prompt = prompt
    }
}

/// Stateful builder of the advanced cognitive cycle: plan, draft, critique,
/// then a final revision. Each prompt embeds the outputs of the previous
/// passes, so the model revises with full context. The engine is a pure
/// value type: advance it with `nextStep()` and feed outputs back with
/// `record(output:)`.
public struct CognitiveCycleEngine: Sendable, Equatable {
    public let userPrompt: String
    public private(set) var outputs: [CognitiveStepKind: String] = [:]
    public private(set) var isFinished = false

    public init(userPrompt: String) {
        self.userPrompt = userPrompt
    }

    /// The next step to run, or nil when the cycle is complete.
    public mutating func nextStep() -> CognitiveStep? {
        guard !isFinished, let kind = nextKind else { return nil }
        return CognitiveStep(kind: kind, prompt: makePrompt(for: kind))
    }

    /// Records the output of the most recent step. The cycle advances only
    /// when every step has produced output.
    public mutating func record(output: String) {
        guard let kind = nextKind else { return }
        outputs[kind] = output
        if nextKind == nil {
            isFinished = true
        }
    }

    private var nextKind: CognitiveStepKind? {
        CognitiveStepKind.allCases.first { outputs[$0] == nil }
    }

    private func makePrompt(for kind: CognitiveStepKind) -> String {
        switch kind {
        case .plan:
            return """
            Plan your answer to the request below. List the main points in brief note form; do not write the full answer yet.

            Request:
            \(userPrompt)
            """
        case .draft:
            return """
            Write the complete answer to the request, following the plan.

            Request:
            \(userPrompt)

            Plan:
            \(outputs[.plan] ?? "")
            """
        case .critique:
            return """
            Critique the draft below. List concrete errors, missing facts, and weak arguments. Be specific and do not rewrite the answer.

            Request:
            \(userPrompt)

            Draft:
            \(outputs[.draft] ?? "")
            """
        case .final:
            return """
            Revise the draft to address the critique. Output only the final, complete answer.

            Request:
            \(userPrompt)

            Draft:
            \(outputs[.draft] ?? "")

            Critique:
            \(outputs[.critique] ?? "")
            """
        }
    }
}

/// Decides whether the final revision pass can be skipped because the
/// critique found nothing actionable. Heuristic, prompt-format dependent.
public enum CognitiveStopPolicy {
    /// Returns true when the critique is empty or contains only positive
    /// "nothing to fix" markers, so the draft can be kept as the answer.
    public static func shouldSkipFinal(critique: String) -> Bool {
        let trimmed = critique.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        let lowered = trimmed.lowercased()
        let positiveMarkers = [
            "no issues", "no errors", "no problems", "no changes",
            "no improvements", "nothing to fix", "nothing to change",
            "looks good", "no major issues",
            "no actionable feedback", "no critical issues",
        ]
        return positiveMarkers.contains { lowered.contains($0) }
    }
}
