import Foundation

public enum ReasoningControl: Sendable, Equatable {
    case binaryThinking
    case thinkingWithEffortLevels(
        defaultEffort: ModelReasoningEffort, supported: [ModelReasoningEffort])
    case effortOnly(defaultEffort: ModelReasoningEffort, supported: [ModelReasoningEffort])

    public var supportedLevels: [ReasoningLevel] {
        switch self {
        case .binaryThinking: [.off, .on]
        case .thinkingWithEffortLevels(_, let supported): [.off] + supported.map(ReasoningLevel.init)
        case .effortOnly(_, let supported): supported.map(ReasoningLevel.init)
        }
    }

    public var levelWhenOn: ReasoningLevel {
        switch self {
        case .binaryThinking: .on
        case .thinkingWithEffortLevels(let defaultEffort, _), .effortOnly(let defaultEffort, _):
            ReasoningLevel(defaultEffort)
        }
    }

    public func effectiveEffort(
        thinkingMode: ModelThinkingMode, effort: ModelReasoningEffort?
    ) -> ModelReasoningEffort? {
        switch self {
        case .binaryThinking: nil
        case .thinkingWithEffortLevels(let defaultEffort, _), .effortOnly(let defaultEffort, _):
            thinkingMode.isEnabled ? effort ?? defaultEffort : nil
        }
    }
}

public enum ReasoningLevel: String, CaseIterable, Sendable, Codable {
    case off, on, minimal, low, medium, high, xhigh, max

    public var displayName: String { self == .xhigh ? "extra high" : rawValue }

    public init(_ effort: ModelReasoningEffort) {
        switch effort {
        case .low: self = .low
        case .medium: self = .medium
        case .high: self = .high
        case .xhigh: self = .xhigh
        }
    }

    public static func requested(_ raw: String) -> ReasoningLevel? {
        let word = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch word {
        case "": return nil
        case "0", "off", "false", "no", "none", "disabled", "nothink", "no_think": return .off
        case "1", "on", "true", "yes", "enabled", "think", "thinking", "auto", "adaptive",
            "default": return .on
        case "minimal", "min", "least", "tiny": return .minimal
        case "low", "light": return .low
        case "medium", "med", "moderate", "normal", "standard", "balanced": return .medium
        case "high", "hard": return .high
        case "xhigh", "extra-high", "extra_high", "extra high", "x-high", "very-high", "highest":
            return .xhigh
        case "max", "maximum", "ultra", "extreme", "unlimited": return .max
        default: return ReasoningLevel(rawValue: word)
        }
    }

    public var effort: ModelReasoningEffort? { ModelReasoningEffort(rawValue: rawValue) }
}

public struct RequestReasoning: Sendable, Equatable {
    public let thinkingMode: ModelThinkingMode
    public let effort: ModelReasoningEffort?

    public init(thinkingMode: ModelThinkingMode, effort: ModelReasoningEffort?) {
        self.thinkingMode = thinkingMode
        self.effort = thinkingMode.isEnabled ? effort : nil
    }

    public func matches(_ loaded: RequestReasoning) -> Bool { self == loaded }
}
