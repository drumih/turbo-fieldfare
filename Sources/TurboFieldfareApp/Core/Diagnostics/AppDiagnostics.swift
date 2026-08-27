import Foundation
import TurboFieldfare

public enum AppStopReason: String, Equatable, Sendable {
    case maxTokens
    case cancelled
    case failed
    case eos
    case endOfTurn
    case stopString
    case toolCalls
}

public struct AppRunnerDiagnostics: Equatable, Sendable {
    public var cb1MillisecondsPerToken: Double
    public var ioMillisecondsPerToken: Double
    public var cb2MillisecondsPerToken: Double
    public var headMillisecondsPerToken: Double
    public var rdadviseMillisecondsPerToken: Double
    public var rdadviseCallsPerToken: Double
    public var rdadviseMegabytesPerToken: Double
    public var rdadviseSkippedPerToken: Double
    public var rdadviseFailures: UInt64

    public init(cb1MillisecondsPerToken: Double = 0,
                ioMillisecondsPerToken: Double = 0,
                cb2MillisecondsPerToken: Double = 0,
                headMillisecondsPerToken: Double = 0,
                rdadviseMillisecondsPerToken: Double = 0,
                rdadviseCallsPerToken: Double = 0,
                rdadviseMegabytesPerToken: Double = 0,
                rdadviseSkippedPerToken: Double = 0,
                rdadviseFailures: UInt64 = 0) {
        self.cb1MillisecondsPerToken = cb1MillisecondsPerToken
        self.ioMillisecondsPerToken = ioMillisecondsPerToken
        self.cb2MillisecondsPerToken = cb2MillisecondsPerToken
        self.headMillisecondsPerToken = headMillisecondsPerToken
        self.rdadviseMillisecondsPerToken = rdadviseMillisecondsPerToken
        self.rdadviseCallsPerToken = rdadviseCallsPerToken
        self.rdadviseMegabytesPerToken = rdadviseMegabytesPerToken
        self.rdadviseSkippedPerToken = rdadviseSkippedPerToken
        self.rdadviseFailures = rdadviseFailures
    }
}

public struct AppDiagnostics: Equatable, Sendable {
    public var generatedTokens: Int
    public var stopReason: AppStopReason
    public var promptTokenCount: Int?
    /// Prompt tokens served from the retained KV instead of prefilled again.
    /// Nil on the single-prompt path, where nothing is retained.
    public var cachedPromptTokens: Int?
    /// Tokens this turn put through prefill. A derivation of the two figures
    /// beside it (`RawCompletion` computes `promptIds.count -
    /// cachedPromptTokens`), so comparing it against them cannot fail. Carried
    /// for reporting, not as evidence.
    public var computedPrefillTokens: Int?
    /// Tokens the conversation's KV holds after this turn.
    public var conversationTokens: Int?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var peakMemoryBytes: UInt64?
    public var visionTowerMappedBytes: UInt64?
    public var runtimeOptions: AppRuntimeOptions
    public var prefill: PrefillExecutionDiagnostics?
    public var runner: AppRunnerDiagnostics?

    public var requestStartTimeToFirstTokenSeconds: Double? {
        guard let prefillSeconds, let timeToFirstTokenSeconds else { return nil }
        return prefillSeconds + timeToFirstTokenSeconds
    }

    public var prefillTokensPerSecond: Double? {
        guard let tokenCount = computedPrefillTokens ?? promptTokenCount,
              tokenCount > 0,
              let prefillSeconds,
              prefillSeconds.isFinite,
              prefillSeconds > 0 else {
            return nil
        }
        let rate = Double(tokenCount) / prefillSeconds
        return rate.isFinite ? rate : nil
    }

    public init(generatedTokens: Int,
                stopReason: AppStopReason,
                promptTokenCount: Int? = nil,
                cachedPromptTokens: Int? = nil,
                computedPrefillTokens: Int? = nil,
                conversationTokens: Int? = nil,
                prefillSeconds: Double? = nil,
                timeToFirstTokenSeconds: Double?,
                decodeSeconds: Double,
                tokensPerSecond: Double,
                peakMemoryBytes: UInt64?,
                visionTowerMappedBytes: UInt64? = nil,
                runtimeOptions: AppRuntimeOptions,
                prefill: PrefillExecutionDiagnostics? = nil,
                runner: AppRunnerDiagnostics? = nil) {
        self.generatedTokens = generatedTokens
        self.stopReason = stopReason
        self.promptTokenCount = promptTokenCount
        self.cachedPromptTokens = cachedPromptTokens
        self.computedPrefillTokens = computedPrefillTokens
        self.conversationTokens = conversationTokens
        self.prefillSeconds = prefillSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryBytes = peakMemoryBytes
        self.visionTowerMappedBytes = visionTowerMappedBytes
        self.runtimeOptions = runtimeOptions
        self.prefill = prefill
        self.runner = runner
    }
}

public struct AppTokenEvent: Equatable, Sendable {
    public var index: Int
    public var textDelta: String
    public var elapsedDecodeSeconds: Double
}

public enum AppInferenceEvent: Equatable, Sendable {
    case prefillProgress(done: Int, total: Int)
    case memorySample
    case token(AppTokenEvent)
    case finished(AppDiagnostics)
    case cancelled(AppDiagnostics)
    case failed(AppInferenceError, partial: AppDiagnostics?)
}
