import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppDiagnosticsTests {
    @Test func requestStartTTFTAddsPrefillAndPostPrefillWait() {
        let diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            prefillSeconds: 1.25,
            timeToFirstTokenSeconds: 0.5,
            decodeSeconds: 0.75,
            tokensPerSecond: 1.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        #expect(diagnostics.requestStartTimeToFirstTokenSeconds == 1.75)
    }

    @Test func requestStartTTFTIsNilWhenEitherSideIsMissing() {
        let missingPrefill = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            prefillSeconds: nil,
            timeToFirstTokenSeconds: 0.5,
            decodeSeconds: 0.75,
            tokensPerSecond: 1.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        let missingFirstToken = AppDiagnostics(
            generatedTokens: 0,
            stopReason: .cancelled,
            prefillSeconds: 1.25,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        #expect(missingPrefill.requestStartTimeToFirstTokenSeconds == nil)
        #expect(missingFirstToken.requestStartTimeToFirstTokenSeconds == nil)
    }

    @Test func prefillRateUsesComputedTokensAndPrefillDuration() {
        var diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            promptTokenCount: 20,
            computedPrefillTokens: 5,
            prefillSeconds: 6.17,
            timeToFirstTokenSeconds: 0.5,
            decodeSeconds: 0.75,
            tokensPerSecond: 1.0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        #expect(abs((diagnostics.prefillTokensPerSecond ?? 0) - 0.810373) < 0.000001)

        diagnostics.computedPrefillTokens = 0
        #expect(diagnostics.prefillTokensPerSecond == nil)
        diagnostics.computedPrefillTokens = 5
        diagnostics.prefillSeconds = 0
        #expect(diagnostics.prefillTokensPerSecond == nil)
        diagnostics.prefillSeconds = .nan
        #expect(diagnostics.prefillTokensPerSecond == nil)
    }

    @Test func prefillRateFallsBackToPromptTokensForOlderEvents() {
        let diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            promptTokenCount: 20,
            computedPrefillTokens: nil,
            prefillSeconds: 5,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())

        #expect(diagnostics.prefillTokensPerSecond == 4)
    }

    @Test func runnerDiagnosticsRetainPublicResultAndAdvancedMetrics() {
        let diagnostics = AppRunnerDiagnostics(
            cb1MillisecondsPerToken: 1,
            ioMillisecondsPerToken: 2,
            cb2MillisecondsPerToken: 3,
            headMillisecondsPerToken: 4,
            rdadviseMillisecondsPerToken: 5,
            rdadviseCallsPerToken: 6,
            rdadviseMegabytesPerToken: 7,
            rdadviseSkippedPerToken: 8,
            rdadviseFailures: 9)

        #expect(diagnostics.cb1MillisecondsPerToken == 1)
        #expect(diagnostics.ioMillisecondsPerToken == 2)
        #expect(diagnostics.cb2MillisecondsPerToken == 3)
        #expect(diagnostics.headMillisecondsPerToken == 4)
        #expect(diagnostics.rdadviseMillisecondsPerToken == 5)
        #expect(diagnostics.rdadviseCallsPerToken == 6)
        #expect(diagnostics.rdadviseMegabytesPerToken == 7)
        #expect(diagnostics.rdadviseSkippedPerToken == 8)
        #expect(diagnostics.rdadviseFailures == 9)
    }
}
