import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelLiveMetricsTests {
    @MainActor
    @Test func prefillProgressUpdatesPhaseAndCounts() {
        let model = AppModel()
        model.apply(.prefillProgress(done: 12, total: 50))
        #expect(model.phase == .prefill)
        #expect(model.livePrefillDone == 12)
        #expect(model.livePrefillTotal == 50)
    }

    @MainActor
    @Test func tokenEventSwitchesToDecodeAndComputesRate() {
        let model = AppModel()
        model.apply(.token(AppTokenEvent(
            index: 4, textDelta: " x", elapsedDecodeSeconds: 2.0)))
        #expect(model.phase == .decode)
        #expect(model.liveTokenCount == 5)
        #expect(model.liveElapsedDecodeSeconds == 2.0)
        #expect(model.liveTokensPerSecond == 2.5)
        #expect(model.liveMemoryBytes != nil)
    }

    @MainActor
    @Test func liveRateIsZeroBeforeAnyToken() {
        let model = AppModel()
        #expect(model.liveTokensPerSecond == 0)
    }

    @MainActor
    @Test func terminalEventsReturnPhaseToIdleKeepingLastValues() {
        let model = AppModel()
        model.apply(.token(AppTokenEvent(
            index: 2, textDelta: "a", elapsedDecodeSeconds: 1.0)))
        model.apply(.finished(AppDiagnostics(
            generatedTokens: 3, stopReason: .eos,
            timeToFirstTokenSeconds: 0.1, decodeSeconds: 1.0, tokensPerSecond: 3,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())))
        #expect(model.phase == .idle)
        #expect(model.liveTokenCount == 3)
    }

    @MainActor
    @Test func runResetsLiveMetricsAndEntersPrefill() async {
        let client = MockInferenceClient(response: "one two", tokenDelayNanos: 1_000_000_000)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.apply(.token(AppTokenEvent(
            index: 9, textDelta: "z", elapsedDecodeSeconds: 5)))
        model.promptText = "go"
        model.run()
        #expect(model.phase == .prefill)
        #expect(model.liveTokenCount == 0)
        #expect(model.liveElapsedDecodeSeconds == 0)
        #expect(model.livePrefillDone == 0)
        model.cancel()
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    @MainActor
    @Test func cancelKeepsPartialLiveMetrics() {
        let model = AppModel()
        model.apply(.token(AppTokenEvent(
            index: 1, textDelta: "p", elapsedDecodeSeconds: 0.5)))
        model.apply(.cancelled(AppDiagnostics(
            generatedTokens: 2, stopReason: .cancelled,
            timeToFirstTokenSeconds: nil, decodeSeconds: 0.5, tokensPerSecond: 4,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())))
        #expect(model.phase == .idle)
        #expect(model.liveTokenCount == 2)
        #expect(model.error == .cancelled)
    }
}
