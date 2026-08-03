import Testing
@testable import TurboFieldfare

@Suite struct PrefillRuntimeConfigTests {
    @Test(arguments: [32, 64, 128])
    func productionUsesCompleteChunkedPath(_ chunkTokens: Int) throws {
        let config = PrefillRuntimeConfig.production(chunkTokens: chunkTokens)
        #expect(config.mode == .chunked)
        #expect(config.chunkTokens == chunkTokens)
    }

    @Test func offDisablesChunkedPrefill() {
        let config = PrefillRuntimeConfig.off
        #expect(config.mode == .off)
        #expect(!config.enabled)
    }

    @Test func plannerUsesConfiguredChunkSize() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 130,
            startPosition: 7,
            config: .production(chunkTokens: 64))
        #expect(spans.map(\.tokenCount) == [64, 64, 2])
        #expect(spans.map(\.startPosition) == [7, 71, 135])
    }

    @Test func multimodalPlannerNeverSplitsVisualBlocks() {
        let spans = PrefillChunkPlanner.multimodalSpans(
            tokenCount: 500,
            startPosition: 0,
            chunkTokens: 128,
            visionRanges: [50..<330])
        #expect(spans.map(\.tokenCount) == [50, 280, 128, 42])
        #expect(spans.map(\.startPosition) == [0, 50, 330, 458])
        #expect(spans.map(\.kind) == [.text, .vision, .text, .text])
        #expect(spans.last?.completedCount == 500)
    }

    @Test func multimodalPlannerSupportsMultipleHistoryImages() {
        let spans = PrefillChunkPlanner.multimodalSpans(
            tokenCount: 450,
            startPosition: 10,
            chunkTokens: 64,
            visionRanges: [20..<90, 180..<320])
        #expect(spans.filter { $0.kind == .vision }.map(\.tokenCount) == [70, 140])
        #expect(spans.filter { $0.kind == .vision }.map(\.startPosition) == [20, 180])
        #expect(spans.allSatisfy { $0.kind == .vision || $0.tokenCount <= 64 })
    }

    @Test func diagnosticsPreserveUnknownValues() {
        let diagnostics = PrefillExecutionDiagnostics(
            config: .production(chunkTokens: 128),
            executedMode: .unsupported,
            kvStorageMode: nil,
            unsupportedReason: "unavailable")
        #expect(diagnostics.kvStorageMode == nil)
        #expect(diagnostics.chunkCompleteness == .unsupported)
        #expect(diagnostics.unsupportedReason == "unavailable")
    }
}
