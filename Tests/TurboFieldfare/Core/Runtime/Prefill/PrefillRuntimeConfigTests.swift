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

    @Test func plannerKeepsExactBoundariesAndProgressOffsets() {
        let spans = PrefillChunkPlanner.spans(
            tokenCount: 128,
            startPosition: 11,
            chunkTokens: 64)
        #expect(spans.count == 2)
        #expect(spans[0].tokenOffset == 0)
        #expect(spans[0].tokenCount == 64)
        #expect(spans[0].startPosition == 11)
        #expect(spans[0].completedCount == 64)
        #expect(spans[1].tokenOffset == 64)
        #expect(spans[1].tokenCount == 64)
        #expect(spans[1].startPosition == 75)
        #expect(spans[1].completedCount == 128)
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

    @Test func scalarCallsAreNotReportedAsChunkedWork() throws {
        var counter = PrefillWorkCounter()
        for _ in 0..<128 {
            counter.recordScalarForward()
        }
        counter.recordCommandBuffers(128 * 4)

        let diagnostics = try #require(counter.diagnostics)
        #expect(diagnostics.executionPath == .scalarFallback)
        #expect(diagnostics.scalarForwardCount == 128)
        #expect(diagnostics.chunkPassCount == 0)
        #expect(diagnostics.commandBufferCount == 512)

        let execution = PrefillExecutionDiagnostics(
            config: .production(chunkTokens: 128),
            executedMode: .chunked,
            kvStorageMode: .fp16,
            work: diagnostics)
        #expect(execution.requestedMode == .chunked)
        #expect(execution.executedMode == .scalarFallback)
        #expect(execution.scalarForwardCount == 128)
        #expect(execution.chunkPassCount == 0)
        #expect(execution.commandBufferCount == 512)
    }

    @Test func mergedChunkWorkRetainsAllPassesAndCommands() throws {
        var counter = PrefillWorkCounter()
        counter.merge(PrefillWorkDiagnostics(executionPath: .chunked,
                                              scalarForwardCount: 0,
                                              chunkPassCount: 42,
                                              commandBufferCount: 160,
                                              embeddingNanos: 1,
                                              mixerNanos: 2,
                                              moePrepareNanos: 3,
                                              expertFetchNanos: 4,
                                              routedMoENanos: 5,
                                              moeReduceNanos: 6,
                                              finalHeadNanos: 7))
        counter.merge(PrefillWorkDiagnostics(executionPath: .chunked,
                                              scalarForwardCount: 0,
                                              chunkPassCount: 21,
                                              commandBufferCount: 80,
                                              embeddingNanos: 10,
                                              mixerNanos: 20,
                                              moePrepareNanos: 30,
                                              expertFetchNanos: 40,
                                              routedMoENanos: 50,
                                              moeReduceNanos: 60,
                                              finalHeadNanos: 70))

        let diagnostics = try #require(counter.diagnostics)
        #expect(diagnostics.executionPath == .chunked)
        #expect(diagnostics.scalarForwardCount == 0)
        #expect(diagnostics.chunkPassCount == 63)
        #expect(diagnostics.commandBufferCount == 240)
        #expect(diagnostics.embeddingNanos == 11)
        #expect(diagnostics.mixerNanos == 22)
        #expect(diagnostics.moePrepareNanos == 33)
        #expect(diagnostics.expertFetchNanos == 44)
        #expect(diagnostics.routedMoENanos == 55)
        #expect(diagnostics.moeReduceNanos == 66)
        #expect(diagnostics.finalHeadNanos == 77)
    }
}
