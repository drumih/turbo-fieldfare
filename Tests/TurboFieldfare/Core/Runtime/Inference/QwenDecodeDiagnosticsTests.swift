import Testing
@testable import TurboFieldfare

struct QwenDecodeDiagnosticsTests {
    @Test func preservesMeasuredDecodeCounters() {
        let diagnostics = QwenDecodeDiagnostics(
            wallNanos: 100,
            embeddingNanos: 10,
            layerNanos: 70,
            logitsNanos: 20,
            expertFetchNanos: 30,
            layerCount: 4,
            fullAttentionLayerCount: 1,
            deltaNetLayerCount: 3,
            commandBufferCount: 9,
            routerEvaluationCount: 4,
            routedExpertCount: 8,
            routedExpertCacheHitCount: 5,
            routedExpertCacheMissCount: 3,
            routedExpertEstimatedBytes: 1234,
            expertReadCount: 3,
            expertReadNanos: 24,
            expertReadMaxNanos: 11)

        #expect(diagnostics.wallNanos == 100)
        #expect(diagnostics.embeddingNanos == 10)
        #expect(diagnostics.layerNanos == 70)
        #expect(diagnostics.logitsNanos == 20)
        #expect(diagnostics.expertFetchNanos == 30)
        #expect(diagnostics.layerCount == 4)
        #expect(diagnostics.fullAttentionLayerCount == 1)
        #expect(diagnostics.deltaNetLayerCount == 3)
        #expect(diagnostics.commandBufferCount == 9)
        #expect(diagnostics.routerEvaluationCount == 4)
        #expect(diagnostics.routedExpertCount == 8)
        #expect(diagnostics.routedExpertCacheHitCount == 5)
        #expect(diagnostics.routedExpertCacheMissCount == 3)
        #expect(diagnostics.routedExpertEstimatedBytes == 1234)
        #expect(diagnostics.expertReadCount == 3)
        #expect(diagnostics.expertReadNanos == 24)
        #expect(diagnostics.expertReadMaxNanos == 11)
        #expect(diagnostics.layers.isEmpty)
    }

    @Test func aggregateSupportsZeroDecodeSteps() {
        var accumulator = QwenDecodeDiagnosticsAggregateAccumulator()
        accumulator.addSamplingNanos(20)

        let aggregate = accumulator.makeDiagnostics(decodeLoopWallNanos: 100)
        #expect(aggregate.schemaVersion == 3)
        #expect(aggregate.decodeStepCount == 0)
        #expect(aggregate.forwardWallNanos == 0)
        #expect(aggregate.samplingNanos == 20)
        #expect(aggregate.attributedWallNanos == 20)
        #expect(aggregate.residualWallNanos == 80)
        #expect(aggregate.expertReadCount == 0)
        #expect(aggregate.expertReadNanos == 0)
        #expect(aggregate.expertReadMaxNanos == 0)
        #expect(aggregate.layers.isEmpty)
    }

    @Test func aggregateMergesLayerTotalsAndAttribution() {
        var accumulator = QwenDecodeDiagnosticsAggregateAccumulator()
        accumulator.add(QwenDecodeDiagnostics(
            wallNanos: 100,
            embeddingNanos: 10,
            layerNanos: 70,
            logitsNanos: 20,
            expertFetchNanos: 30,
            layerCount: 2,
            fullAttentionLayerCount: 1,
            deltaNetLayerCount: 1,
            commandBufferCount: 4,
            routerEvaluationCount: 2,
            routedExpertCount: 4,
            routedExpertCacheHitCount: 3,
            routedExpertCacheMissCount: 1,
            routedExpertEstimatedBytes: 100,
            mixerNanos: 20,
            routerNanos: 25,
            routePlanningNanos: 5,
            sharedExpertNanos: 30,
            routedExpertCombineNanos: 15,
            expertReadCount: 1,
            expertReadNanos: 9,
            expertReadMaxNanos: 9,
            layers: [
                QwenDecodeLayerDiagnostics(layer: 0,
                                           isFullAttention: true,
                                           elapsedNanos: 40,
                                           expertFetchNanos: 10,
                                           expertReadCount: 1,
                                           expertReadNanos: 9,
                                           expertReadMaxNanos: 9,
                                           routedExperts: [4, 7],
                                           routingWeights: [0.75, 0.25]),
                QwenDecodeLayerDiagnostics(layer: 1,
                                           isFullAttention: false,
                                           elapsedNanos: 30,
                                           expertFetchNanos: 20),
            ]))
        accumulator.add(QwenDecodeDiagnostics(
            wallNanos: 150,
            embeddingNanos: 15,
            layerNanos: 100,
            logitsNanos: 35,
            expertFetchNanos: 45,
            layerCount: 2,
            fullAttentionLayerCount: 1,
            deltaNetLayerCount: 1,
            commandBufferCount: 4,
            routerEvaluationCount: 2,
            routedExpertCount: 4,
            routedExpertCacheHitCount: 2,
            routedExpertCacheMissCount: 2,
            routedExpertEstimatedBytes: 200,
            mixerNanos: 30,
            routerNanos: 35,
            routePlanningNanos: 10,
            sharedExpertNanos: 40,
            routedExpertCombineNanos: 20,
            expertReadCount: 2,
            expertReadNanos: 21,
            expertReadMaxNanos: 12,
            layers: [
                QwenDecodeLayerDiagnostics(layer: 0,
                                           isFullAttention: true,
                                           elapsedNanos: 60,
                                           expertFetchNanos: 15,
                                           expertReadCount: 2,
                                           expertReadNanos: 21,
                                           expertReadMaxNanos: 12,
                                           routedExperts: [7, 9],
                                           routingWeights: [0.6, 0.4]),
                QwenDecodeLayerDiagnostics(layer: 1,
                                           isFullAttention: false,
                                           elapsedNanos: 40,
                                           expertFetchNanos: 30),
            ]))
        accumulator.addSamplingNanos(50)

        let aggregate = accumulator.makeDiagnostics(decodeLoopWallNanos: 350)
        #expect(aggregate.decodeStepCount == 2)
        #expect(aggregate.forwardWallNanos == 250)
        #expect(aggregate.embeddingNanos == 25)
        #expect(aggregate.layerNanos == 170)
        #expect(aggregate.logitsNanos == 55)
        #expect(aggregate.mixerNanos == 50)
        #expect(aggregate.routerNanos == 60)
        #expect(aggregate.routePlanningNanos == 15)
        #expect(aggregate.sharedExpertNanos == 70)
        #expect(aggregate.expertFetchNanos == 75)
        #expect(aggregate.routedExpertCombineNanos == 35)
        #expect(aggregate.commandBufferSubmissionCount == 8)
        #expect(aggregate.routerEvaluationCount == 4)
        #expect(aggregate.routedExpertCount == 8)
        #expect(aggregate.routedExpertCacheHitCount == 5)
        #expect(aggregate.routedExpertCacheMissCount == 3)
        #expect(aggregate.routedExpertEstimatedBytes == 300)
        #expect(aggregate.expertReadCount == 3)
        #expect(aggregate.expertReadNanos == 30)
        #expect(aggregate.expertReadMaxNanos == 12)
        #expect(aggregate.samplingNanos == 50)
        #expect(aggregate.attributedWallNanos == 300)
        #expect(aggregate.residualWallNanos == 50)
        #expect(aggregate.layers.map(\.elapsedNanos) == [100, 70])
        #expect(aggregate.layers.map(\.expertFetchNanos) == [25, 50])
        #expect(aggregate.layers.map(\.expertReadCount) == [3, 0])
        #expect(aggregate.layers.map(\.expertReadNanos) == [30, 0])
        #expect(aggregate.layers.map(\.expertReadMaxNanos) == [12, 0])
        #expect(aggregate.layers[0].routedExpertTrace == [[4, 7], [7, 9]])
        #expect(aggregate.layers[0].routingWeightTrace == [[0.75, 0.25], [0.6, 0.4]])
    }

    @Test func aggregateSaturatesCounterOverflow() {
        var accumulator = QwenDecodeDiagnosticsAggregateAccumulator()
        accumulator.add(QwenDecodeDiagnostics(
            wallNanos: .max,
            embeddingNanos: .max,
            layerNanos: .max,
            logitsNanos: .max,
            expertFetchNanos: .max,
            layerCount: 1,
            fullAttentionLayerCount: 1,
            deltaNetLayerCount: 0,
            commandBufferCount: .max,
            routerEvaluationCount: .max,
            routedExpertCount: .max,
            routedExpertCacheHitCount: .max,
            routedExpertCacheMissCount: .max,
            routedExpertEstimatedBytes: .max,
            mixerNanos: .max,
            routerNanos: .max,
            routePlanningNanos: .max,
            sharedExpertNanos: .max,
            routedExpertCombineNanos: .max,
            expertReadCount: .max,
            expertReadNanos: .max,
            expertReadMaxNanos: .max,
            layers: [QwenDecodeLayerDiagnostics(layer: 0,
                                                 isFullAttention: true,
                                                 elapsedNanos: .max,
                                                 expertFetchNanos: .max,
                                                 expertReadCount: .max,
                                                 expertReadNanos: .max,
                                                 expertReadMaxNanos: .max)]))
        accumulator.add(QwenDecodeDiagnostics(
            wallNanos: 1,
            embeddingNanos: 1,
            layerNanos: 1,
            logitsNanos: 1,
            expertFetchNanos: 1,
            layerCount: 1,
            fullAttentionLayerCount: 1,
            deltaNetLayerCount: 0,
            commandBufferCount: 1,
            routerEvaluationCount: 1,
            routedExpertCount: 1,
            routedExpertCacheHitCount: 1,
            routedExpertCacheMissCount: 1,
            routedExpertEstimatedBytes: 1,
            mixerNanos: 1,
            routerNanos: 1,
            routePlanningNanos: 1,
            sharedExpertNanos: 1,
            routedExpertCombineNanos: 1,
            expertReadCount: 1,
            expertReadNanos: 1,
            expertReadMaxNanos: 1,
            layers: [QwenDecodeLayerDiagnostics(layer: 0,
                                                 isFullAttention: true,
                                                 elapsedNanos: 1,
                                                 expertFetchNanos: 1,
                                                 expertReadCount: 1,
                                                 expertReadNanos: 1,
                                                 expertReadMaxNanos: 1)]))

        let aggregate = accumulator.makeDiagnostics(decodeLoopWallNanos: .max)
        #expect(aggregate.forwardWallNanos == .max)
        #expect(aggregate.commandBufferSubmissionCount == .max)
        #expect(aggregate.routerEvaluationCount == .max)
        #expect(aggregate.routedExpertEstimatedBytes == .max)
        #expect(aggregate.expertReadCount == .max)
        #expect(aggregate.expertReadNanos == .max)
        #expect(aggregate.expertReadMaxNanos == .max)
        #expect(aggregate.layers.first?.elapsedNanos == .max)
        #expect(aggregate.layers.first?.expertFetchNanos == .max)
        #expect(aggregate.layers.first?.expertReadCount == .max)
        #expect(aggregate.layers.first?.expertReadNanos == .max)
        #expect(aggregate.layers.first?.expertReadMaxNanos == .max)
    }
}