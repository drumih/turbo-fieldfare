import Metal
import Testing
@testable import TurboFieldfare

struct QwenDecodeDiagnosticsTests {
    @Test func gpuStageTimerResolvesMetalMarkerPasses() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let timer = try #require(QwenGPUStageTimer(device: device))
        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())

        for marker in QwenGPUStageMarker.allCases {
            #expect(timer.encodeMarker(marker, commandBuffer: commandBuffer))
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        #expect(commandBuffer.status == .completed)
        #expect(timer.resolve() != nil)

        let routedCommandBuffer = try #require(queue.makeCommandBuffer())
        for marker in QwenRoutedGPUStageMarker.allCases {
            #expect(timer.encodeMarker(marker, commandBuffer: routedCommandBuffer))
        }
        routedCommandBuffer.commit()
        routedCommandBuffer.waitUntilCompleted()

        #expect(routedCommandBuffer.status == .completed)
        #expect(timer.resolveRouted() != nil)
    }

    @Test func gpuStageTimingsResolveExclusiveMarkerIntervals() throws {
        let timings = try #require(QwenGPUStageTimings(timestamps: [
            10, 12,
            30, 33,
            52, 55,
            70, 73,
        ]))

        #expect(timings.mixerNanos == 18)
        #expect(timings.sharedExpertNanos == 19)
        #expect(timings.routerNanos == 15)
    }

    @Test func routedGPUStageTimingsResolveExclusiveMarkerIntervals() throws {
        let timings = try #require(QwenRoutedGPUStageTimings(timestamps: [
            10, 12,
            30, 33,
            52, 55,
            70, 73,
        ]))

        #expect(timings.phase1Nanos == 18)
        #expect(timings.phase2Nanos == 19)
        #expect(timings.combineNanos == 15)
    }

    @Test(arguments: [
        [UInt64](),
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 0, 6, 7, 8],
        [1, 2, 3, 4, 5, .max, 7, 8],
        [1, 2, 4, 3, 5, 6, 7, 8],
    ])
    func gpuStageTimingsRejectInvalidSamples(_ timestamps: [UInt64]) {
        #expect(QwenGPUStageTimings(timestamps: timestamps) == nil)
        #expect(QwenRoutedGPUStageTimings(timestamps: timestamps) == nil)
    }

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
            mixerNanos: 20,
            routerNanos: 25,
            routePlanningNanos: 5,
            sharedExpertNanos: 30,
            routedSetupNanos: 6,
            routedCommandBufferEncodingNanos: 7,
            routedCommandBufferCommitNanos: 8,
            routedCommandBufferWaitNanos: 9,
            routedGPUStageTimingSampleCount: 4,
            gpuRoutedPhase1Nanos: 14,
            gpuRoutedPhase2Nanos: 15,
            gpuRoutedCombineNanos: 16,
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
        #expect(diagnostics.routedSetupNanos == 6)
        #expect(diagnostics.routedCommandBufferEncodingNanos == 7)
        #expect(diagnostics.routedCommandBufferCommitNanos == 8)
        #expect(diagnostics.routedCommandBufferWaitNanos == 9)
        #expect(diagnostics.routedGPUStageTimingSampleCount == 4)
        #expect(diagnostics.gpuRoutedPhase1Nanos == 14)
        #expect(diagnostics.gpuRoutedPhase2Nanos == 15)
        #expect(diagnostics.gpuRoutedCombineNanos == 16)
        #expect(diagnostics.expertReadCount == 3)
        #expect(diagnostics.expertReadNanos == 24)
        #expect(diagnostics.expertReadMaxNanos == 11)
        #expect(diagnostics.layers.isEmpty)
    }

    @Test func aggregateSupportsZeroDecodeSteps() {
        var accumulator = QwenDecodeDiagnosticsAggregateAccumulator()
        accumulator.addSamplingNanos(20)

        let aggregate = accumulator.makeDiagnostics(decodeLoopWallNanos: 100)
        #expect(aggregate.schemaVersion == 5)
        #expect(aggregate.decodeStepCount == 0)
        #expect(aggregate.forwardWallNanos == 0)
        #expect(aggregate.samplingNanos == 20)
        #expect(aggregate.attributedWallNanos == 20)
        #expect(aggregate.residualWallNanos == 80)
        #expect(aggregate.expertReadCount == 0)
        #expect(aggregate.expertReadNanos == 0)
        #expect(aggregate.expertReadMaxNanos == 0)
        #expect(aggregate.routedSetupNanos == 0)
        #expect(aggregate.routedCommandBufferEncodingNanos == 0)
        #expect(aggregate.routedCommandBufferCommitNanos == 0)
        #expect(aggregate.routedCommandBufferWaitNanos == 0)
        #expect(aggregate.routedGPUStageTimingSampleCount == 0)
        #expect(aggregate.gpuRoutedPhase1Nanos == 0)
        #expect(aggregate.gpuRoutedPhase2Nanos == 0)
        #expect(aggregate.gpuRoutedCombineNanos == 0)
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
            routedSetupNanos: 6,
            routedCommandBufferEncodingNanos: 7,
            routedCommandBufferCommitNanos: 8,
            routedCommandBufferWaitNanos: 9,
            gpuStageTimingSampleCount: 2,
            gpuMixerNanos: 12,
            gpuSharedExpertNanos: 8,
            gpuRouterNanos: 3,
            routedGPUStageTimingSampleCount: 2,
            gpuRoutedPhase1Nanos: 4,
            gpuRoutedPhase2Nanos: 5,
            gpuRoutedCombineNanos: 6,
            routedExpertCombineNanos: 15,
            expertReadCount: 1,
            expertReadNanos: 9,
            expertReadMaxNanos: 9,
            layers: [
                QwenDecodeLayerDiagnostics(layer: 0,
                                           isFullAttention: true,
                                           elapsedNanos: 40,
                                           expertFetchNanos: 10,
                                           routePlanningNanos: 2,
                                           routedSetupNanos: 3,
                                           routedCommandBufferEncodingNanos: 4,
                                           routedCommandBufferCommitNanos: 5,
                                           routedCommandBufferWaitNanos: 6,
                                           expertReadCount: 1,
                                           expertReadNanos: 9,
                                           expertReadMaxNanos: 9,
                                           gpuStageTimingSampled: true,
                                           gpuMixerNanos: 7,
                                           gpuSharedExpertNanos: 4,
                                           gpuRouterNanos: 2,
                                           routedGPUStageTimingSampled: true,
                                           gpuRoutedPhase1Nanos: 3,
                                           gpuRoutedPhase2Nanos: 4,
                                           gpuRoutedCombineNanos: 5,
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
            routedSetupNanos: 11,
            routedCommandBufferEncodingNanos: 12,
            routedCommandBufferCommitNanos: 13,
            routedCommandBufferWaitNanos: 14,
            gpuStageTimingSampleCount: 2,
            gpuMixerNanos: 18,
            gpuSharedExpertNanos: 11,
            gpuRouterNanos: 5,
            routedGPUStageTimingSampleCount: 2,
            gpuRoutedPhase1Nanos: 7,
            gpuRoutedPhase2Nanos: 8,
            gpuRoutedCombineNanos: 9,
            routedExpertCombineNanos: 20,
            expertReadCount: 2,
            expertReadNanos: 21,
            expertReadMaxNanos: 12,
            layers: [
                QwenDecodeLayerDiagnostics(layer: 0,
                                           isFullAttention: true,
                                           elapsedNanos: 60,
                                           expertFetchNanos: 15,
                                           routePlanningNanos: 3,
                                           routedSetupNanos: 7,
                                           routedCommandBufferEncodingNanos: 8,
                                           routedCommandBufferCommitNanos: 9,
                                           routedCommandBufferWaitNanos: 10,
                                           expertReadCount: 2,
                                           expertReadNanos: 21,
                                           expertReadMaxNanos: 12,
                                           gpuStageTimingSampled: true,
                                           gpuMixerNanos: 9,
                                           gpuSharedExpertNanos: 6,
                                           gpuRouterNanos: 3,
                                           routedGPUStageTimingSampled: true,
                                           gpuRoutedPhase1Nanos: 6,
                                           gpuRoutedPhase2Nanos: 7,
                                           gpuRoutedCombineNanos: 8,
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
        #expect(aggregate.gpuStageTimingSampleCount == 4)
        #expect(aggregate.gpuMixerNanos == 30)
        #expect(aggregate.gpuSharedExpertNanos == 19)
        #expect(aggregate.gpuRouterNanos == 8)
        #expect(aggregate.routedSetupNanos == 17)
        #expect(aggregate.routedCommandBufferEncodingNanos == 19)
        #expect(aggregate.routedCommandBufferCommitNanos == 21)
        #expect(aggregate.routedCommandBufferWaitNanos == 23)
        #expect(aggregate.routedGPUStageTimingSampleCount == 4)
        #expect(aggregate.gpuRoutedPhase1Nanos == 11)
        #expect(aggregate.gpuRoutedPhase2Nanos == 13)
        #expect(aggregate.gpuRoutedCombineNanos == 15)
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
        #expect(aggregate.layers.map(\.routePlanningNanos) == [5, 0])
        #expect(aggregate.layers.map(\.routedSetupNanos) == [10, 0])
        #expect(aggregate.layers.map(\.routedCommandBufferEncodingNanos) == [12, 0])
        #expect(aggregate.layers.map(\.routedCommandBufferCommitNanos) == [14, 0])
        #expect(aggregate.layers.map(\.routedCommandBufferWaitNanos) == [16, 0])
        #expect(aggregate.layers.map(\.expertReadCount) == [3, 0])
        #expect(aggregate.layers.map(\.expertReadNanos) == [30, 0])
        #expect(aggregate.layers.map(\.expertReadMaxNanos) == [12, 0])
        #expect(aggregate.layers.map(\.gpuStageTimingSampleCount) == [2, 0])
        #expect(aggregate.layers.map(\.gpuMixerNanos) == [16, 0])
        #expect(aggregate.layers.map(\.gpuSharedExpertNanos) == [10, 0])
        #expect(aggregate.layers.map(\.gpuRouterNanos) == [5, 0])
        #expect(aggregate.layers.map(\.routedGPUStageTimingSampleCount) == [2, 0])
        #expect(aggregate.layers.map(\.gpuRoutedPhase1Nanos) == [9, 0])
        #expect(aggregate.layers.map(\.gpuRoutedPhase2Nanos) == [11, 0])
        #expect(aggregate.layers.map(\.gpuRoutedCombineNanos) == [13, 0])
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
            routedSetupNanos: .max,
            routedCommandBufferEncodingNanos: .max,
            routedCommandBufferCommitNanos: .max,
            routedCommandBufferWaitNanos: .max,
            routedGPUStageTimingSampleCount: .max,
            gpuRoutedPhase1Nanos: .max,
            gpuRoutedPhase2Nanos: .max,
            gpuRoutedCombineNanos: .max,
            routedExpertCombineNanos: .max,
            expertReadCount: .max,
            expertReadNanos: .max,
            expertReadMaxNanos: .max,
            layers: [QwenDecodeLayerDiagnostics(layer: 0,
                                                 isFullAttention: true,
                                                 elapsedNanos: .max,
                                                 expertFetchNanos: .max,
                                                 routePlanningNanos: .max,
                                                 routedSetupNanos: .max,
                                                 routedCommandBufferEncodingNanos: .max,
                                                 routedCommandBufferCommitNanos: .max,
                                                 routedCommandBufferWaitNanos: .max,
                                                 expertReadCount: .max,
                                                 expertReadNanos: .max,
                                                 expertReadMaxNanos: .max,
                                                 routedGPUStageTimingSampled: true,
                                                 gpuRoutedPhase1Nanos: .max,
                                                 gpuRoutedPhase2Nanos: .max,
                                                 gpuRoutedCombineNanos: .max)]))
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
            routedSetupNanos: 1,
            routedCommandBufferEncodingNanos: 1,
            routedCommandBufferCommitNanos: 1,
            routedCommandBufferWaitNanos: 1,
            routedGPUStageTimingSampleCount: 1,
            gpuRoutedPhase1Nanos: 1,
            gpuRoutedPhase2Nanos: 1,
            gpuRoutedCombineNanos: 1,
            routedExpertCombineNanos: 1,
            expertReadCount: 1,
            expertReadNanos: 1,
            expertReadMaxNanos: 1,
            layers: [QwenDecodeLayerDiagnostics(layer: 0,
                                                 isFullAttention: true,
                                                 elapsedNanos: 1,
                                                 expertFetchNanos: 1,
                                                 routePlanningNanos: 1,
                                                 routedSetupNanos: 1,
                                                 routedCommandBufferEncodingNanos: 1,
                                                 routedCommandBufferCommitNanos: 1,
                                                 routedCommandBufferWaitNanos: 1,
                                                 expertReadCount: 1,
                                                 expertReadNanos: 1,
                                                 expertReadMaxNanos: 1,
                                                 routedGPUStageTimingSampled: true,
                                                 gpuRoutedPhase1Nanos: 1,
                                                 gpuRoutedPhase2Nanos: 1,
                                                 gpuRoutedCombineNanos: 1)]))

        let aggregate = accumulator.makeDiagnostics(decodeLoopWallNanos: .max)
        #expect(aggregate.forwardWallNanos == .max)
        #expect(aggregate.commandBufferSubmissionCount == .max)
        #expect(aggregate.routerEvaluationCount == .max)
        #expect(aggregate.routedExpertEstimatedBytes == .max)
        #expect(aggregate.routedSetupNanos == .max)
        #expect(aggregate.routedCommandBufferEncodingNanos == .max)
        #expect(aggregate.routedCommandBufferCommitNanos == .max)
        #expect(aggregate.routedCommandBufferWaitNanos == .max)
        #expect(aggregate.routedGPUStageTimingSampleCount == .max)
        #expect(aggregate.gpuRoutedPhase1Nanos == .max)
        #expect(aggregate.gpuRoutedPhase2Nanos == .max)
        #expect(aggregate.gpuRoutedCombineNanos == .max)
        #expect(aggregate.expertReadCount == .max)
        #expect(aggregate.expertReadNanos == .max)
        #expect(aggregate.expertReadMaxNanos == .max)
        #expect(aggregate.layers.first?.elapsedNanos == .max)
        #expect(aggregate.layers.first?.expertFetchNanos == .max)
        #expect(aggregate.layers.first?.routePlanningNanos == .max)
        #expect(aggregate.layers.first?.routedSetupNanos == .max)
        #expect(aggregate.layers.first?.routedCommandBufferEncodingNanos == .max)
        #expect(aggregate.layers.first?.routedCommandBufferCommitNanos == .max)
        #expect(aggregate.layers.first?.routedCommandBufferWaitNanos == .max)
        #expect(aggregate.layers.first?.routedGPUStageTimingSampleCount == 2)
        #expect(aggregate.layers.first?.gpuRoutedPhase1Nanos == .max)
        #expect(aggregate.layers.first?.gpuRoutedPhase2Nanos == .max)
        #expect(aggregate.layers.first?.gpuRoutedCombineNanos == .max)
        #expect(aggregate.layers.first?.expertReadCount == .max)
        #expect(aggregate.layers.first?.expertReadNanos == .max)
        #expect(aggregate.layers.first?.expertReadMaxNanos == .max)
    }
}