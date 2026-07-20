import Testing
import Metal
@testable import TurboFieldfare

@Suite struct PrefillChunkScratchTests {
    @Test func gemma4T32LayoutMatchesTask7ScratchContract() {
        let layout = PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 32)

        #expect(layout.chunkTokens == 32)
        #expect(layout.hiddenElements == 32 * 2816)
        #expect(layout.normedElements == 32 * 2816)
        #expect(layout.qElements == 32 * 8192)
        #expect(layout.kStageElements == 32 * 2048)
        #expect(layout.vStageElements == 32 * 2048)
        #expect(layout.attentionOutputElements == 32 * 8192)
        #expect(layout.denseXElements == 32 * 2816)
        #expect(layout.routedXElements == 32 * 2816)
        #expect(layout.routerXElements == 32 * 2816)
        #expect(layout.h1Elements == 32 * 2816)
        #expect(layout.h2Elements == 32 * 2816)
        #expect(layout.routePartialElements == 32 * 8 * 2816)
        #expect(layout.routeIDElements == 32 * 8)
        #expect(layout.routeWeightElements == 32 * 8)
        #expect(layout.sharedExpertScratchElements == 2112)
        #expect(layout.routedPairMicrobatchRows == 32)
        #expect(layout.routedGateUpActElements == 3 * 32 * 704)
        #expect(layout.routedDownOutputElements == 32 * 2816)

        let worksheetT32UpperBound = Int(4.5 * 1_048_576.0)
        #expect(layout.totalPersistentBytes <= worksheetT32UpperBound)
    }

    @Test func layoutClampsChunkSizeToRuntimeBounds() {
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 0).chunkTokens == 1)
        #expect(PrefillChunkScratchLayout(config: .gemma4_26B_A4B, chunkTokens: 512).chunkTokens == 128)
    }

    @Test func allocationUsesPrivateScratchAndSharedRouteMetadata() throws {
        let ctx = try MetalContext()
        let toy = ArchConfig(hiddenSize: 64,
                             intermediateSize: 48,
                             moeIntermediateSize: 16,
                             numHeads: 4,
                             numKVHeads: 2,
                             numFullKVHeads: 1,
                             headDim: 16,
                             fullHeadDim: 32,
                             vocabSize: 128,
                             slidingWindow: 16,
                             finalLogitSoftcap: 30.0,
                             ropeTheta: 10_000,
                             fullRopeTheta: 1_000_000,
                             partialRotaryFactor: 0.25,
                             numLayers: 2,
                             numExperts: 8,
                             topKExperts: 2,
                             tieWordEmbeddings: true,
                             attentionKEqV: true,
                             fullAttentionLayerMask: [0, 1],
                             hiddenActivation: "gelu_pytorch_tanh")
        let layout = PrefillChunkScratchLayout(config: toy, chunkTokens: 4)

        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)

        #expect(scratch.layout == layout)
        #expect(scratch.hidden.length == layout.hiddenElements * MemoryLayout<Float16>.stride)
        #expect(scratch.denseX.length == layout.denseXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedX.length == layout.routedXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routerX.length == layout.routerXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routePartials.length == layout.routePartialElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routeIDs.length == layout.routeIDElements * MemoryLayout<UInt32>.stride)
        #expect(scratch.routeWeights.length == layout.routeWeightElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedGateUpActScratch.length == layout.routedGateUpActElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedDownScratch.length == layout.routedDownOutputElements * MemoryLayout<Float16>.stride)
        #expect(scratch.hidden.storageMode == MTLStorageMode.private)
        #expect(scratch.denseX.storageMode == MTLStorageMode.private)
        #expect(scratch.routedX.storageMode == MTLStorageMode.private)
        #expect(scratch.routerX.storageMode == MTLStorageMode.private)
        #expect(scratch.routedGateUpActScratch.storageMode == MTLStorageMode.private)
        #expect(scratch.routedDownScratch.storageMode == MTLStorageMode.private)
        #expect(scratch.routeIDs.storageMode == MTLStorageMode.shared)
        #expect(scratch.routeWeights.storageMode == MTLStorageMode.shared)
    }
}
