import Metal
import Testing

@testable import TinyTitan
import TinyTitanCore

@Suite struct PrefillChunkScratchTests {
    @Test func qwen36T32LayoutMatchesScratchContract() {
        let layout = PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 32)

        #expect(layout.chunkTokens == 32)
        #expect(layout.hiddenElements == 32 * 2048)
        #expect(layout.normedElements == 32 * 2048)
        // q_proj emits per-head [query ; gate] halves: 2 * 16 heads * 256.
        #expect(layout.qElements == 32 * 8192)
        #expect(layout.kStageElements == 32 * 512)
        #expect(layout.vStageElements == 32 * 512)
        #expect(layout.attentionOutputElements == 32 * 4096)
        #expect(layout.denseXElements == 32 * 2048)
        #expect(layout.routedXElements == 32 * 2048)
        #expect(layout.routerXElements == 32 * 2048)
        #expect(layout.h1Elements == 32 * 2048)
        #expect(layout.h2Elements == 32 * 2048)
        #expect(layout.routePartialElements == 32 * 8 * 2048)
        #expect(layout.routeIDElements == 32 * 8)
        #expect(layout.routeWeightElements == 32 * 8)
        // The batched shared-expert path writes a chunk of gate/up/act rows,
        // not one row: chunkTokens * sharedIntermediate (32 * 512 here).
        #expect(layout.sharedExpertScratchElements == 32 * 512)
        #expect(layout.routedPairMicrobatchRows == 32)
        #expect(layout.routedGateUpActElements == 3 * 32 * 512)
        #expect(layout.routedDownOutputElements == 32 * 2048)
        // Qwen 3.6 extensions: split q/gate halves, gated-DeltaNet bundle,
        // and the shared-expert scalar gate.
        #expect(layout.attnGateElementsPerToken == 4096)
        #expect(layout.gdnQKVDim == 8192)
        #expect(layout.gdnValueDim == 4096)
        #expect(layout.gdnVHeads == 32)
        #expect(layout.sharedScalarGateElements == 1)
        #expect(layout.gdnConvOutElements == 32 * 8192)
        #expect(layout.gdnZElements == 32 * 4096)
        #expect(layout.gdnAElements == 32 * 32)
        #expect(layout.sharedScalarGateBufferElements == 32)

        // 4.5 MB worksheet bound plus the batched shared expert's three
        // chunk rows (3 * 32 * 512 * 2 bytes = 96 KB at T32).
        let worksheetT32UpperBound = Int(4.6 * 1_048_576.0)
        #expect(layout.totalPersistentBytes <= worksheetT32UpperBound)
    }

    @Test func layoutClampsChunkSizeToRuntimeBounds() {
        #expect(PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 0).chunkTokens == 1)
        #expect(
            PrefillChunkScratchLayout(config: .qwen36_35B_A3B, chunkTokens: 8_192).chunkTokens
                == 4_096)
    }

    @Test func qwenLongChunkScratchRemainsBounded() {
        let layout = PrefillChunkScratchLayout(
            config: .qwen36_35B_A3B,
            chunkTokens: 4_096)
        #expect(layout.chunkTokens == 4_096)
        #expect(layout.totalPersistentBytes < 1_024 * 1_048_576)
    }

    @Test func allocationUsesPrivateScratchAndSharedRouteMetadata() throws {
        let ctx = try MetalContext()
        let toy = ArchConfig.qwenToy()
        let layout = PrefillChunkScratchLayout(config: toy, chunkTokens: 4)

        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)

        #expect(scratch.layout == layout)
        #expect(scratch.hidden.length == layout.hiddenElements * MemoryLayout<Float16>.stride)
        #expect(scratch.denseX.length == layout.denseXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routedX.length == layout.routedXElements * MemoryLayout<Float16>.stride)
        #expect(scratch.routerX.length == layout.routerXElements * MemoryLayout<Float16>.stride)
        #expect(
            scratch.routePartials.length == layout.routePartialElements
                * MemoryLayout<Float16>.stride)
        #expect(scratch.routeIDs.length == layout.routeIDElements * MemoryLayout<UInt32>.stride)
        #expect(
            scratch.routeWeights.length == layout.routeWeightElements * MemoryLayout<Float16>.stride
        )
        #expect(
            scratch.routedGateUpActScratch.length == layout.routedGateUpActElements
                * MemoryLayout<Float16>.stride)
        #expect(
            scratch.routedDownScratch.length == layout.routedDownOutputElements
                * MemoryLayout<Float16>.stride)
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
