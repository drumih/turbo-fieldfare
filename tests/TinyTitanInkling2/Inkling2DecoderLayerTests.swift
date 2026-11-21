import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2DecoderLayerTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realLayerZeroBlockRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let attention = try Inkling2SlidingAttention(
            tensors: tensors, layer: 0, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 1024,
            hidden: 2304, kernelSize: 4)
        let layer = try Inkling2DecoderLayer(
            tensors: tensors, store: store, layer: 0, hidden: 2304,
            kernelSize: 4, isBlockBoundary: true, attention: attention)
        var blocks: [[Float]] = []
        let output = try layer.forward(tensors.embeddingRow(0), blocks: &blocks)
        #expect(output.count == 2304)
        #expect(output.allSatisfy { $0.isFinite })
        #expect(blocks.count == 1)
    }
}
