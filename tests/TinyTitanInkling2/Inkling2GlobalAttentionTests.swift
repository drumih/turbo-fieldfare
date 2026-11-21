import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2GlobalAttentionTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realGlobalLayerFiveRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let tensors = try Inkling2DenseTensors(directory: URL(fileURLWithPath: path))
        let attention = try Inkling2GlobalAttention(
            tensors: tensors, layer: 5, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 0,
            hidden: 2304, kernelSize: 4)
        let x = try tensors.embeddingRow(0)
        let cpu = try Inkling2GlobalAttention(
            tensors: tensors, layer: 5, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 0,
            hidden: 2304, kernelSize: 4)
        let gpuAttention = try Inkling2GlobalAttention(
            tensors: tensors, layer: 5, heads: 24, kvHeads: 8, headDim: 128,
            relativeDim: 16, extent: 1024, sliding: 0,
            hidden: 2304, kernelSize: 4, gpu: try Inkling2GPUExpert())
        let result = try attention.forward(x, tokenCount: 1)
        let cpuResult = try cpu.forward(x, tokenCount: 1)
        let gpuResult = try gpuAttention.forward(x, tokenCount: 1)
        #expect(result.values.count == 2304)
        #expect(result.values.allSatisfy { $0.isFinite })
        var maxDifference: Float = 0
        for index in 0..<2304 {
            maxDifference = max(
                maxDifference, abs(cpuResult.values[index] - gpuResult.values[index]))
        }
        print("global attention max difference=\(maxDifference)")
        #expect(result.state.k.count == 1024 * 3)
        #expect(result.state.v.count == 1024 * 3)
    }
}
