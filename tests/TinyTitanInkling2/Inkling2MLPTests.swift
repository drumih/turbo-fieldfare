import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2MLPTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realLayerZeroMLPRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let mlp = try Inkling2MLP(
            tensors: tensors, store: store, layer: 0,
            gpu: try Inkling2GPUExpert())
        let output = try mlp.forward(tensors.embeddingRow(0))
        #expect(output.count == 2304)
        #expect(output.allSatisfy { $0.isFinite })
    }
}
