import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2MLPTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func warmMLPMicrobenchmark() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let x = try tensors.embeddingRow(0)
        let mlp = try Inkling2MLP(
            tensors: tensors, store: store, layer: 0,
            gpu: try Inkling2GPUExpert())
        _ = try mlp.forward(x)
        let iterations = 20
        let start = Date()
        for _ in 0..<iterations { _ = try mlp.forward(x) }
        let seconds = Date().timeIntervalSince(start)
        print("warm MLP: \(seconds / Double(iterations) * 1000) ms/op")
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func realLayerZeroMLPRuns() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let tensors = try Inkling2DenseTensors(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let x = try tensors.embeddingRow(0)
        let cpu = try Inkling2MLP(tensors: tensors, store: store, layer: 0)
        let gpu = try Inkling2MLP(
            tensors: tensors, store: store, layer: 0,
            gpu: try Inkling2GPUExpert())
        let cpuOutput = try cpu.forward(x)
        let output = try gpu.forward(x)
        #expect(output.count == 2304)
        #expect(output.allSatisfy { $0.isFinite })
        var maxDifference: Float = 0
        for index in 0..<output.count {
            maxDifference = max(maxDifference, abs(cpuOutput[index] - output[index]))
        }
        print("mlp max difference=\(maxDifference)")
    }
}
