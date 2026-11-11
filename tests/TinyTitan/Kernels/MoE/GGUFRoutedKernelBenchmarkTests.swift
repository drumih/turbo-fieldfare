import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// GPU time of the batched routed stage on real experts, per layer, against
/// the bytes it reads: the number that says whether the expert kernels run at
/// memory bandwidth. Opt-in (`TINYTITAN_KERNEL_BENCH=1` plus a model), since
/// it is a measurement rather than a check; it still asserts the stage ran.
@Suite struct GGUFRoutedKernelBenchmarkTests {
    private static var enabled: Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment["TINYTITAN_GGUF_MODEL"] != nil
            && environment["TINYTITAN_KERNEL_BENCH"] == "1"
    }

    @Test(.enabled(if: enabled))
    func routedStageGPUTimePerLayer() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"])
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let cache = try GGUFExpertBufferCache(
            model: model, device: context.device, capacity: arch.topKExperts)
        let stage = try GGUFMoERoutedStage(context: context, arch: arch)
        var rng = SeedTree(0xBE7C).key("routed-bench")
        let bits = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)).bitPattern }
        let x: MTLBuffer? = bits.withUnsafeBytes { raw in
            raw.baseAddress.flatMap {
                context.device.makeBuffer(bytes: $0, length: raw.count, options: .storageModeShared)
            }
        }
        let out = context.device.makeBuffer(
            length: arch.hiddenSize * 2, options: .storageModeShared)
        let xBuffer = try #require(x)
        let outBuffer = try #require(out)
        let experts = Array(stride(from: 5, to: 5 + 41 * arch.topKExperts, by: 41))
        let weights = [Float](repeating: 0.1, count: arch.topKExperts)
        var totalSeconds = 0.0
        var totalBytes = 0
        for layer in [0, 1, 2, 3, 12, 24, 47] {
            let fetched = try cache.experts(layer: layer, experts: experts)
            let bytes = fetched.reduce(0) { $0 + $1.fetchedBytes }
            let repeats = 20
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            for _ in 0..<repeats {
                try stage.encode(
                    commandBuffer: commandBuffer, experts: fetched, x: xBuffer,
                    weights: weights, out: outBuffer)
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            let seconds = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / Double(repeats)
            totalSeconds += seconds
            totalBytes += bytes
            let types = "\(fetched[0].gate.type.name)/\(fetched[0].down.type.name)"
            print(
                String(
                    format: "layer %2d %@: %.3f ms, %.1f MB, %.1f GB/s", layer, types,
                    seconds * 1000, Double(bytes) / 1e6, Double(bytes) / seconds / 1e9))
        }
        print(
            String(
                format: "routed stage: %.1f GB/s overall; 48 layers would take %.1f ms",
                Double(totalBytes) / totalSeconds / 1e9, totalSeconds / 7 * 48 * 1000))
        #expect(totalSeconds > 0)
    }
}
