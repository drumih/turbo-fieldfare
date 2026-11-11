import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The batched routed-MoE route (one dispatch per projection for all selected
/// experts) against the one-expert-at-a-time route it replaces, on real GGUF
/// weights: the batched kernels call the same row functions and the weighted
/// sum adds in the same order with the same rounding, so the outputs must be
/// bit-identical, not merely close.
@Suite struct GGUFMoEBatchedRouteTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func batchedRouteIsBitIdenticalToTheOneAtATimeRoute() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"])
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let cache = try GGUFExpertBufferCache(
            model: model, device: context.device, capacity: arch.topKExperts)
        let stage = try GGUFMoERoutedStage(context: context, arch: arch)

        var rng = SeedTree(0xBA7C4).key("gguf-moe-batched")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let experts = Array(stride(from: 3, to: 3 + 37 * arch.topKExperts, by: 37))
        let raw = (0..<arch.topKExperts).map { _ in Float(rng.uniform(0.1, 1)) }
        let total = raw.reduce(0, +)
        let weights = raw.map { $0 / total }

        let xBuffer = try Self.buffer(x.map(\.bitPattern), device: context.device)
        // Several layers, so every projection format the checkpoint uses is hit.
        for layer in [0, 1, 12, 47] {
            let fetched = try cache.experts(layer: layer, experts: experts)
            let batched = try Self.run(context: context, hidden: arch.hiddenSize) { cb, out in
                try stage.encode(
                    commandBuffer: cb, experts: fetched, x: xBuffer, weights: weights, out: out)
            }
            let each = try Self.run(context: context, hidden: arch.hiddenSize) { cb, out in
                try stage.encodeEach(
                    commandBuffer: cb, experts: fetched, x: xBuffer, weights: weights,
                    out: out, outOffset: 0)
            }
            #expect(batched.contains { $0 != 0 }, "layer \(layer): routed output is all zero")
            #expect(batched == each, "layer \(layer): batched route differs from one-at-a-time")
        }
    }

    private static func run(
        context: MetalContext,
        hidden: Int,
        _ body: (MTLCommandBuffer, MTLBuffer) throws -> Void
    ) throws -> [UInt16] {
        let out = try buffer([UInt16](repeating: 0, count: hidden), device: context.device)
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try body(commandBuffer, out)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }
        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: pointer, count: hidden))
    }

    private static func buffer(_ values: [UInt16], device: MTLDevice) throws -> MTLBuffer {
        let made: MTLBuffer? = values.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let made else { throw MetalError.bufferAllocationFailed("batched route test") }
        return made
    }
}
