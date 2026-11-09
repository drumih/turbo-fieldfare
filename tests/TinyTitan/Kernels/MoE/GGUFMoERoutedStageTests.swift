import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The routed-MoE stage on real GGUF weights: two experts of layer 0 through
/// fused GEMVs, SiLU and weighted accumulation, against a CPU reference built
/// from the same decoders.
@Suite struct GGUFMoERoutedStageTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func routedStageMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let provider = GGUFMappedExperts(
            model: model, mapped: try GGUFMappedShards(model: model, device: context.device))
        let stage = try GGUFMoERoutedStage(context: context, arch: arch)

        var rng = SeedTree(0x5EED).key("gguf-moe-x")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let experts = [0, 1]
        let weights: [Float] = [0.6, 0.4]

        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer,
            let out = context.device.makeBuffer(
                length: arch.hiddenSize * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            throw MetalError.bufferAllocationFailed("routed MoE test buffers")
        }
        memset(out.contents(), 0, arch.hiddenSize * MemoryLayout<UInt16>.size)
        try stage.encode(
            commandBuffer: commandBuffer, provider: provider, layer: 0,
            x: xBuffer, experts: experts, weights: weights, out: out)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }

        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        let gpu = (0..<arch.hiddenSize).map { Float(Float16(bitPattern: pointer[$0])) }

        // CPU reference from the same decoders.
        var reference = [Float](repeating: 0, count: arch.hiddenSize)
        let xf = x.map(Float.init)
        for (index, expert) in experts.enumerated() {
            let w = try GGUFExpertLayout.weights(layer: 0, expert: expert, in: model)
            func decoded(_ range: GGUFTensorRange) throws -> [Float] {
                let shard = model.shards[range.shardIndex]
                let handle = try FileHandle(forReadingFrom: shard.url)
                defer { try? handle.close() }
                try handle.seek(toOffset: range.fileOffset)
                let data = try #require(try handle.read(upToCount: Int(range.byteCount)))
                return try GGUFDequant.decode(type: range.type, bytes: [UInt8](data))
            }
            let gate = try decoded(w.gate)
            let up = try decoded(w.up)
            let down = try decoded(w.down)
            var act = [Float](repeating: 0, count: arch.moeIntermediateSize)
            for row in 0..<arch.moeIntermediateSize {
                var g: Float = 0
                var u: Float = 0
                let base = row * arch.hiddenSize
                for n in 0..<arch.hiddenSize {
                    g += gate[base + n] * xf[n]
                    u += up[base + n] * xf[n]
                }
                let gh = Float(Float16(g))
                let uh = Float(Float16(u))
                act[row] = Float(Float16(gh / (1 + exp(-gh)) * uh))
            }
            for row in 0..<arch.hiddenSize {
                var acc: Float = 0
                let base = row * arch.moeIntermediateSize
                for n in 0..<arch.moeIntermediateSize {
                    acc += down[base + n] * act[n]
                }
                reference[row] += weights[index] * Float(Float16(acc))
            }
        }

        var sqDiff: Double = 0
        var sqRef: Double = 0
        for i in 0..<arch.hiddenSize {
            let diff = Double(gpu[i] - reference[i])
            sqDiff += diff * diff
            sqRef += Double(reference[i] * reference[i])
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("routed MoE stage RMS relative error: \(relative)")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }
}
