import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// One layer's whole MoE on the real checkpoint: router -> top-10 -> routed
/// experts -> shared expert, against a CPU reference.
@Suite struct GGUFMoELayerTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func layerMoEMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let router = try GGUFRouter(context: context, arch: arch, model: model)
        let provider = GGUFMappedExperts(
            model: model, mapped: try GGUFMappedShards(model: model, device: context.device))
        let routed = try GGUFMoERoutedStage(context: context, arch: arch)
        let shared = try GGUFSharedExpertStage(context: context, arch: arch, model: model)

        var rng = SeedTree(0xF00D).key("layer-moe-x")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
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
            let routeCB = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("layer MoE buffers") }
        try router.encode(commandBuffer: routeCB, layer: 0, x: xBuffer)
        routeCB.commit()
        routeCB.waitUntilCompleted()
        if let error = routeCB.error { throw MetalError.invalidState("\(error)") }
        let selection = router.select(topK: arch.topKExperts)

        memset(out.contents(), 0, arch.hiddenSize * MemoryLayout<UInt16>.size)
        guard let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try routed.encode(
            commandBuffer: cb, provider: provider, layer: 0, x: xBuffer,
            experts: selection.experts, weights: selection.weights, out: out)
        try shared.encode(commandBuffer: cb, layer: 0, x: xBuffer, out: out)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }
        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        let gpu = (0..<arch.hiddenSize).map { Float(Float16(bitPattern: pointer[$0])) }

        // CPU reference for the same selection.
        func tensor(_ name: String) throws -> [Float] {
            let location = try #require(model.tensor(named: name))
            let shard = model.shards[location.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            defer { try? handle.close() }
            try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
            let data = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
            if location.tensor.type == .f32 {
                return data.withUnsafeBytes { raw in
                    let count = raw.count / 4
                    let p = raw.bindMemory(to: Float.self)
                    return (0..<count).map { p[$0] }
                }
            }
            return try GGUFDequant.decode(
                type: location.tensor.type, bytes: [UInt8](data))
        }
        let xf = x.map(Float.init)
        func decoded(_ range: GGUFTensorRange) throws -> [Float] {
            let shard = model.shards[range.shardIndex]
            let handle = try FileHandle(forReadingFrom: shard.url)
            defer { try? handle.close() }
            try handle.seek(toOffset: range.fileOffset)
            let data = try #require(try handle.read(upToCount: Int(range.byteCount)))
            return try GGUFDequant.decode(type: range.type, bytes: [UInt8](data))
        }
        var reference = [Float](repeating: 0, count: arch.hiddenSize)
        for (index, expert) in selection.experts.enumerated() {
            let weights = try GGUFExpertLayout.weights(layer: 0, expert: expert, in: model)
            let gate = try decoded(weights.gate)
            let up = try decoded(weights.up)
            let down = try decoded(weights.down)
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
                for n in 0..<arch.moeIntermediateSize { acc += down[base + n] * act[n] }
                reference[row] += selection.weights[index] * Float(Float16(acc))
            }
        }
        // Shared expert.
        let sGate = try tensor("blk.0.ffn_gate_shexp.weight")
        let sUp = try tensor("blk.0.ffn_up_shexp.weight")
        let sDown = try tensor("blk.0.ffn_down_shexp.weight")
        let sScalar = try tensor("blk.0.ffn_gate_inp_shexp.weight")
        var sAct = [Float](repeating: 0, count: arch.moeIntermediateSize)
        for row in 0..<arch.moeIntermediateSize {
            var g: Float = 0
            var u: Float = 0
            let base = row * arch.hiddenSize
            for n in 0..<arch.hiddenSize {
                g += sGate[base + n] * xf[n]
                u += sUp[base + n] * xf[n]
            }
            let gh = Float(Float16(g))
            let uh = Float(Float16(u))
            sAct[row] = Float(Float16(gh / (1 + exp(-gh)) * uh))
        }
        var scalar: Float = 0
        for n in 0..<arch.hiddenSize { scalar += sScalar[n] * xf[n] }
        let scale = Float(Float16(1 / (1 + exp(-scalar))))
        for row in 0..<arch.hiddenSize {
            var acc: Float = 0
            let base = row * arch.moeIntermediateSize
            for n in 0..<arch.moeIntermediateSize { acc += sDown[base + n] * sAct[n] }
            reference[row] += scale * Float(Float16(acc))
        }

        var sqDiff: Double = 0
        var sqRef: Double = 0
        for i in 0..<arch.hiddenSize {
            let diff = Double(gpu[i] - reference[i])
            sqDiff += diff * diff
            sqRef += Double(reference[i] * reference[i])
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print(
            "layer MoE RMS relative error: \(relative); experts "
                + "\(selection.experts.prefix(3))...")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }
}
