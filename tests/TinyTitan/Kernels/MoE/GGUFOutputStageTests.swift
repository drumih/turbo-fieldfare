import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The model's tail on the real checkpoint: the output hyper-connection
/// collapses 4 streams to 2560, and the Q6_K head projects it to logits.
@Suite struct GGUFOutputStageTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static func bf16(_ value: Float) -> Float {
        Float(bitPattern: UInt32(Quantization.bf16Bits(value)) << 16)
    }

    private static func decoded(_ model: GGUFModel, name: String) throws -> [Float] {
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
        return try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](data))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func outputTailMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let arena = try GGUFResidentArena.build(
            names: ["output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight"],
            from: model, context: context)
        let stage = try GGUFOutputStage(
            context: context, arch: arch, model: model, arena: arena)

        let dim = arch.hiddenSize
        let streams = arch.hyperConnections.count
        let lowRank = arch.hyperConnections.lowRank
        var rng = SeedTree(0x0075).key("output-streams")
        let residual = (0..<(streams * dim)).map { _ in Float16(rng.uniform(-1, 1)) }
        let residualBits = residual.map(\.bitPattern)
        let residualBuffer: MTLBuffer? = residualBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let residualBuffer, let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.bufferAllocationFailed("output tail buffers")
        }
        try stage.encode(commandBuffer: cb, streamsBuffer: residualBuffer)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        // CPU reference: the same read path, then 512 head rows.
        let normW = try Self.decoded(model, name: "output_hc_norm.weight").map(Self.bf16)
        let downW = try Self.decoded(model, name: "output_hc_down.weight").map(Self.bf16)
        let upW = try Self.decoded(model, name: "output_hc_up.weight").map(Self.bf16)
        let x = residual.map(Float.init)
        let wide = streams * dim
        let eps: Float = 1e-6
        var normed = [Float](repeating: 0, count: wide)
        for s in 0..<streams {
            var square: Float = 0
            for d in 0..<dim { square += x[s * dim + d] * x[s * dim + d] }
            let invRms = 1 / (square / Float(dim) + eps).squareRoot()
            for d in 0..<dim {
                normed[s * dim + d] = Self.bf16(x[s * dim + d] * invRms * normW[s * dim + d])
            }
        }
        var low = [Float](repeating: 0, count: lowRank)
        for r in 0..<lowRank {
            var acc: Float = 0
            let base = r * wide
            for n in 0..<wide { acc += downW[base + n] * normed[n] }
            let scaled = acc / Float(streams)
            low[r] = Self.bf16(scaled / (1 + exp(-scaled)))
        }
        var mix = [Float](repeating: 0, count: wide)
        for m in 0..<wide {
            var acc: Float = 0
            let base = m * lowRank
            for n in 0..<lowRank { acc += upW[base + n] * low[n] }
            mix[m] = Self.bf16(acc)
        }
        var mixed = [Float](repeating: 0, count: dim)
        for d in 0..<dim {
            var acc: Float = 0
            for s in 0..<streams {
                let g = Self.bf16(Float(Float16(1 / (1 + exp(-mix[s * dim + d])))))
                acc += g * normed[s * dim + d]
            }
            mixed[d] = Self.bf16(acc / Float(streams))
        }
        let headRows = 512
        let headBytes = headRows * (2560 / 256) * 210
        let headLocation = try #require(model.tensor(named: "output.weight"))
        let headShard = model.shards[headLocation.shardIndex]
        let handle = try FileHandle(forReadingFrom: headShard.url)
        defer { try? handle.close() }
        try handle.seek(
            toOffset: headShard.dataOffset + headLocation.tensor.offset)
        let headData = try #require(try handle.read(upToCount: headBytes))
        let headValues = try GGUFDequant.decode(type: .q6K, bytes: [UInt8](headData))
        let pointer = stage.logits.contents().assumingMemoryBound(to: UInt16.self)
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for row in 0..<headRows {
            var acc: Float = 0
            let base = row * dim
            for n in 0..<dim { acc += headValues[base + n] * mixed[n] }
            let reference = Float(Float16(acc))
            let got = Float(Float16(bitPattern: pointer[row]))
            let diff = Double(got - reference)
            sqDiff += diff * diff
            sqRef += Double(reference * reference)
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("output tail RMS relative error over \(headRows) logits: \(relative)")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }
}
