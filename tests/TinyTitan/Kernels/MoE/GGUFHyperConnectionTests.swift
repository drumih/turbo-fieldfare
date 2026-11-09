import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The engine's own hyper-connection kernels fed from the GGUF arena: the
/// first stage where an engine kernel reads bf16 views of checkpoint tensors.
@Suite struct GGUFHyperConnectionTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
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

    /// bf16 rounding, which is what the arena stores.
    private static func bf16(_ value: Float) -> Float {
        // bf16 bits are the top 16 bits of the fp32 pattern, not an fp16 value.
        Float(bitPattern: UInt32(Quantization.bf16Bits(value)) << 16)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func hyperConnectionMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let dim = 2_560
        let streams = 4
        let lowRank = 320
        let names = [
            "blk.0.hc_attn_norm.weight",
            "blk.0.hc_attn_down.weight",
            "blk.0.hc_attn_up.weight",
            "blk.0.hc_attn_inject.weight",
        ]
        let arena = try GGUFResidentArena.build(
            names: names, from: model, context: context)

        var rng = SeedTree(0x4C4C).key("hc-streams")
        var streamValues = [Float16]()
        for _ in 0..<(streams * dim) { streamValues.append(Float16(rng.uniform(-1, 1))) }
        let streamBits = streamValues.map(\.bitPattern)
        var blockOut = [Float16]()
        for _ in 0..<dim { blockOut.append(Float16(rng.uniform(-1, 1))) }

        func makeBuffer<T>(_ values: [T]) -> MTLBuffer? {
            values.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return nil }
                return context.device.makeBuffer(
                    bytes: base, length: raw.count, options: .storageModeShared)
            }
        }
        guard
            let streamsBuffer = makeBuffer(streamBits),
            let blockOutBuffer = makeBuffer(blockOut.map(\.bitPattern)),
            let blockIn = context.device.makeBuffer(length: dim * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("hc buffers") }

        func weights(_ name: String) throws -> HyperConnection.Weights {
            let entry = try #require(arena.entries[name])
            return HyperConnection.Weights(
                weights: arena.buffer, weightsOffset: Int(entry.offset),
                scales: arena.buffer, biases: arena.buffer, isBF16: true)
        }
        let hc = try HyperConnection(
            context: context, dim: dim, streams: streams, lowRank: lowRank, weightBits: 4)
        let normEntry = try #require(arena.entries["blk.0.hc_attn_norm.weight"])
        guard let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try hc.encodeRead(
            commandBuffer: cb, streamsBuffer: streamsBuffer,
            hcNorm: arena.buffer, hcNormOffset: Int(normEntry.offset),
            down: try weights("blk.0.hc_attn_down.weight"),
            up: try weights("blk.0.hc_attn_up.weight"),
            blockInput: blockIn, eps: 1e-6)
        try hc.encodeWrite(
            commandBuffer: cb, streamsBuffer: streamsBuffer,
            inject: try weights("blk.0.hc_attn_inject.weight"),
            blockOut: blockOutBuffer)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        // CPU reference, mirroring the shaders' rounding points.
        let normW = try Self.decoded(model, name: "blk.0.hc_attn_norm.weight").map(Self.bf16)
        let downW = try Self.decoded(model, name: "blk.0.hc_attn_down.weight").map(Self.bf16)
        let upW = try Self.decoded(model, name: "blk.0.hc_attn_up.weight").map(Self.bf16)
        let injectW = try Self.decoded(model, name: "blk.0.hc_attn_inject.weight").map(Self.bf16)
        let eps: Float = 1e-6
        let wide = streams * dim
        let x = streamValues.map(Float.init)

        var normed = [Float](repeating: 0, count: wide)
        for s in 0..<streams {
            var square: Float = 0
            for d in 0..<dim { square += x[s * dim + d] * x[s * dim + d] }
            let mean = square / Float(dim) + eps
            let scale = 1 / mean.squareRoot()
            for d in 0..<dim {
                normed[s * dim + d] = Self.bf16(x[s * dim + d] * scale * normW[s * dim + d])
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
        var referenceIn = [Float](repeating: 0, count: dim)
        for d in 0..<dim {
            var acc: Float = 0
            for s in 0..<streams {
                let g = Self.bf16(Float(Float16(1 / (1 + exp(-mix[s * dim + d])))))
                acc += g * normed[s * dim + d]
            }
            referenceIn[d] = Self.bf16(acc / Float(streams))
        }
        var inject = [Float](repeating: 0, count: streams)
        for s in 0..<streams {
            var acc: Float = 0
            let base = s * wide
            for d in 0..<wide { acc += injectW[base + d] * normed[d] }
            inject[s] = acc
        }
        var referenceStreams = x
        for s in 0..<streams {
            let g = Self.bf16(Float(Float16(2 / (1 + exp(-inject[s] / Float(streams))))))
            for d in 0..<dim {
                referenceStreams[s * dim + d] = Self.bf16(
                    x[s * dim + d] + Float(blockOut[d]) * g)
            }
        }

        let inPointer = blockIn.contents().assumingMemoryBound(to: UInt16.self)
        let streamPointer = streamsBuffer.contents().assumingMemoryBound(to: UInt16.self)
        var sqIn: Double = 0
        var refIn: Double = 0
        for d in 0..<dim {
            let got = Double(Float(Float16(bitPattern: inPointer[d])))
            let diff = got - Double(referenceIn[d])
            sqIn += diff * diff
            refIn += Double(referenceIn[d] * referenceIn[d])
        }
        var sqS: Double = 0
        var refS: Double = 0
        for i in 0..<wide {
            let got = Double(Float(Float16(bitPattern: streamPointer[i])))
            let diff = got - Double(referenceStreams[i])
            sqS += diff * diff
            refS += Double(referenceStreams[i] * referenceStreams[i])
        }
        let inRelative = (sqIn / max(refIn, 1e-12)).squareRoot()
        let streamRelative = (sqS / max(refS, 1e-12)).squareRoot()
        print("hyper-connection block input RMS: \(inRelative), streams RMS: \(streamRelative)")
        #expect(inRelative < 5e-3)
        #expect(streamRelative < 5e-3)
    }
}
