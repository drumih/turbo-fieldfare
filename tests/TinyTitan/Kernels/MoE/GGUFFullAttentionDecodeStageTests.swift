import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// A full-attention layer's decode step over two positions, against a CPU
/// reference that walks the same pipeline: q split, q/k norms, partial NeoX
/// RoPE, GQA softmax attention, sigmoid gate, o projection.
@Suite struct GGUFFullAttentionDecodeStageTests {
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
    func firstTokenMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let layer = try #require(
            (0..<arch.numLayers).first { arch.fullAttentionLayerMask[$0] == 1 })
        let names = [
            "blk.\(layer).attn_q.weight", "blk.\(layer).attn_k.weight",
            "blk.\(layer).attn_v.weight", "blk.\(layer).attn_output.weight",
            "blk.\(layer).attn_q_norm.weight", "blk.\(layer).attn_k_norm.weight",
        ]
        let arena = try GGUFResidentArena.build(names: names, from: model, context: context)
        let stage = try GGUFFullAttentionDecodeStage(
            context: context, arch: arch, arena: arena, maxSeq: 8)
        var rng = SeedTree(0xA77E).key("full-attn-x")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer,
            let out = context.device.makeBuffer(
                length: arch.hiddenSize * 2, options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("full attention buffers") }
        try stage.encode(commandBuffer: cb, layer: layer, position: 0, x: xBuffer, out: out)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        // With one cached key, attention collapses to v (softmax of a single
        // score is 1), so the reference is o_proj(sigmoid(gate) * v).
        let hidden = arch.hiddenSize
        let headDim = arch.fullHeadDim
        let qDim = arch.numHeads * headDim
        let kvDim = arch.numFullKVHeads * headDim
        let xf = x.map(Float.init)
        func matvec(_ name: String, rows: Int) throws -> [Float] {
            let weights = try Self.decoded(model, name: name).map(Self.bf16)
            var result = [Float](repeating: 0, count: rows)
            for r in 0..<rows {
                var acc: Float = 0
                let base = r * hidden
                for n in 0..<hidden { acc += weights[base + n] * xf[n] }
                result[r] = Float(Float16(acc))
            }
            return result
        }
        let packed = try matvec("blk.\(layer).attn_q.weight", rows: 2 * qDim)
        // The projection packs [q ; gate] per head: 2*headDim per head.
        var gate = [Float](repeating: 0, count: qDim)
        for head in 0..<arch.numHeads {
            for i in 0..<headDim {
                gate[head * headDim + i] = packed[head * 2 * headDim + headDim + i]
            }
        }
        let v = try matvec("blk.\(layer).attn_v.weight", rows: kvDim)
        let group = arch.numHeads / arch.numFullKVHeads
        var attn = [Float](repeating: 0, count: qDim)
        for head in 0..<arch.numHeads {
            let kvHead = head / group
            for i in 0..<headDim {
                let gated = Float(Float16(v[kvHead * headDim + i]))
                let g = Float(Float16(1 / (1 + exp(-gate[head * headDim + i]))))
                attn[head * headDim + i] = Float(Float16(gated * g))
            }
        }
        let oW = try Self.decoded(model, name: "blk.\(layer).attn_output.weight").map(Self.bf16)
        var reference = [Float](repeating: 0, count: hidden)
        for r in 0..<hidden {
            var acc: Float = 0
            let base = r * qDim
            for n in 0..<qDim { acc += oW[base + n] * attn[n] }
            reference[r] = Float(Float16(acc))
        }
        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for i in 0..<hidden {
            let got = Double(Float(Float16(bitPattern: pointer[i])))
            let diff = got - Double(reference[i])
            sqDiff += diff * diff
            sqRef += Double(reference[i] * reference[i])
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("full attention token-0 RMS relative error: \(relative)")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func secondTokenMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let context = try MetalContext()
        let layer = try #require(
            (0..<arch.numLayers).first { arch.fullAttentionLayerMask[$0] == 1 },
            "no full-attention layer in the mask")
        let names = [
            "blk.\(layer).attn_q.weight", "blk.\(layer).attn_k.weight",
            "blk.\(layer).attn_v.weight", "blk.\(layer).attn_output.weight",
            "blk.\(layer).attn_q_norm.weight", "blk.\(layer).attn_k_norm.weight",
        ]
        let arena = try GGUFResidentArena.build(names: names, from: model, context: context)
        let stage = try GGUFFullAttentionDecodeStage(
            context: context, arch: arch, arena: arena, maxSeq: 8)

        var rng = SeedTree(0xA77E).key("full-attn-x")
        let x0 = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let x1 = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        func buffer(_ values: [Float16]) -> MTLBuffer? {
            let bits = values.map(\.bitPattern)
            return bits.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return nil }
                return context.device.makeBuffer(
                    bytes: base, length: raw.count, options: .storageModeShared)
            }
        }
        guard let x0Buffer = buffer(x0), let x1Buffer = buffer(x1),
            let out = context.device.makeBuffer(
                length: arch.hiddenSize * 2, options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("full attention buffers") }
        try stage.encode(
            commandBuffer: cb, layer: layer, position: 0, x: x0Buffer, out: out)
        cb.commit()
        cb.waitUntilCompleted()
        guard let cb2 = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try stage.encode(
            commandBuffer: cb2, layer: layer, position: 1, x: x1Buffer, out: out)
        cb2.commit()
        cb2.waitUntilCompleted()
        if let error = cb2.error { throw MetalError.invalidState("\(error)") }

        // CPU reference for both positions.
        let hidden = arch.hiddenSize
        let headDim = arch.fullHeadDim
        let qDim = arch.numHeads * headDim
        let kvDim = arch.numFullKVHeads * headDim
        let rotaryDim = Int(Double(headDim) * arch.partialRotaryFactor)
        let scale = Float(arch.attentionScale)
        let theta = Float(arch.fullRopeTheta)
        let eps: Float = 1e-6
        func matvec(_ name: String, rows: Int, x: [Float16]) throws -> [Float] {
            let weights = try Self.decoded(model, name: name).map(Self.bf16)
            let xf = x.map(Float.init)
            var result = [Float](repeating: 0, count: rows)
            for r in 0..<rows {
                var acc: Float = 0
                let base = r * hidden
                for n in 0..<hidden { acc += weights[base + n] * xf[n] }
                result[r] = Float(Float16(acc))
            }
            return result
        }
        let qNormW = try Self.decoded(model, name: "blk.\(layer).attn_q_norm.weight").map(Self.bf16)
        let kNormW = try Self.decoded(model, name: "blk.\(layer).attn_k_norm.weight").map(Self.bf16)
        var qs: [[Float]] = [], ks: [[Float]] = [], vs: [[Float]] = [], gates: [[Float]] = []
        for x in [x0, x1] {
            let packed = try matvec("blk.\(layer).attn_q.weight", rows: 2 * qDim, x: x)
            var q = [Float](repeating: 0, count: qDim)
            var gate = [Float](repeating: 0, count: qDim)
            for head in 0..<arch.numHeads {
                for i in 0..<headDim {
                    q[head * headDim + i] = packed[head * 2 * headDim + i]
                    gate[head * headDim + i] = packed[head * 2 * headDim + headDim + i]
                }
            }
            var k = try matvec("blk.\(layer).attn_k.weight", rows: kvDim, x: x)
            let v = try matvec("blk.\(layer).attn_v.weight", rows: kvDim, x: x)
            // per-head norm
            for head in 0..<arch.numHeads {
                var square: Float = 0
                for i in 0..<headDim { square += q[head * headDim + i] * q[head * headDim + i] }
                let invRms = 1 / (square / Float(headDim) + eps).squareRoot()
                for i in 0..<headDim {
                    q[head * headDim + i] = Float(
                        Float16(q[head * headDim + i] * invRms * qNormW[i]))
                }
            }
            for head in 0..<arch.numFullKVHeads {
                var square: Float = 0
                for i in 0..<headDim { square += k[head * headDim + i] * k[head * headDim + i] }
                let invRms = 1 / (square / Float(headDim) + eps).squareRoot()
                for i in 0..<headDim {
                    k[head * headDim + i] = Float(
                        Float16(k[head * headDim + i] * invRms * kNormW[i]))
                }
            }
            qs.append(q); ks.append(k); vs.append(v); gates.append(gate)
        }
        // rope over the first rotaryDim of each head, NeoX pairing
        func rope(_ data: inout [Float], heads: Int, position: Int) {
            let halfRotary = rotaryDim / 2
            for head in 0..<heads {
                let base = head * headDim
                for pair in 0..<halfRotary {
                    let angle = Float(position) * pow(theta, -2 * Float(pair) / Float(rotaryDim))
                    let c = cos(angle), s = sin(angle)
                    let lower = base + pair
                    let upper = base + halfRotary + pair
                    let x0 = data[lower], x1 = data[upper]
                    data[lower] = Float(Float16(x0 * c - x1 * s))
                    data[upper] = Float(Float16(x0 * s + x1 * c))
                }
            }
        }
        rope(&qs[0], heads: arch.numHeads, position: 0)
        rope(&qs[1], heads: arch.numHeads, position: 1)
        rope(&ks[0], heads: arch.numFullKVHeads, position: 0)
        rope(&ks[1], heads: arch.numFullKVHeads, position: 1)

        // attention for position 1 over both keys, then gate and o_proj
        let group = arch.numHeads / arch.numFullKVHeads
        var attn = [Float](repeating: 0, count: qDim)
        for head in 0..<arch.numHeads {
            let kvHead = head / group
            var scores = [Float](repeating: 0, count: 2)
            for t in 0..<2 {
                var acc: Float = 0
                for i in 0..<headDim {
                    acc += qs[1][head * headDim + i] * ks[t][kvHead * headDim + i]
                }
                scores[t] = acc * scale
            }
            let maximum = scores.max() ?? 0
            var sum: Float = 0
            for t in 0..<2 { scores[t] = exp(scores[t] - maximum); sum += scores[t] }
            for t in 0..<2 { scores[t] /= sum }
            for i in 0..<headDim {
                var acc: Float = 0
                for t in 0..<2 { acc += scores[t] * vs[t][kvHead * headDim + i] }
                let gated = Float(Float16(acc))
                let gate = Float(Float16(1 / (1 + exp(-gates[1][head * headDim + i]))))
                attn[head * headDim + i] = Float(Float16(gated * gate))
            }
        }
        let oW = try Self.decoded(model, name: "blk.\(layer).attn_output.weight").map(Self.bf16)
        var reference = [Float](repeating: 0, count: hidden)
        for r in 0..<hidden {
            var acc: Float = 0
            let base = r * qDim
            for n in 0..<qDim { acc += oW[base + n] * attn[n] }
            reference[r] = Float(Float16(acc))
        }
        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for i in 0..<hidden {
            let got = Double(Float(Float16(bitPattern: pointer[i])))
            let diff = got - Double(reference[i])
            sqDiff += diff * diff
            sqRef += Double(reference[i] * reference[i])
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("full attention token-1 RMS relative error: \(relative)")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }
}
