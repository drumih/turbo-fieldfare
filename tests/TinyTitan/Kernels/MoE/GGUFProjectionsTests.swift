import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Every projection shape the model's non-MoE stages start from, run through
/// the arena's bf16 views with the engine's BF16GEMV: the GDN qkv/gate/alpha/
/// beta and the full-attention q/k/v/output.
@Suite struct GGUFProjectionsTests {
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
    func projectionsMatchTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let hidden = 2_560
        // (name, rows=out, columns=in = hidden)
        let shapes: [(String, Int)] = [
            ("blk.0.attn_qkv.weight", 10_240),
            ("blk.0.attn_gate.weight", 6_144),
            ("blk.0.ssm_alpha.weight", 48),
            ("blk.0.ssm_beta.weight", 48),
            ("blk.3.attn_q.weight", 12_288),
            ("blk.3.attn_k.weight", 512),
            ("blk.3.attn_v.weight", 512),
            ("blk.3.attn_output.weight", 2_560),
        ]
        let names = shapes.map(\.0)
        let arena = try GGUFResidentArena.build(names: names, from: model, context: context)
        let gemv = try BF16GEMV(context: context)

        var rng = SeedTree(0x9A9A).key("projection-x")
        let x = (0..<hidden).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer else { throw MetalError.bufferAllocationFailed("projection x") }

        for (name, rows) in shapes {
            let entry = try #require(arena.entries[name])
            guard
                let y = context.device.makeBuffer(
                    length: rows * 2, options: .storageModeShared),
                let cb = context.queue.makeCommandBuffer()
            else { throw MetalError.bufferAllocationFailed("projection y") }
            try gemv.encode(
                commandBuffer: cb, weights: arena.buffer, weightsOffset: Int(entry.offset),
                x: xBuffer, y: y, m: UInt32(rows), n: UInt32(hidden))
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw MetalError.invalidState("\(error)") }

            let weights = try Self.decoded(model, name: name)
            let xf = x.map(Float.init)
            let pointer = y.contents().assumingMemoryBound(to: UInt16.self)
            var sqDiff: Double = 0
            var sqRef: Double = 0
            for row in 0..<rows {
                var acc: Float = 0
                let base = row * hidden
                for n in 0..<hidden { acc += Self.bf16(weights[base + n]) * xf[n] }
                let reference = Float(Float16(acc))
                let got = Float(Float16(bitPattern: pointer[row]))
                let diff = Double(got - reference)
                sqDiff += diff * diff
                sqRef += Double(reference * reference)
            }
            let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
            print("\(name) [\(rows)x\(hidden)] RMS: \(relative)")
            #expect(relative < 5e-3, "\(name) RMS \(relative)")
        }
    }
}
