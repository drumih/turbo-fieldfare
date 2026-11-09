import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Weights read straight out of the mapped shards: a fused GEMV over a mapped
/// Q6_K tensor must agree with the CPU, and nothing is copied to make it so.
@Suite struct GGUFMappedShardsTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func mappedWeightsFeedTheFusedGEMV() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let mapped = try GGUFMappedShards(model: model, device: context.device)
        print("mapped \(String(format: "%.1f", Double(mapped.mappedBytes) / 1e9)) GB (virtual)")

        let name = "blk.0.attn_qkv.weight"
        let weight = try #require(try mapped.weight(named: name))
        #expect(weight.type == .q6K)
        #expect(weight.rows == 10_240)
        #expect(weight.columns == 2_560)

        let rows = 256
        var rng = SeedTree(0x3A9F).key("mapped-x")
        let x = (0..<weight.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer,
            let y = context.device.makeBuffer(length: rows * 2, options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("mapped GEMV buffers") }
        let gemv = try GGUFExpertGEMV(context: context)
        try gemv.encode(
            commandBuffer: cb, weights: weight.buffer, weightsOffset: weight.offset,
            x: xBuffer, y: y, rows: rows, columns: weight.columns, type: weight.type)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        // CPU: the same rows decoded from the file.
        let location = try #require(model.tensor(named: name))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        let rowBytes = (weight.columns / 256) * 210
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(try handle.read(upToCount: rows * rowBytes))
        let values = try GGUFDequant.decode(type: .q6K, bytes: [UInt8](data))
        var sqDiff: Double = 0
        var sqRef: Double = 0
        let pointer = y.contents().assumingMemoryBound(to: UInt16.self)
        let xf = x.map(Float.init)
        for row in 0..<rows {
            var acc: Float = 0
            let base = row * weight.columns
            for n in 0..<weight.columns { acc += values[base + n] * xf[n] }
            let reference = Float(Float16(acc))
            let got = Float(Float16(bitPattern: pointer[row]))
            let diff = Double(got - reference)
            sqDiff += diff * diff
            sqRef += Double(reference * reference)
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("mapped GEMV RMS relative error over \(rows) rows: \(relative)")
        #expect(relative < 5e-3)
    }
}
