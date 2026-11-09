import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// The model's input and output path on real Q6_K tensors: the embedding gather
/// and the head GEMV.
@Suite struct GGUFEmbeddingHeadTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static func bytes(
        _ model: GGUFModel, name: String, offset: Int = 0, count: Int
    ) throws -> [UInt8] {
        let location = try #require(model.tensor(named: name))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset + UInt64(offset))
        let data = try #require(try handle.read(upToCount: count))
        return [UInt8](data)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func embeddingGatherMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let gather = try GGUFEmbeddingGather(context: context)
        let ids: [UInt32] = [0, 1, 100, 1_234, 50_000, 248_319]
        let rowBytes = (2_560 / 256) * 210
        // One contiguous window would be huge; read each row's bytes separately
        // and stitch a compact table whose row index is the array index.
        var table: [UInt8] = []
        for id in ids {
            table += try Self.bytes(
                model, name: "token_embd.weight", offset: Int(id) * rowBytes, count: rowBytes)
        }
        let indices = (0..<UInt32(ids.count)).map { $0 }
        guard
            let tableBuffer = context.device.makeBuffer(
                bytes: table, length: table.count, options: .storageModeShared),
            let rowBuffer = indices.withUnsafeBytes({ raw in
                guard let base = raw.baseAddress else { return nil as MTLBuffer? }
                return context.device.makeBuffer(
                    bytes: base, length: raw.count, options: .storageModeShared)
            }),
            let out = context.device.makeBuffer(
                length: ids.count * 2_560 * 2, options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("embedding buffers") }
        try gather.encode(
            commandBuffer: cb, table: tableBuffer, rows: rowBuffer,
            count: ids.count, out: out)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        let reference = try GGUFDequant.decode(type: .q6K, bytes: table)
        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        var mismatches = 0
        for index in 0..<(ids.count * 2_560)
        where pointer[index] != Quantization.bf16Bits(reference[index]) {
            mismatches += 1
        }
        print("embedding gather mismatches: \(mismatches)/\(ids.count * 2_560)")
        #expect(mismatches == 0)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func headGEMVMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let rows = 512
        let rowBytes = (2_560 / 256) * 210
        let head = try Self.bytes(
            model, name: "output.weight", count: rows * rowBytes)
        var rng = SeedTree(0x4EAD).key("head-x")
        let x = (0..<2_560).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let w = context.device.makeBuffer(
                bytes: head, length: head.count, options: .storageModeShared),
            let xBuffer,
            let y = context.device.makeBuffer(
                length: rows * 2, options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("head buffers") }
        let kernel = try Q6KGEMV(context: context)
        try kernel.encode(
            commandBuffer: cb, weights: w, x: xBuffer, y: y,
            rows: rows, columns: 2_560)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }
        let values = try GGUFDequant.decode(type: .q6K, bytes: head)
        let pointer = y.contents().assumingMemoryBound(to: UInt16.self)
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for row in 0..<rows {
            var acc: Float = 0
            let base = row * 2_560
            for n in 0..<2_560 { acc += values[base + n] * Float(x[n]) }
            let ref = Float(Float16(acc))
            let diff = Double(Float(Float16(bitPattern: pointer[row])) - ref)
            sqDiff += diff * diff
            sqRef += Double(ref * ref)
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("head GEMV RMS relative error over \(rows) rows: \(relative)")
        #expect(relative < 5e-3)
    }
}
