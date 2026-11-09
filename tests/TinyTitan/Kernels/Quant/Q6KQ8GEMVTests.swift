import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Q6_K and Q8_0 fused GEMV on the real shared-expert matrices.
@Suite struct Q6KQ8GEMVTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private struct Matrix {
        let bytes: [UInt8]
        let rows: Int
        let columns: Int
        let values: [Float]
        let type: GGUFType
    }

    private static func load(_ model: GGUFModel, name: String, rows: Int, columns: Int) throws
        -> Matrix
    {
        let location = try #require(model.tensor(named: name))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
        return Matrix(
            bytes: [UInt8](data), rows: rows, columns: columns,
            values: try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](data)),
            type: location.tensor.type)
    }

    private static func run(
        _ context: MetalContext, matrix: Matrix, x: [Float16], iterations: Int
    ) throws -> (values: [Float16], seconds: Double) {
        let encode: (MTLCommandBuffer, MTLBuffer, MTLBuffer, MTLBuffer) throws -> Void
        if matrix.type == .q6K {
            let kernel = try Q6KGEMV(context: context)
            encode = { cb, w, xx, y in
                try kernel.encode(
                    commandBuffer: cb, weights: w, x: xx, y: y,
                    rows: matrix.rows, columns: matrix.columns)
            }
        } else {
            let kernel = try Q80GEMV(context: context)
            encode = { cb, w, xx, y in
                try kernel.encode(
                    commandBuffer: cb, weights: w, x: xx, y: y,
                    rows: matrix.rows, columns: matrix.columns)
            }
        }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard
            let w = context.device.makeBuffer(
                bytes: matrix.bytes, length: matrix.bytes.count, options: .storageModeShared),
            let xBuffer,
            let y = context.device.makeBuffer(
                length: matrix.rows * MemoryLayout<UInt16>.size, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("GEMV buffers") }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard let cb = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try encode(cb, w, xBuffer, y)
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw MetalError.invalidState("\(error)") }
        }
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        let pointer = y.contents().assumingMemoryBound(to: UInt16.self)
        return ((0..<matrix.rows).map { Float16(bitPattern: pointer[$0]) }, seconds)
    }

    private static func compare(
        _ matrix: Matrix, x: [Float16], gpu: [Float16]
    ) -> Double {
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for row in 0..<matrix.rows {
            var ref: Float = 0
            let base = row * matrix.columns
            for n in 0..<matrix.columns { ref += matrix.values[base + n] * Float(x[n]) }
            let refHalf = Float(Float16(ref))
            let diff = Double(Float(gpu[row]) - refHalf)
            sqDiff += diff * diff
            sqRef += Double(refHalf * refHalf)
        }
        return (sqDiff / max(sqRef, 1e-12)).squareRoot()
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func q6kGEMVMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.load(
            model, name: "blk.0.ffn_gate_shexp.weight", rows: 640, columns: 2_560)
        try #require(matrix.type == .q6K)
        var rng = SeedTree(0x66).key("q6k-gemv-x")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let (gpu, _) = try Self.run(context, matrix: matrix, x: x, iterations: 1)
        let relative = Self.compare(matrix, x: x, gpu: gpu)
        print("Q6_K GEMV RMS relative error: \(relative)")
        #expect(relative < 5e-3)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func q8GEMVMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let matrix = try Self.load(
            model, name: "blk.0.ffn_down_shexp.weight", rows: 2_560, columns: 640)
        try #require(matrix.type == .q8_0)
        var rng = SeedTree(0x88).key("q8-gemv-x")
        let x = (0..<matrix.columns).map { _ in Float16(rng.uniform(-1, 1)) }
        let context = try MetalContext()
        let (gpu, _) = try Self.run(context, matrix: matrix, x: x, iterations: 1)
        let relative = Self.compare(matrix, x: x, gpu: gpu)
        print("Q8_0 GEMV RMS relative error: \(relative)")
        #expect(relative < 5e-3)
    }
}
