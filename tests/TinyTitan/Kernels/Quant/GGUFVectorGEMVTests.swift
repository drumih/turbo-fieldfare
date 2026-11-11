import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The vectorised Q6_K and Q8_0 GEMVs (`gguf_gemv_vec.metal`) against the CPU
/// decoder, which is bit-exact with ggml, on random blocks: every scale, high
/// bit and sign pattern is exercised, not only those a real matrix happens to
/// hold. Shapes are the decoder's own (GDN qkv 2,560 wide; HC up 320 wide).
@Suite struct GGUFVectorGEMVTests {
    private struct Generator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 16
        }
        mutating func byte() -> UInt8 { UInt8(truncatingIfNeeded: next()) }
        mutating func unit() -> Float { Float(next() % 2_000_001) / 1_000_000 - 1 }
    }

    /// Random blocks with a sane fp16 scale at `scaleOffset` in each block.
    private static func blocks(
        count: Int, bytes: Int, scaleOffset: Int, generator: inout Generator
    ) -> [UInt8] {
        var out = (0..<(count * bytes)).map { _ in generator.byte() }
        for block in 0..<count {
            let d = Float16(0.001 + 0.01 * abs(generator.unit())).bitPattern
            out[block * bytes + scaleOffset] = UInt8(truncatingIfNeeded: d)
            out[block * bytes + scaleOffset + 1] = UInt8(truncatingIfNeeded: d >> 8)
        }
        return out
    }

    private static func check(
        type: GGUFType, rows: Int, columns: Int, blockValues: Int, blockBytes: Int,
        scaleOffset: Int
    ) throws -> Double {
        var generator = Generator(state: UInt64(rows * 31 + columns))
        let weights = blocks(
            count: rows * columns / blockValues, bytes: blockBytes,
            scaleOffset: scaleOffset, generator: &generator)
        let x = (0..<columns).map { _ in Float16(generator.unit()) }
        let context = try MetalContext()
        let device = context.device
        let w = try #require(
            device.makeBuffer(bytes: weights, length: weights.count, options: .storageModeShared))
        let xb = try #require(
            device.makeBuffer(
                bytes: x.map(\.bitPattern), length: columns * 2, options: .storageModeShared))
        let y = try #require(device.makeBuffer(length: rows * 2, options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        if type == .q6K {
            try Q6KGEMV(context: context).encode(
                commandBuffer: commandBuffer, weights: w, x: xb, y: y, rows: rows,
                columns: columns)
        } else {
            try Q80GEMV(context: context).encode(
                commandBuffer: commandBuffer, weights: w, x: xb, y: y, rows: rows,
                columns: columns)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw MetalError.invalidState("\(error)") }

        let values = try GGUFDequant.decode(type: type, bytes: weights)
        let out = y.contents().assumingMemoryBound(to: Float16.self)
        var squaredError = 0.0
        var squaredReference = 0.0
        for row in 0..<rows {
            var reference = 0.0
            for column in 0..<columns {
                reference += Double(values[row * columns + column]) * Double(Float(x[column]))
            }
            let difference = Double(Float(out[row])) - reference
            squaredError += difference * difference
            squaredReference += reference * reference
        }
        return (squaredError / max(squaredReference, 1e-30)).squareRoot()
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func q6KMatchesTheCPUDecoder() throws {
        let error = try Self.check(
            type: .q6K, rows: 96, columns: 2_560, blockValues: 256, blockBytes: 210,
            scaleOffset: 208)
        // Half output rounding is ~5e-4 relative; the old per-byte kernel was
        // held to 5e-3 on a real matrix.
        #expect(error < 1e-3, "RMS relative error \(error)")
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func q8_0MatchesTheCPUDecoderOnWideAndNarrowRows() throws {
        for columns in [2_560, 320] {
            let error = try Self.check(
                type: .q8_0, rows: 72, columns: columns, blockValues: 32, blockBytes: 34,
                scaleOffset: 0)
            #expect(error < 1e-3, "\(columns) columns: RMS relative error \(error)")
        }
    }
}
